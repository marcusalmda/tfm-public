"""Passo A — modelo sequencial causal com objetivo autorregressivo (baseline estilo nuFormer).

Encoder Transformer com mascara causal, SEM grafo. A cada posicao preve o proximo
evento (tipo, via cross-entropy; delta_t, via regressao). Para o downstream, usa a
representacao da ultima posicao valida da sequencia (convencao GPT), em vez do
token [CLS] bidirecional do TFM.

Desacoplado do TFM: nao importa GNNEncoder nem grafo. Reusa apenas a ordem de
features (mesma do pipeline) para compatibilidade dos dados.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S, d_model)
        return x + self.pe[:, : x.size(1)]


class CausalEventModel(nn.Module):
    """Transformer causal para sequencias de eventos.

    Args:
      cat_facets: lista (nome, vocab_size, emb_dim) na ordem em que as colunas
                  categoricas aparecem em x_cat.
      num_continuous: numero de colunas continuas em x_cont.
      num_event_types: numero de classes do alvo autorregressivo (tipo do evento).
    """

    def __init__(self, cat_facets: List[tuple], num_continuous: int,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 max_seq_len: int = 1024, num_event_types: int = 16,
                 gnn_config: Optional[dict] = None,
                 gnn_node_types: Optional[List[str]] = None,
                 gnn_edge_types: Optional[List[tuple]] = None,
                 gnn_input_feature_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.d_model = d_model
        self.cat_names = [f[0] for f in cat_facets]
        self.num_continuous = num_continuous

        # Encoder de grafo opcional (HGT do Eduardo, importado sem modificacao).
        # Quando presente, o embedding do documento entra como conditioning na
        # posicao 0 da sequencia (ver encode(prefix=...)).
        self.gnn_encoder = None
        if gnn_config is not None:
            from src.models.gnn_encoder import GNNEncoder
            assert gnn_config.get("out_channels", d_model) == d_model, \
                "GNN out_channels deve ser igual a d_model (embedding do grafo vira token)."
            self.gnn_encoder = GNNEncoder(
                node_types=gnn_node_types,
                edge_types_tuples=gnn_edge_types,
                hidden_channels=gnn_config["hidden_channels"],
                out_channels=d_model,
                num_layers=gnn_config.get("num_layers", 2),
                gnn_type=gnn_config.get("type", "HGTConv"),
                num_heads=gnn_config.get("num_heads", 4),
                dropout_rate=gnn_config.get("dropout", 0.1),
                input_feature_dims=gnn_input_feature_dims,
            )

        self.embeddings = nn.ModuleDict()
        emb_sum = 0
        for name, vocab, dim in cat_facets:
            self.embeddings[name] = nn.Embedding(vocab, dim, padding_idx=0)
            emb_sum += dim
        in_dim = emb_sum + num_continuous
        self.input_projection = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()

        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len + 1)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        # Cabecas autorregressivas de EVENTO COMPLETO (simetricas ao MEM do TFM):
        # uma cabeca por atributo. next_heads[nome] preve o atributo 'nome' do
        # proximo evento. feature_targets descreve cada alvo:
        #   (nome, tipo, coluna_no_tensor)  onde tipo in {'cat','cont'}
        # Preenchido por set_feature_targets() antes do treino.
        self.next_heads = nn.ModuleDict()
        self.feature_targets = []  # lista de (nome, tipo, col, vocab_or_None)

        # Cabeca de classificacao downstream (anexada sob demanda)
        self.classification_head: Optional[nn.Module] = None

    def set_feature_targets(self, targets):
        """targets: lista de dicts {name, kind ('cat'|'cont'), col, vocab}.
        Cria uma cabeca linear por atributo (Linear->vocab p/ cat, Linear->1 p/ cont)."""
        self.feature_targets = targets
        for t in targets:
            out_dim = t["vocab"] if t["kind"] == "cat" else 1
            self.next_heads[t["name"]] = nn.Linear(self.d_model, out_dim)

    # ------------------------------------------------------------------
    def _project(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """So a projecao dos eventos para d_model (sem pos-encoding/norm)."""
        parts = [self.embeddings[name](x_cat[:, :, i]) for i, name in enumerate(self.cat_names)]
        if self.num_continuous > 0:
            parts.append(x_cont)
        return self.input_projection(torch.cat(parts, dim=-1))

    def _finalize(self, h: torch.Tensor) -> torch.Tensor:
        """Pos-encoding + norm + dropout, aplicado a sequencia ja montada (com prefix)."""
        return self.dropout(self.input_norm(self.pos_encoder(h)))

    @staticmethod
    def _causal_mask(seq_len: int, device) -> torch.Tensor:
        # True nas posicoes proibidas (futuro). nn.TransformerEncoder soma -inf.
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, x_cat, x_cont, padding_mask=None, causal: bool = True,
               prefix: Optional[torch.Tensor] = None):
        """causal=True -> mascara triangular (AR). causal=False -> bidirecional (MEM).
        prefix (B, d_model): conditioning (ex.: embedding do grafo) inserido na
        POSICAO 0. Sob mascara causal ele so atende a si mesmo; os eventos atendem
        de volta -> veem grafo + passado.
        Retorna (saida (B, S, d), pad_mask_ajustada, n_prefixo) onde S = L(+1) e
        n_prefixo = 1 se ha prefix, 0 caso contrario (para a indexacao downstream)."""
        h = self._project(x_cat, x_cont)                    # (B, L, d)
        n_prefix = 0
        if prefix is not None:
            h = torch.cat([prefix.unsqueeze(1), h], dim=1)  # (B, L+1, d)
            n_prefix = 1
            if padding_mask is not None:
                pad0 = torch.zeros((h.size(0), 1), dtype=torch.bool, device=h.device)
                padding_mask = torch.cat([pad0, padding_mask], dim=1)
        h = self._finalize(h)
        mask = self._causal_mask(h.size(1), h.device) if causal else None
        out = self.encoder(h, mask=mask, src_key_padding_mask=padding_mask)
        return out, padding_mask, n_prefix

    def graph_embed(self, graph_data, doc_indices: torch.Tensor) -> torch.Tensor:
        """Roda o HGT sobre o grafo (full-batch) e devolve o embedding (B, d_model)
        dos documentos do batch. Requer gnn_encoder configurado."""
        if self.gnn_encoder is None:
            raise RuntimeError("Modelo criado sem gnn_config; graph_embed indisponivel.")
        node_emb = self.gnn_encoder(graph_data)          # dict por tipo de no
        doc_emb = node_emb.get("Document")
        if doc_emb is None:
            raise RuntimeError(f"GNN nao retornou 'Document'. Chaves: {list(node_emb.keys())}")
        return doc_emb[doc_indices.to(doc_emb.device)]   # (B, d_model)

    # ------------------------------------------------------------------
    def apply_next_heads(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Aplica as cabecas de predicao a um tensor de estados ja coletado (M, d).
        Serve tanto para AR (predizer o proximo evento) quanto para MEM (reconstruir
        o evento mascarado) — as cabecas sao as mesmas; muda so a posicao onde se
        aplica (proxima posicao no AR, posicao mascarada no MEM) e a mascara de
        atencao (causal no AR, bidirecional no MEM)."""
        return {name: head(hidden) for name, head in self.next_heads.items()}

    def forward_classify(self, x_cat, x_cont, padding_mask=None, causal: bool = True,
                         prefix: Optional[torch.Tensor] = None,
                         readout: str = "auto") -> torch.Tensor:
        """readout controla a agregacao para classificacao:
          'auto' -> ultima posicao se causal (GPT), media se bidirecional (BERT);
          'last' -> forca ultima posicao valida;
          'mean' -> forca media das posicoes validas (mais robusto a drift temporal,
                    mesmo sob mascara causal — cada posicao ja viu seu passado + grafo).
        """
        if self.classification_head is None:
            raise RuntimeError("attach_classification_head nao foi chamado.")
        out, pad_adj, _ = self.encode(x_cat, x_cont, padding_mask, causal=causal, prefix=prefix)
        if readout == "last":
            rep = self._last_valid_repr(out, pad_adj)
        elif readout == "mean":
            rep = self._mean_valid_repr(out, pad_adj)
        else:  # auto
            rep = self._last_valid_repr(out, pad_adj) if causal else self._mean_valid_repr(out, pad_adj)
        return self.classification_head(rep)

    def _mean_valid_repr(self, out: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if padding_mask is None:
            return out.mean(dim=1)
        m = (~padding_mask).float().unsqueeze(-1)  # (B, S, 1)
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)

    def attach_classification_head(self, num_classes: int = 2):
        self.classification_head = nn.Linear(self.d_model, num_classes)

    def _last_valid_repr(self, out: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Representacao da ultima posicao nao-pad (convencao autorregressiva)."""
        if padding_mask is None:
            return out[:, -1, :]
        lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1  # indice do ultimo valido
        idx = lengths.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        return out.gather(1, idx).squeeze(1)

