import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List

class PositionalEncoding(nn.Module):
    """Standard Positional Encoding. Assumes input shape (seq_len, batch_size, d_model)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (d_model/2)

        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) # (max_len, 1, d_model) - to be broadcastable if batch_size > 1

        self.register_buffer('pe', pe, persistent=False) # Not a parameter, but should be part of state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # self.pe is (max_len, 1, d_model). We need to match x's seq_len.
        # x is (S, B, D). self.pe[:x.size(0)] will be (S, 1, D)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EventEmbedder(nn.Module):
    """Embeds categorical event facets plus continuous event features into d_model."""
    def __init__(self,
                 d_model: int,
                 event_facet_vocab_sizes: Optional[Dict[str, int]] = None,
                 event_facet_embedding_dims: Optional[Dict[str, int]] = None,
                 num_continuous_features: int = 0,
                 categorical_feature_order: Optional[List[str]] = None,
                 max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.categorical_facet_names = categorical_feature_order if categorical_feature_order else []
        if not self.categorical_facet_names and event_facet_vocab_sizes:
            self.categorical_facet_names = list(event_facet_vocab_sizes.keys())

        self.facet_embeddings = nn.ModuleDict()
        input_dim = 0
        if event_facet_vocab_sizes and event_facet_embedding_dims:
            for facet_name in self.categorical_facet_names:
                vocab_size = event_facet_vocab_sizes.get(facet_name)
                embedding_dim = event_facet_embedding_dims.get(facet_name)
                if vocab_size is None or embedding_dim is None:
                    raise ValueError(f"Vocab size or embedding dim for facet '{facet_name}' not specified correctly.")
                self.facet_embeddings[facet_name] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                input_dim += embedding_dim

        self.num_categorical_facets = len(self.categorical_facet_names)
        self.num_continuous_features = num_continuous_features
        input_dim += num_continuous_features
        if input_dim <= 0:
            raise ValueError("EventEmbedder needs at least one categorical or continuous feature.")
        self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        if x_cat.shape[-1] != self.num_categorical_facets:
            raise ValueError(
                f"Input x_cat last dimension ({x_cat.shape[-1]}) does not match "
                f"declared categorical facets ({self.num_categorical_facets})."
            )
        if x_cont.shape[-1] != self.num_continuous_features:
            raise ValueError(
                f"Input x_cont last dimension ({x_cont.shape[-1]}) does not match "
                f"declared continuous features ({self.num_continuous_features})."
            )

        parts: List[torch.Tensor] = []
        for i, facet_name in enumerate(self.categorical_facet_names):
            parts.append(self.facet_embeddings[facet_name](x_cat[:, :, i]))
        if self.num_continuous_features > 0:
            parts.append(x_cont)
        return self.input_projection(torch.cat(parts, dim=-1))

class TransactionTransformer(nn.Module):
    """
    Transformer Encoder for processing sequences of transactional events.
    Each event is a collection of features (facets).
    Uses batch_first=True for nn.TransformerEncoder.
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 event_facet_vocab_sizes: Optional[Dict[str, int]] = None,
                 event_facet_embedding_dims: Optional[Dict[str, int]] = None,
                 num_continuous_features: int = 0,
                 categorical_feature_order: Optional[List[str]] = None, # Explicit order of categorical features in input tensor
                 max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model

        self.facet_embeddings = nn.ModuleDict()
        current_embedding_dim_sum = 0

        self.categorical_facet_names: List[str] = categorical_feature_order if categorical_feature_order else []
        if not self.categorical_facet_names and event_facet_vocab_sizes: # Infer from vocab_sizes if not given
             self.categorical_facet_names = list(event_facet_vocab_sizes.keys())


        if event_facet_vocab_sizes and event_facet_embedding_dims:
            for facet_name in self.categorical_facet_names: # Iterate in specified order
                vocab_size = event_facet_vocab_sizes.get(facet_name)
                embedding_dim = event_facet_embedding_dims.get(facet_name)
                if vocab_size is None or embedding_dim is None:
                    raise ValueError(f"Vocab size or embedding dim for facet '{facet_name}' not specified correctly.")
                self.facet_embeddings[facet_name] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # Assuming 0 is padding_idx for facets
                current_embedding_dim_sum += embedding_dim

        current_embedding_dim_sum += num_continuous_features
        self.num_categorical_facets = len(self.categorical_facet_names)
        self.num_continuous_features = num_continuous_features


        if current_embedding_dim_sum == 0 :
             raise ValueError("No features defined (neither categorical facets nor continuous features). Model would be empty.")

        if current_embedding_dim_sum != d_model:
            self.input_projection = nn.Linear(current_embedding_dim_sum, d_model)
            print(f"Projecting input event features from {current_embedding_dim_sum} to {d_model}")
        else:
            self.input_projection = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len + 1) # +1 for CLS token
        self.input_layer_norm = nn.LayerNorm(d_model) # Add LayerNorm
        self.dropout_layer = nn.Dropout(p=dropout) # Using self.dropout_layer for clarity if self.dropout is used elsewhere

        # TransformerEncoderLayer expects (S, B, E) or (B, S, E) if batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation=F.gelu)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self,
                x_cat: torch.Tensor, # (batch_size, seq_len, num_categorical_features), dtype=long
                x_cont: torch.Tensor, # (batch_size, seq_len, num_continuous_features), dtype=float
                cls_token_embedding: torch.Tensor, # (batch_size, d_model)
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor: # (batch_size, seq_len), True for pad

        batch_size, seq_len, num_cat_features_tensor = x_cat.shape
        _, _, num_cont_features_tensor = x_cont.shape

        if num_cat_features_tensor != self.num_categorical_facets:
            raise ValueError(f"Input x_cat last dimension ({num_cat_features_tensor}) "
                             f"does not match declared number of categorical facets ({self.num_categorical_facets}).")
        if num_cont_features_tensor != self.num_continuous_features:
            raise ValueError(f"Input x_cont last dimension ({num_cont_features_tensor}) "
                             f"does not match declared number of continuous features ({self.num_continuous_features}).")

        embedded_categorical_facets: List[torch.Tensor] = []

        # Process categorical features based on self.categorical_facet_names order
        # x_cat is already (batch, seq, num_cat_features_in_order)
        for i, facet_name in enumerate(self.categorical_facet_names):
            # Each slice x_cat[:, :, i] corresponds to the i-th categorical feature in self.categorical_facet_names
            facet_indices = x_cat[:, :, i] # Already long

            embedded_categorical_facets.append(self.facet_embeddings[facet_name](facet_indices))

        # Prepare list of all features to be concatenated
        all_features_to_combine = []
        all_features_to_combine.extend(embedded_categorical_facets)

        if self.num_continuous_features > 0:
            # x_cont is already the tensor of continuous features
            all_features_to_combine.append(x_cont)

        if not all_features_to_combine:
            # This case should ideally be prevented by initialization checks (current_embedding_dim_sum > 0)
            # If seq_len is 0, this might also lead to issues if not handled earlier (e.g. in dataset or collate)
            # For safety, if truly no features, create empty projected embeddings of correct d_model size.
             projected_event_embeddings = torch.empty((batch_size, seq_len, self.d_model), device=x_cat.device if x_cat.nelement() > 0 else (x_cont.device if x_cont.nelement() > 0 else cls_token_embedding.device))
        else:
            combined_event_embeddings = torch.cat(all_features_to_combine, dim=-1)
            projected_event_embeddings = self.input_projection(combined_event_embeddings)

        # Prepend [CLS] token: (batch_size, 1, d_model)
        cls_token_unsq = cls_token_embedding.unsqueeze(1)
        full_sequence_embeddings = torch.cat([cls_token_unsq, projected_event_embeddings], dim=1)
        # Now shape is (batch_size, seq_len + 1, d_model)

        # Add positional encoding: TransformerEncoder is batch_first=True, so input is (B, S, E)
        # PositionalEncoding class takes (S, B, E), so permute, apply, permute back.
        full_sequence_permuted = full_sequence_embeddings.permute(1, 0, 2) # (seq_len+1, batch_size, d_model)
        full_sequence_pos_encoded = self.pos_encoder(full_sequence_permuted) # PositionalEncoding includes dropout

        # Permute back to (batch_size, seq_len+1, d_model) before LayerNorm
        x_with_pe_batch_first = full_sequence_pos_encoded.permute(1, 0, 2)

        # Apply LayerNorm and another dropout (if desired, though PositionalEncoding already has one)
        # User's plan: "Apply input_layer_norm to the input tensor after positional encoding and dropout."
        # PositionalEncoding already applies dropout. If an additional dropout is desired here, it's fine.
        # The self.dropout_layer was added for this.
        # If the dropout in PositionalEncoding is sufficient, self.dropout_layer(x_with_pe_batch_first) can be skipped
        # and just use x_with_pe_batch_first for LayerNorm.
        # For now, let's assume the dropout after PE is the one referred to by the user.
        # The user instruction was: "Apply input_layer_norm to the input tensor after positional encoding and dropout."
        # PositionalEncoding's output is already "after positional encoding and dropout".

        x_stabilized = self.input_layer_norm(x_with_pe_batch_first)
        # If an additional dropout post-LayerNorm (or pre-LayerNorm but post-PE's dropout) is intended by self.dropout_layer:
        # x_stabilized = self.dropout_layer(x_stabilized) # Option: apply dropout after LayerNorm too

        full_sequence_input_to_transformer = x_stabilized # (batch_size, seq_len+1, d_model)


        final_padding_mask = None
        if padding_mask is not None:
            cls_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=padding_mask.device)
            final_padding_mask = torch.cat([cls_padding, padding_mask], dim=1) # (B, S+1)
            # TransformerEncoder expects True where values ARE NOT to be attended.
            # So if padding_mask is True for PAD, it's already correct for src_key_padding_mask.

        transformer_output = self.transformer_encoder(
            full_sequence_input_to_transformer,
            src_key_padding_mask=final_padding_mask
        )
        # Output shape: (batch_size, seq_len + 1, d_model)
        return transformer_output

if __name__ == '__main__':
    print("TransactionTransformer Example (Illustrative)")

    mock_d_model = 64
    mock_nhead = 4
    mock_num_layers = 2
    mock_dim_feedforward = 128
    mock_dropout = 0.1
    max_s_len = 50

    # Define facets and their properties
    # Order matters for slicing input tensor if categorical_feature_order is used.
    cat_feat_order = ['event_type', 'value_bin', 'hour']
    mock_facet_vocabs = {'event_type': 10, 'value_bin': 11, 'hour': 24, '<PAD>':0} # Ensure padding_idx=0 used in Embedding
    mock_facet_embs = {'event_type': 10, 'value_bin': 10, 'hour': 6} # Sum = 26
    mock_num_continuous = 2 # e.g. value_log_scaled, delta_t_log_scaled
    # Total raw concatenated dim = 26 + 2 = 28. This will be projected to d_model (64).

    transformer = TransactionTransformer(
        d_model=mock_d_model, nhead=mock_nhead, num_encoder_layers=mock_num_layers,
        dim_feedforward=mock_dim_feedforward, dropout=mock_dropout,
        event_facet_vocab_sizes=mock_facet_vocabs,
        event_facet_embedding_dims=mock_facet_embs,
        num_continuous_features=mock_num_continuous,
        categorical_feature_order=cat_feat_order,
        max_seq_len=max_s_len
    )
    print("\nTransactionTransformer instantiated.")

    batch_s = 4
    seq_l = 20 # Actual sequence length of events (excluding CLS)
    num_cat_feats = len(cat_feat_order)

    # Dummy input: (batch, seq, features_per_event)
    # features_per_event: cat_idx_event_type, cat_idx_value_bin, cat_idx_hour, cont_val1, cont_val2
    dummy_event_data = torch.rand(batch_s, seq_l, num_cat_feats + mock_num_continuous)
    dummy_event_data[:, :, 0] = torch.randint(1, mock_facet_vocabs['event_type'], (batch_s, seq_l)) # Avoid padding_idx 0 for real data
    dummy_event_data[:, :, 1] = torch.randint(1, mock_facet_vocabs['value_bin'], (batch_s, seq_l))
    dummy_event_data[:, :, 2] = torch.randint(1, mock_facet_vocabs['hour'], (batch_s, seq_l))

    dummy_cls = torch.randn(batch_s, mock_d_model)

    # Padding mask: True for padded tokens. Example: last 5 tokens are padded.
    dummy_pad_mask = torch.zeros(batch_s, seq_l, dtype=torch.bool)
    if seq_l > 5 : dummy_pad_mask[:, -5:] = True

    print(f"\nInput event data shape: {dummy_event_data.shape}")
    print(f"Input CLS shape: {dummy_cls.shape}")
    print(f"Input padding mask shape: {dummy_pad_mask.shape if dummy_pad_mask is not None else 'None'}")

    try:
        output = transformer(dummy_event_data, dummy_cls, dummy_pad_mask)
        print(f"Output shape: {output.shape}") # Expected (batch_s, seq_l + 1, d_model)
        assert output.shape == (batch_s, seq_l + 1, mock_d_model)

        # Test without padding mask
        output_no_mask = transformer(dummy_event_data, dummy_cls, None)
        print(f"Output shape (no mask): {output_no_mask.shape}")
        assert output_no_mask.shape == (batch_s, seq_l + 1, mock_d_model)

        print("\nForward pass successful.")
    except Exception as e:
        print(f"\nError during forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone with TransactionTransformer examples.")
