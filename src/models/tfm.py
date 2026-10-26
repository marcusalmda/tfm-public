import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple, List

# Need to ensure correct import paths if this file is run directly vs as part of a package
try:
    from .gnn_encoder import GNNEncoder
    from .transformer_encoder import TransactionTransformer
except ImportError: # Fallback for running script directly
    from gnn_encoder import GNNEncoder
    from transformer_encoder import TransactionTransformer

class MEMPredictionHeads(nn.Module):
    """
    Prediction heads for Masked Event Modeling (MEM).
    Takes hidden states corresponding to masked tokens and predicts the original values
    of their constituent features.
    """
    def __init__(self, d_model: int, feature_info: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.d_model = d_model
        self.feature_info = feature_info
        self.heads = nn.ModuleDict()

        for key, info in feature_info.items():
            if info['type'] == 'categorical':
                self.heads[key] = nn.Linear(d_model, info['vocab_size'])
            elif info['type'] == 'continuous':
                self.heads[key] = nn.Linear(d_model, 1) # Predict a single continuous value
            # else: type might be 'identifier' or other non-predictable, or already handled

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): Hidden states for masked tokens,
                                          shape (N_masked_tokens_in_batch, d_model).
        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are feature names (e.g., 'event_type_id')
                                     and values are the corresponding predictions (logits for categorical,
                                     raw values for continuous).
                                     Shape of each value tensor: (N_masked_tokens_in_batch, vocab_size) for categorical
                                                               (N_masked_tokens_in_batch, 1) for continuous
        """
        predictions = {}
        for key, head_layer in self.heads.items():
            predictions[key] = head_layer(hidden_states)
        return predictions


class TFM(nn.Module):
    """
    Transactional Foundation Model (TFM).
    Combines a GNNEncoder for graph-structured data and a TransactionTransformer
    for sequential event data.
    """
    def __init__(self,
                 gnn_config: Dict[str, Any],
                 transformer_config: Dict[str, Any],
                 gnn_input_feature_dims: Dict[str, int],
                 gnn_node_types: List[str],
                 gnn_edge_types_tuples: List[Tuple[str,str,str]],
                 event_facet_vocab_sizes: Dict[str, int], # Vocab sizes for each categorical event facet
                 event_facet_embedding_dims: Dict[str, int], # Embedding dims for each cat event facet
                 num_continuous_event_features: int, # Number of continuous features per event
                 categorical_event_feature_order: List[str], # Order of categorical features for EventEmbedder
                 # Added for MEM heads:
                 mem_feature_info: Optional[Dict[str, Dict[str, Any]]] = None # From TFMPretrainDataset.feature_info
                ):
        super().__init__()

        if gnn_config.get('out_channels') != transformer_config.get('d_model'):
            raise ValueError(
                f"GNN out_channels ({gnn_config.get('out_channels')}) must match "
                f"Transformer d_model ({transformer_config.get('d_model')}) "
                "for CLS token initialization."
            )

        self.d_model = transformer_config['d_model']

        self.gnn_encoder = GNNEncoder(
            node_types=gnn_node_types,
            edge_types_tuples=gnn_edge_types_tuples,
            hidden_channels=gnn_config['hidden_channels'],
            out_channels=gnn_config['out_channels'],
            num_layers=gnn_config['num_layers'],
            gnn_type=gnn_config.get('type', 'HGTConv'),
            num_heads=gnn_config.get('num_heads', 4),
            dropout_rate=gnn_config.get('dropout', 0.1),
            input_feature_dims=gnn_input_feature_dims
        )

        self.transformer_encoder = TransactionTransformer(
            d_model=transformer_config['d_model'],
            nhead=transformer_config['nhead'],
            num_encoder_layers=transformer_config['num_encoder_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout=transformer_config.get('dropout', 0.1),
            event_facet_vocab_sizes=event_facet_vocab_sizes,
            event_facet_embedding_dims=event_facet_embedding_dims,
            num_continuous_features=num_continuous_event_features,
            categorical_feature_order=categorical_event_feature_order,
            max_seq_len=transformer_config.get('max_seq_len', 256)
        )

        self.mem_heads = None
        if mem_feature_info:
            self.mem_heads = MEMPredictionHeads(self.d_model, mem_feature_info)

        self.classification_head = None
        self.d_model_transformer = transformer_config['d_model'] # Store for classification head


    def forward(self,
                graph_data: Any, # torch_geometric.data.HeteroData
                event_sequences_cat: torch.Tensor, # (batch_size, seq_len, num_categorical_features)
                event_sequences_cont: torch.Tensor, # (batch_size, seq_len, num_continuous_features)
                document_node_indices_in_graph: torch.Tensor, # (batch_size) GNN int indices for documents
                padding_mask: Optional[torch.Tensor] = None, # (batch_size, seq_len)
                # For MEM, we need to know which tokens were masked to select their hidden states
                masked_token_indices_in_sequence: Optional[torch.Tensor] = None # (total_masked_tokens_in_batch, 2) [batch_idx, seq_idx]
               ) -> Dict[str, torch.Tensor]:
        """
        Args:
            graph_data (HeteroData): The entire heterogeneous graph.
            event_sequences_cat (torch.Tensor): Batch of categorical event sequences.
            event_sequences_cont (torch.Tensor): Batch of continuous event sequences.
            document_node_indices_in_graph (torch.Tensor): GNN indices for documents in batch.
            padding_mask (Optional[torch.Tensor]): Padding mask for event_sequences.
            masked_token_indices_in_sequence (Optional[torch.Tensor]):
                If provided, indicates which tokens' hidden states to pass to MEM heads.
                Shape: (total_num_masked_tokens_in_batch, 2), where each row is [batch_index, sequence_index_of_masked_token].
                If None, MEM predictions are not computed.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                'transformer_output': Full output of the transformer (batch_size, seq_len+1, d_model).
                'cls_embedding': CLS token embedding from transformer (batch_size, d_model).
                'mem_predictions': (Optional) Dict of predictions from MEM heads.
                'classification_logits': (Optional) Logits from classification head.
        """
        outputs = {}

        gnn_output_embeddings: Dict[str, torch.Tensor] = self.gnn_encoder(graph_data)
        document_embeddings_all = gnn_output_embeddings.get('Document') # CORRECTED CASE
        if document_embeddings_all is None:
            available_keys = list(gnn_output_embeddings.keys()) # Improved error message
            raise RuntimeError(f"GNNEncoder did not return embeddings for 'Document' node type. Available keys: {available_keys}")

        doc_indices_device = document_node_indices_in_graph.to(document_embeddings_all.device)
        cls_token_gnn_embeddings = document_embeddings_all[doc_indices_device]
        outputs['cls_embedding_gnn'] = cls_token_gnn_embeddings # For contrastive loss

        transformer_full_output = self.transformer_encoder( # (batch_size, seq_len+1, d_model)
            x_cat=event_sequences_cat,
            x_cont=event_sequences_cont,
            cls_token_embedding=cls_token_gnn_embeddings, # Initialize CLS with GNN embedding
            padding_mask=padding_mask
        )
        outputs['transformer_output'] = transformer_full_output
        outputs['cls_embedding_transformer'] = transformer_full_output[:, 0, :] # CLS token from Transformer

        # MEM Predictions
        if self.mem_heads is not None and masked_token_indices_in_sequence is not None:
            # transformer_full_output is (batch, seq_len+1, dim)
            # We need hidden states for event tokens, so exclude CLS: transformer_full_output[:, 1:, :]
            event_token_hidden_states = transformer_full_output[:, 1:, :] # (batch, seq_len, dim)

            # Select the hidden states of the masked tokens
            # masked_token_indices_in_sequence provides [batch_idx, seq_idx]
            # We need to gather these from event_token_hidden_states
            batch_indices = masked_token_indices_in_sequence[:, 0]
            sequence_indices = masked_token_indices_in_sequence[:, 1]

            selected_hidden_states = event_token_hidden_states[batch_indices, sequence_indices, :] # (total_masked_tokens, d_model)

            outputs['mem_predictions'] = self.mem_heads(selected_hidden_states)

        # Classification
        if self.classification_head is not None:
            # Use the CLS token embedding from the Transformer for classification
            cls_output_for_classification = outputs['cls_embedding_transformer']
            outputs['classification_logits'] = self.classification_head(cls_output_for_classification)

        return outputs


    def attach_classification_head(self, num_classes: int, head_type: str = "linear"):
        if head_type == "linear":
            # Use the stored d_model from the transformer for the input dimension of the head
            self.classification_head = nn.Linear(self.d_model_transformer, num_classes)
            # Initialize weights for the new head (optional, but often good practice)
            # For example, Xavier uniform initialization:
            nn.init.xavier_uniform_(self.classification_head.weight)
            if self.classification_head.bias is not None:
                nn.init.zeros_(self.classification_head.bias)
            print(f"Attached and initialized a linear classification head: Linear({self.d_model_transformer}, {num_classes})")
        else:
            raise ValueError(f"Unsupported classification head type: {head_type}")

    def freeze_base_model(self, freeze_gnn: bool = True, freeze_transformer: bool = True):
        if freeze_gnn:
            if hasattr(self, 'gnn_encoder') and self.gnn_encoder is not None:
                for param in self.gnn_encoder.parameters():
                    param.requires_grad = False
                print("GNN encoder parameters frozen.")
            else:
                print("GNN encoder not found or not initialized, skipping GNN freeze.")

        if freeze_transformer:
            if hasattr(self, 'transformer_encoder') and self.transformer_encoder is not None:
                for param in self.transformer_encoder.parameters():
                    param.requires_grad = False
                # Also freeze the MEM heads if they exist, as they are part of the base model from pre-training
                if hasattr(self, 'mem_heads') and self.mem_heads is not None:
                    for param in self.mem_heads.parameters():
                        param.requires_grad = False
                    print("MEM prediction heads frozen.")
                print("Transformer encoder parameters frozen.")
            else:
                print("Transformer encoder not found or not initialized, skipping Transformer freeze.")

        # Ensure the classification head (if attached) remains trainable
        if hasattr(self, 'classification_head') and self.classification_head is not None:
            for param in self.classification_head.parameters():
                param.requires_grad = True
            print("Classification head parameters remain trainable.")


if __name__ == '__main__':
    print("TFM Example (Illustrative)")

    # Import HeteroData for the example if PyG is available
    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        print("torch_geometric.data.HeteroData not found. Skipping TFM __main__ example.")
        HeteroData = None # To prevent NameError if PyG is not fully installed

    if HeteroData:
        mock_gnn_config = {'hidden_channels': 64, 'out_channels': 32, 'num_layers': 2, 'type': 'HGTConv', 'num_heads': 2, 'dropout': 0.1}
        mock_transformer_config = {'d_model': 32, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1, 'max_seq_len': 50}

        # JULES: Changed to CamelCase for consistency
        mock_gnn_node_types = ['Document', 'Account', 'User']
        mock_gnn_edge_types = [('Document', 'HAS_ACCOUNT', 'Account'), ('Document', 'HAS_USER', 'User')]
        mock_gnn_input_dims = {'Document': 10, 'Account': 8, 'User': 0}

        cat_event_order = ['event_type', 'value_bin', 'hour']
        mock_event_vocabs = {'event_type': 10, 'value_bin': 5, 'hour': 24, '<PAD>':0} # Ensure padding_idx=0 for nn.Embedding
        mock_event_embs = {'event_type': 8, 'value_bin': 4, 'hour': 4}
        mock_num_cont_event_feats = 2

        try:
            tfm_model = TFM(
                gnn_config=mock_gnn_config,
                transformer_config=mock_transformer_config,
                gnn_input_feature_dims=mock_gnn_input_dims,
                gnn_node_types=mock_gnn_node_types,
                gnn_edge_types_tuples=mock_gnn_edge_types,
                event_facet_vocab_sizes=mock_event_vocabs,
                event_facet_embedding_dims=mock_event_embs,
                num_continuous_event_features=mock_num_cont_event_feats,
                categorical_event_feature_order=cat_event_order
            )
            print("\nTFM model instantiated.")

            dummy_graph = HeteroData()
            # JULES: Changed to CamelCase for consistency
            dummy_graph['Document'].x = torch.randn(5, 10)
            dummy_graph['Document'].num_nodes = 5
            dummy_graph['Account'].x = torch.randn(8, 8)
            dummy_graph['Account'].num_nodes = 8
            dummy_graph['User'].num_nodes = 5

            dummy_graph['Document','HAS_ACCOUNT','Account'].edge_index = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]], dtype=torch.long)
            dummy_graph['Document','HAS_USER','User'].edge_index = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]], dtype=torch.long)

            batch_s = 3
            seq_l = 15
            num_cat_event_f = len(cat_event_order)
            dummy_events = torch.rand(batch_s, seq_l, num_cat_event_f + mock_num_cont_event_feats)
            dummy_events[:,:,0] = torch.randint(1, mock_event_vocabs['event_type'], (batch_s, seq_l)) # Start from 1 to avoid padding_idx
            dummy_events[:,:,1] = torch.randint(1, mock_event_vocabs['value_bin'], (batch_s, seq_l))
            dummy_events[:,:,2] = torch.randint(1, mock_event_vocabs['hour'], (batch_s, seq_l))

            # These are direct GNN integer indices for 'document' nodes in dummy_graph
            dummy_doc_gnn_indices_in_batch = torch.tensor([0, 2, 4], dtype=torch.long)

            print(f"\nGraph: {dummy_graph}")
            print(f"Event sequences shape: {dummy_events.shape}")
            print(f"Document GNN node indices in batch: {dummy_doc_gnn_indices_in_batch}")

            output = tfm_model(dummy_graph, dummy_events, dummy_doc_gnn_indices_in_batch)
            print(f"\nTFM output shape (pre-training): {output.shape}")
            assert output.shape == (batch_s, seq_l + 1, mock_transformer_config['d_model'])

            num_fraud_classes = 2
            tfm_model.attach_classification_head(num_classes=num_fraud_classes)
            logits = tfm_model(dummy_graph, dummy_events, dummy_doc_gnn_indices_in_batch)
            print(f"TFM output shape (classification logits): {logits.shape}")
            assert logits.shape == (batch_s, num_fraud_classes)

            tfm_model.freeze_base_model(freeze_gnn=True, freeze_transformer=False)
            for name, param in tfm_model.gnn_encoder.named_parameters():
                assert not param.requires_grad, f"GNN param {name} not frozen"
            for name, param in tfm_model.transformer_encoder.named_parameters():
                assert param.requires_grad, f"Transformer param {name} should not be frozen"
            for name, param in tfm_model.classification_head.named_parameters():
                assert param.requires_grad, f"Classification head param {name} not trainable"
            print("Parameter freezing test passed.")

        except Exception as e:
            print(f"\nError during TFM example: {e}")
            import traceback
            traceback.print_exc()
        print("\nDone with TFM examples.")
