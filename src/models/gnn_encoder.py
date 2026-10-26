import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, RGCNConv, Linear #് HGraphConv if using latest PyG for general GNNs
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple

class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder for processing a heterogeneous graph.
    Outputs a dictionary of node embeddings for each node type.
    """
    def __init__(self,
                 node_types: List[str],
                 edge_types_tuples: List[Tuple[str, str, str]], # List of (src, relation, dst)
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 gnn_type: str = "HGTConv", # "HGTConv" or "RGCNConv"
                 num_heads: Optional[int] = 4, # For HGTConv
                 dropout_rate: float = 0.1,
                 input_feature_dims: Optional[Dict[str, int]] = None): # Dim of raw input features for each node type
        """
        Initializes the GNNEncoder.

        Args:
            node_types (List[str]): List of all node type names.
            edge_types_tuples (List[Tuple[str,str,str]]): List of all edge types, each as (src, relation, dst).
            hidden_channels (int): Number of hidden channels in GNN layers.
            out_channels (int): Dimension of the output embeddings for each node.
            num_layers (int): Number of GNN layers.
            gnn_type (str): Type of GNN layer to use ('HGTConv' or 'RGCNConv').
            num_heads (Optional[int]): Number of attention heads if using HGTConv. Default is 4.
            dropout_rate (float): Dropout rate. Default is 0.1.
            input_feature_dims (Optional[Dict[str, int]]): Dictionary mapping node type to its
                                                           raw input feature dimension. If None,
                                                           Linear layers won't be used to project
                                                           input features. Nodes without features
                                                           will get a learnable embedding.
        """
        super().__init__()

        if gnn_type not in ["HGTConv", "RGCNConv"]:
            raise ValueError("gnn_type must be 'HGTConv' or 'RGCNConv'")
        if gnn_type == "HGTConv" and num_heads is None:
            raise ValueError("num_heads must be specified for HGTConv")

        self.node_types = node_types
        self.edge_types_tuples = edge_types_tuples
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.input_lins = nn.ModuleDict()
        self.node_type_embeddings = nn.ModuleDict()

        for node_type in node_types:
            in_dim = input_feature_dims.get(node_type, 0) if input_feature_dims else 0
            if in_dim > 0:
                self.input_lins[node_type] = Linear(in_dim, hidden_channels)
            else:
                # If no input features, create a learnable embedding for this node type
                self.node_type_embeddings[node_type] = nn.Embedding(1, hidden_channels)


        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "HGTConv":
                # HGTConv metadata: (node_types_list, edge_types_list_of_tuples)
                conv = HGTConv(hidden_channels, hidden_channels, metadata=(node_types, edge_types_tuples), heads=num_heads)
            elif gnn_type == "RGCNConv":
                # RGCNConv is typically used within a HeteroConv wrapper for full heterogeneous graphs.
                # A single RGCNConv layer expects a homogeneous graph or specific edge_type argument.
                # For simplicity here, we'll raise an error if RGCNConv is chosen without HeteroConv,
                # as its direct application in this loop is non-trivial for truly heterogeneous data.
                # The recommended PyG way is:
                # from torch_geometric.nn import HeteroConv
                # conv = HeteroConv({
                #    et: RGCNConv(hidden_channels, hidden_channels, num_relations=1, ...) for et in edge_types_tuples
                # }, aggr='sum')
                # This structure assumes a single conv layer type for all relations if not using HeteroConv.
                # For this GNNEncoder, we'll assume HGTConv or a future equivalent general Hetero GNN.
                raise NotImplementedError("RGCNConv in this GNNEncoder structure requires HeteroConv wrapper. Use HGTConv or implement HeteroConv with RGCN.")
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout_rate)
        self.output_lins = nn.ModuleDict()
        for node_type in node_types:
            self.output_lins[node_type] = Linear(hidden_channels, out_channels)


    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the GNNEncoder.
        """
        x_dict: Dict[str, torch.Tensor] = {}

        for node_type in self.node_types:
            if node_type in self.input_lins: # Node type has defined input features
                if hasattr(data[node_type], 'x') and data[node_type].x is not None and data[node_type].x.size(0) > 0 :
                    x_dict[node_type] = self.input_lins[node_type](data[node_type].x)
                else: # Should not happen if input_dims are correctly specified based on data
                    print(f"Warning: Node type {node_type} expected input features but found None or empty in data. Using zeros.")
                    num_nodes_of_type = data[node_type].num_nodes
                    x_dict[node_type] = torch.zeros((num_nodes_of_type, self.hidden_channels), device=next(self.parameters()).device)
            elif node_type in self.node_type_embeddings: # Node type uses learnable embedding
                num_nodes_of_type = data[node_type].num_nodes
                if num_nodes_of_type > 0:
                    x_dict[node_type] = self.node_type_embeddings[node_type].weight.repeat(num_nodes_of_type, 1)
                else: # No nodes of this type in the current batch/graph
                    x_dict[node_type] = torch.empty((0, self.hidden_channels), device=self.node_type_embeddings[node_type].weight.device)
            else: # Fallback, should ideally not be reached if all node types are handled
                print(f"Warning: Node type {node_type} has no defined input projection or embedding. Using zeros.")
                num_nodes_of_type = data[node_type].num_nodes
                x_dict[node_type] = torch.zeros((num_nodes_of_type, self.hidden_channels), device=next(self.parameters()).device)

        for i, conv_layer in enumerate(self.convs):
            # Assuming conv_layer is HGTConv or a similarly general heterogeneous GNN layer
            x_dict = conv_layer(x_dict, data.edge_index_dict)

            if i < self.num_layers - 1: # Apply activation and dropout to intermediate layers
                for node_type in x_dict.keys():
                    # Check if tensor is empty (e.g. a node type not present in this batch's graph)
                    if x_dict[node_type].size(0) > 0:
                        x_dict[node_type] = F.relu(x_dict[node_type])
                        x_dict[node_type] = self.dropout(x_dict[node_type])

        out_embeddings: Dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type].size(0) > 0: # Ensure node type is in output and has nodes
                # Apply final activation/dropout before output projection
                final_hidden_state = F.relu(x_dict[node_type])
                final_hidden_state = self.dropout(final_hidden_state)
                out_embeddings[node_type] = self.output_lins[node_type](final_hidden_state)
            elif node_type in x_dict and x_dict[node_type].size(0) == 0 : # Node type present but no nodes in this graph
                 out_embeddings[node_type] = torch.empty((0, self.out_channels), device=x_dict[node_type].device)


        return out_embeddings

if __name__ == '__main__':
    print("GNNEncoder Example (Illustrative)")

    mock_node_types = ['document', 'account', 'user', 'external_entity']
    mock_edge_types_tuples = [
        ('document', 'HAS_ACCOUNT', 'account'),
        ('document', 'HAS_USER', 'user'),
        ('account', 'TRANSACTS_WITH', 'account'),
        ('account', 'TRANSACTS_WITH', 'external_entity'),
        ('user', 'INITIATES_EVENT_ON', 'account')
    ]
    # Scenario 1: All nodes have features
    mock_input_dims_all_feat = { 'document': 45, 'account': 59, 'user': 10, 'external_entity': 5 }

    dummy_data_all_feat = HeteroData()
    for nt, dim in mock_input_dims_all_feat.items():
        num_nodes = 10 if nt != 'account' else 15 # Arbitrary num nodes
        dummy_data_all_feat[nt].x = torch.randn(num_nodes, dim) if dim > 0 else None
        dummy_data_all_feat[nt].num_nodes = num_nodes # Important for HGTConv

    # Populate edge_index (example)
    for src, rel, dst in mock_edge_types_tuples:
        num_src_nodes = dummy_data_all_feat[src].num_nodes
        num_dst_nodes = dummy_data_all_feat[dst].num_nodes
        if num_src_nodes > 0 and num_dst_nodes > 0:
            dummy_data_all_feat[src, rel, dst].edge_index = torch.randint(0, min(num_src_nodes, num_dst_nodes), (2, 20), dtype=torch.long)
        else: # Handle cases where a node type might have 0 nodes
             dummy_data_all_feat[src, rel, dst].edge_index = torch.empty((2,0), dtype=torch.long)


    gnn_encoder_all_feat = GNNEncoder(
        node_types=mock_node_types, edge_types_tuples=mock_edge_types_tuples,
        hidden_channels=64, out_channels=32, num_layers=2,
        gnn_type="HGTConv", num_heads=2,
        input_feature_dims=mock_input_dims_all_feat
    )
    print("\nGNNEncoder (all nodes have features) instantiated.")
    try:
        output_all_feat = gnn_encoder_all_feat(dummy_data_all_feat)
        print("Output Embeddings (all_feat - shapes):")
        for nt, emb in output_all_feat.items(): print(f"- {nt}: {emb.shape}")
        assert all(output_all_feat[nt].shape[1] == 32 for nt in mock_node_types if output_all_feat[nt].numel() > 0)
    except Exception as e: print(f"Error (all_feat): {e}")


    # Scenario 2: Some nodes have no features (use nn.Embedding)
    mock_input_dims_some_no_feat = { 'document': 45, 'account': 59, 'user': 0, 'external_entity': 0 }
    dummy_data_some_no_feat = HeteroData()
    for nt in mock_node_types:
        num_nodes = 10 if nt != 'account' else 15
        dim = mock_input_dims_some_no_feat.get(nt, 0)
        dummy_data_some_no_feat[nt].x = torch.randn(num_nodes, dim) if dim > 0 else None # PyG expects None or no .x if no features
        dummy_data_some_no_feat[nt].num_nodes = num_nodes

    for src, rel, dst in mock_edge_types_tuples:
        num_src_nodes = dummy_data_some_no_feat[src].num_nodes
        num_dst_nodes = dummy_data_some_no_feat[dst].num_nodes
        if num_src_nodes > 0 and num_dst_nodes > 0:
            dummy_data_some_no_feat[src, rel, dst].edge_index = torch.randint(0, min(num_src_nodes, num_dst_nodes), (2, 20), dtype=torch.long)
        else:
            dummy_data_some_no_feat[src, rel, dst].edge_index = torch.empty((2,0), dtype=torch.long)


    gnn_encoder_some_no_feat = GNNEncoder(
        node_types=mock_node_types, edge_types_tuples=mock_edge_types_tuples,
        hidden_channels=32, out_channels=16, num_layers=1,
        input_feature_dims=mock_input_dims_some_no_feat,
        gnn_type="HGTConv", num_heads=2
    )
    print("\nGNNEncoder (some nodes no initial features) instantiated.")
    try:
        output_some_no_feat = gnn_encoder_some_no_feat(dummy_data_some_no_feat)
        print("Output Embeddings (some_no_feat - shapes):")
        for nt, emb in output_some_no_feat.items(): print(f"- {nt}: {emb.shape}")
        assert output_some_no_feat['user'].shape == (dummy_data_some_no_feat['user'].num_nodes, 16)
        assert output_some_no_feat['external_entity'].shape == (dummy_data_some_no_feat['external_entity'].num_nodes, 16)
    except Exception as e: print(f"Error (some_no_feat): {e}")

    print("\nDone with GNNEncoder examples.")
