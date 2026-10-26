import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
# import pickle # No longer using pickle.load directly for main data
import random
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd # Added for downstream labels Series conversion (if used)
import os # Added for file existence check in example (if used)

class TFMPretrainDataset(Dataset):
    """
    Dataset for TFM Pre-training. (Jules' version)
    Loads preprocessed graph data and event sequences.
    Generates samples for Masked Event Modeling (MEM).
    """
    def __init__(self,
                 graph_data_path: str,
                 sequences_data_path: str,
                 config: Dict[str, Any]):
        print("Loading TFMPretrainDataset (Jules' version)...")
        try:
            self.graph_data = torch.load(graph_data_path, weights_only=False) # Use weights_only=False
            print(f"Graph data loaded from {graph_data_path}")
        except Exception as e:
            print(f"Error loading graph data from {graph_data_path}: {e}")
            raise

        try:
            # discrete_event_sequences.pt is saved by torch.save in build_discrete_sequences.py
            self.sequences_data: Dict[Any, List[Dict[str, Any]]] = torch.load(sequences_data_path, weights_only=False) # Use weights_only=False
            print(f"Discrete event sequences loaded from {sequences_data_path} using torch.load()")
        except Exception as e:
            print(f"Error loading discrete event sequences from {sequences_data_path}: {e}")
            raise

        # JULES: Load the final graph document map for consistency
        # Assumes doc_id_to_graph_idx.pt is in the same directory as graph_data_path
        final_graph_map_path = os.path.join(os.path.dirname(graph_data_path), 'doc_id_to_graph_idx.pt')
        valid_doc_ids_from_graph_map = None
        try:
            final_graph_doc_map = torch.load(final_graph_map_path, weights_only=False)
            valid_doc_ids_from_graph_map = set(final_graph_doc_map.keys())
            print(f"Successfully loaded final graph document map from {final_graph_map_path} with {len(valid_doc_ids_from_graph_map)} entries.")
        except FileNotFoundError:
            print(f"WARNING: Final graph document map '{final_graph_map_path}' not found. "
                  "TFMPretrainDataset will use all document IDs from sequence data. "
                  "This may lead to errors if the graph was sampled or filtered differently "
                  "than the sequence data.")
        except Exception as e:
            print(f"WARNING: Error loading final graph document map from {final_graph_map_path}: {e}. "
                  "Proceeding with all document IDs from sequence data.")

        initial_sequence_doc_ids = list(self.sequences_data.keys())
        print(f"Initial document IDs from sequence data: {len(initial_sequence_doc_ids)}")

        if valid_doc_ids_from_graph_map is not None:
            self.document_ids = sorted([doc_id for doc_id in initial_sequence_doc_ids if doc_id in valid_doc_ids_from_graph_map])
            filtered_count = len(initial_sequence_doc_ids) - len(self.document_ids)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} document IDs from sequences that were not in the final graph map.")

            # Sanity check: if all sequence IDs were supposed to be in graph map (due to graph being built from sequences)
            # but some are missing from dataset after intersection, it's an issue.
            # This check is more relevant if build_graph.py uses SAMPLE_FRACTION < 1.0 *after* filtering by sequence IDs.
            # If build_graph.py only filters by sequence IDs and SAMPLE_FRACTION = 1.0, then
            # valid_doc_ids_from_graph_map should be identical to initial_sequence_doc_ids (if sequence file was source for graph)
            if len(self.document_ids) != len(valid_doc_ids_from_graph_map) and len(initial_sequence_doc_ids) >= len(valid_doc_ids_from_graph_map):
                 # This means some IDs in the graph map (which should be a subset of or equal to sequence IDs) are not in the sequence IDs.
                 # This indicates an inconsistency if build_graph.py correctly used sequence IDs as its source of truth.
                 # However, if build_graph.py did further sampling, then valid_doc_ids_from_graph_map would be smaller.
                 # The current filtering `doc_id in valid_doc_ids_from_graph_map` is correct.
                 pass # Logging for this specific scenario might be too noisy or confusing.
                     # The critical part is that self.document_ids are a subset of valid_doc_ids_from_graph_map AND initial_sequence_doc_ids.
        else:
            # Fallback if map wasn't loaded
            self.document_ids = sorted(initial_sequence_doc_ids)

        if not self.document_ids:
            raise ValueError("No document IDs available for the dataset after filtering. Check data paths and consistency.")

        self.mask_probability = config.get('training', {}).get('mask_probability', 0.15)
        self.mask_continuous_input = config.get('training', {}).get('mask_continuous_input', False)
        self.continuous_mask_value = float(config.get('training', {}).get('continuous_mask_value', 0.0))
        self.max_sequence_length = config.get('data', {}).get('max_sequence_length', None)

        self.feature_keys_ordered = [
            'event_type_id', 'value_processed', 'delta_t_seconds', 'counterparty_idx',
            'specific_event_idx', 'time_hour', 'time_weekday', 'time_day_of_year'
        ]
        self.num_features = len(self.feature_keys_ordered)

        model_config = config.get('model', {})
        self.event_type_vocab_size = model_config.get('event_type_vocab_size', 20)
        self.counterparty_vocab_size = model_config.get('counterparty_vocab_size', 50000)
        self.specific_event_vocab_size = model_config.get('specific_event_vocab_size', 1000)

        self.feature_info = {
            'event_type_id': {'idx': 0, 'type': 'categorical', 'vocab_size': self.event_type_vocab_size, 'mask_token': self.event_type_vocab_size -1},
            'value_processed': {'idx': 1, 'type': 'continuous'},
            'delta_t_seconds': {'idx': 2, 'type': 'continuous'},
            'counterparty_idx': {'idx': 3, 'type': 'categorical', 'vocab_size': self.counterparty_vocab_size, 'mask_token': self.counterparty_vocab_size -1},
            'specific_event_idx': {'idx': 4, 'type': 'categorical', 'vocab_size': self.specific_event_vocab_size, 'mask_token': self.specific_event_vocab_size-1},
            'time_hour': {'idx': 5, 'type': 'categorical', 'vocab_size': 25, 'mask_token': 24},
            'time_weekday': {'idx': 6, 'type': 'categorical', 'vocab_size': 8, 'mask_token': 7},
            'time_day_of_year': {'idx': 7, 'type': 'categorical', 'vocab_size': 367, 'mask_token': 366}
        }

        self.categorical_feature_indices_in_tensor = [
            info['idx'] for key, info in self.feature_info.items() if info['type'] == 'categorical'
        ]
        self.continuous_feature_indices_in_tensor = [
            info['idx'] for key, info in self.feature_info.items() if info['type'] == 'continuous'
        ]
        self.mask_token_ids_for_facets = {
            info['idx']: info['mask_token'] for key, info in self.feature_info.items() if info['type'] == 'categorical'
        }

        # Mappings from feature_key (string) to its column index within the _cat or _cont tensor
        self.categorical_feature_to_idx_in_tensor: Dict[str, int] = {
            self.feature_keys_ordered[original_idx]: i
            for i, original_idx in enumerate(self.categorical_feature_indices_in_tensor)
        }
        self.continuous_feature_to_idx_in_tensor: Dict[str, int] = {
            self.feature_keys_ordered[original_idx]: i
            for i, original_idx in enumerate(self.continuous_feature_indices_in_tensor)
        }

        # --- NORMALIZATION FOR CONTINUOUS FEATURES ---
        print("Calculating statistics for normalization of continuous features...")
        self.recompute_normalization_stats(self.document_ids)
        # --- END NORMALIZATION ---

        print(f"Dataset initialized with {len(self.document_ids)} documents.")

    # _events_to_tensor is removed as its logic is now integrated into __getitem__

    def recompute_normalization_stats(self, doc_ids) -> None:
        """(Re)compute cont_mean/cont_std over the given doc_ids only.

        Called with all document_ids at init. Call again with the train split's
        ids (data.normalization_scope: train_only) to keep val/test events out
        of the normalization parameters.
        """
        num_cont_feats = len(self.continuous_feature_indices_in_tensor)
        cont_count = 0
        cont_sum = torch.zeros(num_cont_feats, dtype=torch.float64)
        cont_sumsq = torch.zeros(num_cont_feats, dtype=torch.float64)
        if not doc_ids:
            print("Warning: No documents available to calculate continuous feature statistics.")

        for doc_id_for_stats in doc_ids:
            event_list_for_stats = self.sequences_data.get(doc_id_for_stats)
            if event_list_for_stats:
                for event_for_stats in event_list_for_stats:
                    cont_vector = [
                        float(event_for_stats.get(self.feature_keys_ordered[original_idx], 0.0))
                        for original_idx in self.continuous_feature_indices_in_tensor
                    ]
                    cont_tensor = torch.tensor(cont_vector, dtype=torch.float64)
                    cont_sum += cont_tensor
                    cont_sumsq += cont_tensor * cont_tensor
                    cont_count += 1

        if cont_count > 0:
            mean = cont_sum / cont_count
            if cont_count > 1:
                variance = (cont_sumsq - (cont_sum * cont_sum / cont_count)) / (cont_count - 1)
            else:
                variance = torch.zeros_like(mean)
            variance = torch.clamp(variance, min=0.0)
            self.cont_mean = mean.to(dtype=torch.float32)
            self.cont_std = torch.sqrt(variance).to(dtype=torch.float32)
            # Add a small epsilon to std to prevent division by zero if std is 0 for a feature
            self.cont_std[self.cont_std == 0] = 1e-6
            print(f"  Normalization stats over {len(doc_ids)} docs / {cont_count} events.")
            print(f"  Mean of continuous features: {self.cont_mean}")
            print(f"  Std dev of continuous features: {self.cont_std}")
        else:
            print("Warning: No continuous features found in the dataset to calculate statistics. Using default mean/std.")
            self.cont_mean = torch.zeros(num_cont_feats)
            self.cont_std = torch.ones(num_cont_feats)

    def _safe_categorical_value(self, key: str, value: Any) -> int:
        info = self.feature_info.get(key, {})
        vocab_size = int(info.get('vocab_size', 0) or 0)
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            value_int = 0
        if vocab_size <= 0:
            return value_int
        return max(0, min(value_int, vocab_size - 1))

    def __len__(self):
        return len(self.document_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc_id = self.document_ids[idx]
        event_list = self.sequences_data.get(doc_id) # Use .get for safety, though IDs should be filtered
        if event_list and self.max_sequence_length and len(event_list) > self.max_sequence_length:
            event_list = event_list[-self.max_sequence_length:]

        if not event_list: # Should not happen if document_ids are correctly filtered from sequences_data keys
            return {
                'doc_id': doc_id,
                'sequence_cat': torch.empty((0, len(self.categorical_feature_indices_in_tensor)), dtype=torch.long),
                'sequence_cont': torch.empty((0, len(self.continuous_feature_indices_in_tensor)), dtype=torch.float32),
                'labels_cat': torch.empty((0, len(self.categorical_feature_indices_in_tensor)), dtype=torch.long),
                'labels_cont': torch.empty((0, len(self.continuous_feature_indices_in_tensor)), dtype=torch.float32),
                'is_masked_token': torch.empty(0, dtype=torch.bool)
            }

        cat_data = []
        cont_data = []

        # Determine the actual indices for categorical and continuous features based on feature_keys_ordered
        cat_indices_ordered = self.categorical_feature_indices_in_tensor
        cont_indices_ordered = self.continuous_feature_indices_in_tensor

        for event in event_list:
            current_cat_features = []
            for original_idx in cat_indices_ordered:
                key = self.feature_keys_ordered[original_idx]
                # Default to 0 for categorical, as it's often used for padding/unknown
                current_cat_features.append(self._safe_categorical_value(key, event.get(key, 0)))
            cat_data.append(current_cat_features)

            current_cont_features = []
            for original_idx in cont_indices_ordered:
                key = self.feature_keys_ordered[original_idx]
                current_cont_features.append(float(event.get(key, 0.0)))
            cont_data.append(current_cont_features)

        sequence_cat = torch.tensor(cat_data, dtype=torch.long)
        sequence_cont = torch.tensor(cont_data, dtype=torch.float32)

        # Apply normalization to continuous features
        if sequence_cont.numel() > 0 : # Check if tensor is not empty
             # Ensure self.cont_mean and self.cont_std are on the same device and broadcastable
            mean_to_use = self.cont_mean.to(sequence_cont.device)
            std_to_use = self.cont_std.to(sequence_cont.device)
            sequence_cont = (sequence_cont - mean_to_use) / std_to_use


        masked_sequence_cat = sequence_cat.clone()
        # As per user's plan, sequence_cont is not directly modified by masking for MEM.
        # Predictions will be made against labels_cont.
        masked_sequence_cont = sequence_cont.clone() # This will now be the normalized version

        labels_cat = torch.full_like(sequence_cat, -100, dtype=torch.long)
        # Labels for continuous features should be the *original* (before normalization) values if the model
        # is expected to predict the original scale. Or, they can be the *normalized* values if the model
        # predicts normalized values. Given the problem, predicting normalized values is usually more stable.
        # Here, we'll make labels_cont also based on the *normalized* sequence_cont.
        # If original values are needed, they'd have to be stored separately before normalization.
        labels_cont = torch.full_like(sequence_cont, -100.0, dtype=torch.float32) # Based on normalized data

        num_tokens_in_seq = sequence_cat.size(0)
        is_masked_token = torch.empty(0, dtype=torch.bool) # Default for empty sequence

        if num_tokens_in_seq > 0:
            is_masked_token = torch.rand(num_tokens_in_seq) < self.mask_probability
            for token_idx in range(num_tokens_in_seq):
                if is_masked_token[token_idx]:
                    # Mascarar features categóricas
                    # i is the index within the categorical features subset
                    for i, original_feature_idx in enumerate(cat_indices_ordered):
                        labels_cat[token_idx, i] = sequence_cat[token_idx, i].item() # Store original
                        mask_val = self.mask_token_ids_for_facets.get(original_feature_idx)
                        if mask_val is not None:
                            masked_sequence_cat[token_idx, i] = mask_val
                        else:
                            # This case should ideally not happen if mask_token_ids_for_facets is complete
                            print(f"Warning: No MASK token ID for cat facet original index {original_feature_idx}")
                            # Fallback: keep original or use a generic mask like 0 if appropriate
                            # masked_sequence_cat[token_idx, i] = 0 # Or some other default mask

                    # Mascarar features contínuas (store original in label, set masked input to 0.0 or special value)
                    # i is the index within the continuous features subset
                    for i, original_feature_idx in enumerate(cont_indices_ordered):
                        labels_cont[token_idx, i] = sequence_cont[token_idx, i].item() # Store original
                        if self.mask_continuous_input:
                            masked_sequence_cont[token_idx, i] = self.continuous_mask_value


        return {
            'doc_id': doc_id,
            'sequence_cat': masked_sequence_cat,
            'sequence_cont': masked_sequence_cont, # Using clone of original, not explicitly masking input values here
            'labels_cat': labels_cat,
            'labels_cont': labels_cont,
            'is_masked_token': is_masked_token # Unpadded, boolean
        }

def tfm_pretrain_collate_fn(batch: List[Dict[str, Any]],
                            graph_data: Any, # This is graph_data_for_collate from pretrain.py (already on device)
                            doc_id_to_gnn_idx_map: Dict[Any, int]
                           ) -> Dict[str, Any]:

    # Extract new separate sequences and labels
    seq_cat_list = [item['sequence_cat'] for item in batch]
    seq_cont_list = [item['sequence_cont'] for item in batch]
    lbl_cat_list = [item['labels_cat'] for item in batch]
    lbl_cont_list = [item['labels_cont'] for item in batch]
    doc_ids_original = [item['doc_id'] for item in batch]
    # is_masked_token_list = [item['is_masked_token'] for item in batch] # If needed later

    # Pad sequences and labels
    # Categorical padding: 0 (commonly used for padding_idx in embeddings)
    # Continuous padding: 0.0
    # Label padding: -100 (standard ignore_index for CrossEntropyLoss like losses)
    padded_seq_cat = pad_sequence(seq_cat_list, batch_first=True, padding_value=0)
    padded_seq_cont = pad_sequence(seq_cont_list, batch_first=True, padding_value=0.0)
    padded_lbl_cat = pad_sequence(lbl_cat_list, batch_first=True, padding_value=-100)
    padded_lbl_cont = pad_sequence(lbl_cont_list, batch_first=True, padding_value=-100.0)

    # Sequence lengths for padding mask (can use seq_cat_list as reference for lengths)
    seq_lengths = torch.tensor([s.size(0) for s in seq_cat_list if s.nelement() > 0], dtype=torch.long)
    if len(seq_lengths) == 0 and any(s.nelement() == 0 for s in seq_cat_list) and len(seq_cat_list) > 0 : # Handle all-empty-sequence batch
        seq_lengths = torch.zeros(len(seq_cat_list), dtype=torch.long)


    batch_size = len(batch)
    # Max length for padding mask, ensure it's at least 1 if all sequences are empty to avoid issues with arange(0)
    max_len = padded_seq_cat.size(1) if batch_size > 0 and padded_seq_cat.nelement() > 0 else 0

    if max_len == 0 and batch_size > 0 : # If sequences are empty, but batch is not, create a dummy mask
         padding_mask = torch.ones(batch_size, 0, dtype=torch.bool) # All positions are padding if seq_len is 0
    elif batch_size == 0: # No items in batch
        padding_mask = torch.empty(0,0, dtype=torch.bool)
    else: # Normal case
        padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= seq_lengths.unsqueeze(1)


    try:
        doc_gnn_indices = torch.tensor([doc_id_to_gnn_idx_map[doc_id] for doc_id in doc_ids_original], dtype=torch.long)
    except KeyError as e:
        missing_id = e.args[0]
        available_ids_sample = list(doc_id_to_gnn_idx_map.keys())[:5]
        raise KeyError(f"Document ID '{missing_id}' from batch not found in doc_id_to_gnn_idx_map. "
                       f"Sample of available IDs in map: {available_ids_sample}") from e
    except Exception as e: # Catch other potential errors, e.g. if doc_ids_original is empty
        print(f"Error generating doc_gnn_indices. doc_ids_original: {doc_ids_original}")
        # Fallback or re-raise:
        if not doc_ids_original: # If no doc_ids, create empty tensor
            doc_gnn_indices = torch.empty(0, dtype=torch.long)
        else:
            raise e # Re-raise if it's another unexpected error


    return {
        'graph_data': graph_data,
        'event_sequences_cat': padded_seq_cat,
        'event_sequences_cont': padded_seq_cont,
        'document_node_indices_in_graph': doc_gnn_indices,
        'padding_mask_transformer': padding_mask,
        'labels_cat': padded_lbl_cat,
        'labels_cont': padded_lbl_cont,
    }

class TFMDownstreamDataset(Dataset):
    def __init__(self, config: Dict[str, Any], # Pass the full config
                 # Optional: allow direct injection for easier testing or varied data sources
                 sequences_data: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
                 labels_dict: Optional[Dict[Any, int]] = None,
                 doc_id_to_graph_idx_map: Optional[Dict[Any, int]] = None,
                 pretrain_setup_for_features: Optional[TFMPretrainDataset] = None
                ):
        print("Loading TFMDownstreamDataset...")
        data_cfg = config['data']
        self.config = config # Store config for potential use

        # Load labels
        if labels_dict:
            self.labels_dict = labels_dict
            print(f"Using provided labels_dict with {len(self.labels_dict)} entries.")
        else:
            labels_path = data_cfg['labels_path']
            try:
                self.labels_dict = torch.load(labels_path)
                print(f"Labels loaded from {labels_path} ({len(self.labels_dict)} entries).")
            except FileNotFoundError:
                print(f"ERROR: Labels file not found at {labels_path}")
                raise
            except Exception as e:
                print(f"Error loading labels from {labels_path}: {e}")
                raise

        # Load doc_id to GNN index map (essential for linking to graph nodes)
        if doc_id_to_graph_idx_map:
            self.doc_id_to_gnn_idx_map = doc_id_to_graph_idx_map
            print(f"Using provided doc_id_to_gnn_idx_map with {len(self.doc_id_to_gnn_idx_map)} entries.")
        else:
            # Default: expect it next to the graph_path from config
            map_path = os.path.join(os.path.dirname(data_cfg['graph_path']), 'doc_id_to_graph_idx.pt')
            try:
                self.doc_id_to_gnn_idx_map = torch.load(map_path)
                print(f"doc_id_to_gnn_idx_map loaded from {map_path} ({len(self.doc_id_to_gnn_idx_map)} entries).")
            except FileNotFoundError:
                print(f"ERROR: doc_id_to_graph_idx.pt not found at {map_path}. This map is crucial.")
                raise
            except Exception as e:
                print(f"Error loading doc_id_to_graph_idx.pt from {map_path}: {e}")
                raise

        valid_graph_docs_set = set(self.doc_id_to_gnn_idx_map.keys())

        # Intersect document IDs: must have a label AND be in the graph
        labeled_doc_ids_set = set(self.labels_dict.keys())
        self.document_ids = sorted(list(valid_graph_docs_set.intersection(labeled_doc_ids_set)))

        if not self.document_ids:
            raise ValueError("No common document IDs found between graph, sequences, and labels. Dataset is empty.")
        print(f"Found {len(self.document_ids)} common document IDs with labels and present in the graph map for fine-tuning.")

        # Feature processing setup (reusing TFMPretrainDataset's logic)
        if pretrain_setup_for_features:
            print("Using provided TFMPretrainDataset for feature setup.")
            # Ensure the provided setup object has the necessary attributes
            if not all(hasattr(pretrain_setup_for_features, attr) for attr in
                       ['feature_info', 'categorical_feature_indices_in_tensor',
                        'continuous_feature_indices_in_tensor', 'feature_keys_ordered',
                        'cont_mean', 'cont_std', 'sequences_data']):
                raise ValueError("Provided pretrain_setup_for_features is missing required attributes.")
            self._pretrain_setup = pretrain_setup_for_features
        else:
            print("Initializing TFMPretrainDataset internally for feature setup...")
            # This requires graph_path and sequences_path to be valid in the passed config
            # TFMPretrainDataset will load its own copy of sequences_data and graph_data.
            # This is acceptable if TFMDownstreamDataset is created once per training run.
            self._pretrain_setup = TFMPretrainDataset(
                graph_data_path=data_cfg['graph_path'], # Pretrain dataset needs this for its own setup
                sequences_data_path=data_cfg['sequences_path'],
                config=config # Pass the full config, TFMPretrainDataset will pick what it needs
            )
            print("Internal TFMPretrainDataset initialized for feature setup.")

        # Copy necessary attributes from the setup object
        self.feature_info = self._pretrain_setup.feature_info
        self.categorical_feature_indices_in_tensor = self._pretrain_setup.categorical_feature_indices_in_tensor
        self.continuous_feature_indices_in_tensor = self._pretrain_setup.continuous_feature_indices_in_tensor
        self.feature_keys_ordered = self._pretrain_setup.feature_keys_ordered
        self.cont_mean = self._pretrain_setup.cont_mean
        self.cont_std = self._pretrain_setup.cont_std
        # CRITICAL: Use the sequences_data from the setup object, especially if it was passed in.
        # This avoids reloading sequences_data if TFMPretrainDataset was passed externally.
        self.sequences_data = self._pretrain_setup.sequences_data

        # Filter self.sequences_data to only contain document_ids relevant to this dataset
        # This is important if TFMPretrainDataset was initialized internally and loaded all sequences
        self.sequences_data = {doc_id: self.sequences_data[doc_id]
                               for doc_id in self.document_ids if doc_id in self.sequences_data}
        print(f"Filtered sequences_data to {len(self.sequences_data)} entries relevant to this fine-tuning dataset.")

    def recompute_normalization_stats(self, doc_ids) -> None:
        """Recompute cont_mean/cont_std over the given doc_ids (e.g. the train split).

        Reuses TFMPretrainDataset's implementation over this dataset's own
        sequences_data and updates this instance's copies, which __getitem__ uses.
        """
        TFMPretrainDataset.recompute_normalization_stats(self, doc_ids)

    def __len__(self):
        return len(self.document_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc_id = self.document_ids[idx]
        label = self.labels_dict[doc_id]

        event_list = self.sequences_data.get(doc_id)
        if event_list and self.config.get('data', {}).get('max_sequence_length') and len(event_list) > self.config['data']['max_sequence_length']:
            event_list = event_list[-self.config['data']['max_sequence_length']:]
        if not event_list: # Should not happen if document_ids are correctly filtered
            # Create empty tensors with correct number of features if a doc_id somehow has no sequence
            # This helps prevent errors in collate_fn with completely empty data
            num_cat_feats = len(self.categorical_feature_indices_in_tensor)
            num_cont_feats = len(self.continuous_feature_indices_in_tensor)
            return {
                'doc_id': doc_id,
                'sequence_cat': torch.empty((0, num_cat_feats), dtype=torch.long),
                'sequence_cont': torch.empty((0, num_cont_feats), dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.long) # Still return label
            }

        cat_data = []
        cont_data = []

        for event in event_list:
            current_cat_features = []
            for original_idx in self.categorical_feature_indices_in_tensor:
                key = self.feature_keys_ordered[original_idx]
                current_cat_features.append(self._pretrain_setup._safe_categorical_value(key, event.get(key, 0))) # Default to 0 for missing cat
            cat_data.append(current_cat_features)

            current_cont_features = []
            for original_idx in self.continuous_feature_indices_in_tensor:
                key = self.feature_keys_ordered[original_idx]
                current_cont_features.append(float(event.get(key, 0.0))) # Default to 0.0 for missing cont
            cont_data.append(current_cont_features)

        sequence_cat = torch.tensor(cat_data, dtype=torch.long) if cat_data else torch.empty((0, len(self.categorical_feature_indices_in_tensor)), dtype=torch.long)
        sequence_cont = torch.tensor(cont_data, dtype=torch.float32) if cont_data else torch.empty((0, len(self.continuous_feature_indices_in_tensor)), dtype=torch.float32)

        # Apply normalization to continuous features
        if sequence_cont.numel() > 0:
            mean_to_use = self.cont_mean.to(sequence_cont.device)
            std_to_use = self.cont_std.to(sequence_cont.device)
            sequence_cont = (sequence_cont - mean_to_use) / std_to_use

        return {
            'doc_id': doc_id,
            'sequence_cat': sequence_cat,
            'sequence_cont': sequence_cont,
            'label': torch.tensor(label, dtype=torch.long)
        }

def tfm_downstream_collate_fn(batch: List[Dict[str, Any]],
                               graph_data: Any, # Expected to be on the correct device
                               doc_id_to_gnn_idx_map: Dict[Any, int]
                              ) -> Dict[str, Any]:

    seq_cat_list = [item['sequence_cat'] for item in batch]
    seq_cont_list = [item['sequence_cont'] for item in batch]
    labels_list = [item['label'] for item in batch] # List of scalar tensors
    doc_ids_original = [item['doc_id'] for item in batch]

    # Pad sequences
    padded_seq_cat = pad_sequence(seq_cat_list, batch_first=True, padding_value=0) # Assuming 0 is pad for cat
    padded_seq_cont = pad_sequence(seq_cont_list, batch_first=True, padding_value=0.0)

    seq_lengths = torch.tensor([s.size(0) for s in seq_cat_list if s.nelement() > 0], dtype=torch.long)
    if len(seq_lengths) == 0 and any(s.nelement() == 0 for s in seq_cat_list) and len(seq_cat_list) > 0:
        seq_lengths = torch.zeros(len(seq_cat_list), dtype=torch.long)

    batch_size = len(batch)
    max_len = padded_seq_cat.size(1) if batch_size > 0 and padded_seq_cat.nelement() > 0 else 0

    if max_len == 0 and batch_size > 0:
        padding_mask = torch.ones(batch_size, 0, dtype=torch.bool)
    elif batch_size == 0:
        padding_mask = torch.empty(0,0, dtype=torch.bool)
    else:
        padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= seq_lengths.unsqueeze(1)

    try:
        doc_gnn_indices = torch.tensor([doc_id_to_gnn_idx_map[doc_id] for doc_id in doc_ids_original], dtype=torch.long)
    except KeyError as e:
        missing_id = e.args[0]
        raise KeyError(f"Collate: Document ID '{missing_id}' from batch not found in doc_id_to_gnn_idx_map.") from e
    except Exception as e:
        if not doc_ids_original:
            doc_gnn_indices = torch.empty(0, dtype=torch.long)
        else:
            raise e

    return {
        'graph_data': graph_data, # This is already on device
        'event_sequences_cat': padded_seq_cat,
        'event_sequences_cont': padded_seq_cont,
        'document_node_indices_in_graph': doc_gnn_indices,
        'padding_mask_transformer': padding_mask,
        'labels': torch.stack(labels_list), # Stack the list of label tensors
        'document_ids_original': doc_ids_original
    }


class GNNOnlyDataset(Dataset):
    """
    Dataset for GNN-only classification.
    Loads document IDs and their corresponding labels.
    Relies on a map to link document IDs to their indices in a graph.
    Does not load or process event sequences.
    """
    def __init__(self,
                 config: Dict[str, Any], # Main config dictionary
                 labels_path: str,
                 doc_id_to_graph_idx_map_path: str):
        """
        Initializes the GNNOnlyDataset.

        Args:
            config (Dict[str, Any]): The main configuration dictionary. Expected to have a 'data' key
                                     if specific sub-paths are needed, but primarily uses direct paths here.
            labels_path (str): Path to the file containing labels (e.g., a torch-saved dictionary
                               mapping document_id to label).
            doc_id_to_graph_idx_map_path (str): Path to the file containing the mapping from
                                                document_id to its GNN node index.
        """
        print("Loading GNNOnlyDataset...")
        self.config = config

        try:
            self.labels_dict: Dict[Any, int] = torch.load(labels_path)
            print(f"Labels loaded from {labels_path} ({len(self.labels_dict)} entries).")
        except FileNotFoundError:
            print(f"ERROR: Labels file not found at {labels_path}")
            raise
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            raise

        try:
            self.doc_id_to_gnn_idx_map: Dict[Any, int] = torch.load(doc_id_to_graph_idx_map_path)
            print(f"doc_id_to_gnn_idx_map loaded from {doc_id_to_graph_idx_map_path} ({len(self.doc_id_to_gnn_idx_map)} entries).")
        except FileNotFoundError:
            print(f"ERROR: doc_id_to_graph_idx map file not found at {doc_id_to_graph_idx_map_path}")
            raise
        except Exception as e:
            print(f"Error loading doc_id_to_graph_idx map from {doc_id_to_graph_idx_map_path}: {e}")
            raise

        # Filter document IDs: must have a label AND be in the doc_id_to_gnn_idx_map
        labeled_doc_ids_set = set(self.labels_dict.keys())
        graph_mapped_doc_ids_set = set(self.doc_id_to_gnn_idx_map.keys())

        self.document_ids = sorted(list(labeled_doc_ids_set.intersection(graph_mapped_doc_ids_set)))

        if not self.document_ids:
            raise ValueError("No common document IDs found between labels and the graph index map. Dataset is empty.")

        num_filtered_labels = len(labeled_doc_ids_set) - len(self.document_ids)
        if num_filtered_labels > 0:
            print(f"Filtered out {num_filtered_labels} document IDs from labels that were not in the graph index map.")

        num_filtered_map = len(graph_mapped_doc_ids_set) - len(self.document_ids)
        if num_filtered_map > 0:
            print(f"Filtered out {num_filtered_map} document IDs from graph index map that were not in the labels.")

        print(f"GNNOnlyDataset initialized with {len(self.document_ids)} common document IDs.")

    def __len__(self):
        return len(self.document_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single data point.

        Args:
            idx (int): Index of the data point.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'doc_id': The document identifier.
                - 'label': The label for the document (as a tensor).
        """
        doc_id = self.document_ids[idx]
        label = self.labels_dict[doc_id]
        return {
            'doc_id': doc_id,
            'label': torch.tensor(label, dtype=torch.long)
        }

def gnn_only_collate_fn(batch: List[Dict[str, Any]],
                        graph_data: 'torch_geometric.data.HeteroData', # The full graph object
                        doc_id_to_gnn_idx_map: Dict[Any, int] # The loaded map
                       ) -> Dict[str, Any]:
    """
    Collate function for GNNOnlyDataset.
    Prepares a batch for input to the GNNOnlyModel.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                      is an output of GNNOnlyDataset.__getitem__.
        graph_data (HeteroData): The complete graph data object (typically pre-loaded and
                                 moved to the target device).
        doc_id_to_gnn_idx_map (Dict[Any, int]): Mapping from document ID to its
                                               node index in the graph for the 'document' type.

    Returns:
        Dict[str, Any]: A dictionary ready for model input, containing:
            - 'graph_data': The passed-in graph_data object.
            - 'document_node_indices_in_graph': Tensor of GNN node indices for 'document' nodes in the batch.
            - 'labels': Tensor of labels for the batch.
    """
    doc_ids_original = [item['doc_id'] for item in batch]
    labels_list = [item['label'] for item in batch] # List of scalar tensors

    # Convert original document IDs to their corresponding GNN node indices
    try:
        doc_gnn_indices = torch.tensor([doc_id_to_gnn_idx_map[doc_id] for doc_id in doc_ids_original], dtype=torch.long)
    except KeyError as e:
        missing_id = e.args[0]
        available_ids_sample = list(doc_id_to_gnn_idx_map.keys())[:5]
        raise KeyError(f"Collate GNNOnly: Document ID '{missing_id}' from batch not found in doc_id_to_gnn_idx_map. "
                       f"Sample of available IDs in map: {available_ids_sample}") from e
    except Exception as e:
        # Handle cases like empty doc_ids_original if necessary, though batch shouldn't be empty.
        if not doc_ids_original:
            doc_gnn_indices = torch.empty(0, dtype=torch.long)
        else:
            raise e # Re-raise if it's another unexpected error

    return {
        'graph_data': graph_data, # This is the same graph object for all batches, already on device
        'document_node_indices_in_graph': doc_gnn_indices,
        'labels': torch.stack(labels_list), # Stack list of tensors into a single tensor
        'document_ids': doc_ids_original
    }
