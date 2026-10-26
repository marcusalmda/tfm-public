import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import math
import logging
# import pickle # Not strictly needed if using torch.save/load for all
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import torch.nn.functional as F
import traceback # For more detailed error logging

from src.models.tfm import TFM
from src.datasets import TFMPretrainDataset, tfm_pretrain_collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_checkpoint_atomic(checkpoint: Dict[str, Any], path: str):
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)

def configure_cuda_runtime(config: Dict[str, Any], device: torch.device):
    training_cfg = config.get('training', {})
    if device.type != 'cuda':
        return None, False

    if training_cfg.get('allow_tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        logger.info("CUDA TF32 enabled for float32 matmul/convolution where supported.")

    precision = str(training_cfg.get('mixed_precision', 'bf16')).lower()
    if precision in ('bf16', 'bfloat16'):
        logger.info("Using CUDA autocast with bfloat16.")
        return torch.bfloat16, True
    if precision in ('fp16', 'float16'):
        logger.info("Using CUDA autocast with float16.")
        return torch.float16, True
    if precision in ('none', 'fp32', 'float32', 'false'):
        logger.info("Mixed precision disabled by config.")
        return None, False

    logger.warning("Unknown mixed_precision=%s. Falling back to fp32.", precision)
    return None, False

def load_config(path: str = "config/pretrain_config.yaml") -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration from {path}: {e}")
        return None

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    optimizer_name = config.get('optimizer', 'AdamW').lower()
    lr = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay', 0.01))
    params_to_optimize = list(model.parameters())
    if not params_to_optimize:
        raise ValueError("No parameters to optimize. Check model definition.")
    if optimizer_name == 'adamw':
        return optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"Optimizer '{optimizer_name}' not recognized. Defaulting to AdamW.")
        return optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer: optim.Optimizer, scheduler_config: Dict[str, Any], num_epochs: int, steps_per_epoch: int):
    scheduler_name = scheduler_config.get('name', 'ReduceLROnPlateau').lower()

    if scheduler_name == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=scheduler_config.get('factor', 0.1),
                                                    patience=scheduler_config.get('patience',10))
    elif scheduler_name == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch)
    elif scheduler_name == 'warmupcosine':
        try:
            from transformers import get_cosine_schedule_with_warmup
            num_warmup_steps = scheduler_config.get('num_warmup_steps', int(0.1 * num_epochs * steps_per_epoch))
            num_training_steps = num_epochs * steps_per_epoch
            return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        except Exception as exc:
            logger.warning("Could not use transformers warmup+cosine scheduler (%s); using local LambdaLR fallback.", exc)
            num_warmup_steps = scheduler_config.get('num_warmup_steps', int(0.1 * num_epochs * steps_per_epoch))
            num_training_steps = max(1, num_epochs * steps_per_epoch)

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.info(f"Scheduler '{scheduler_name}' not recognized or not specified. No scheduler will be used.")
        return None

def compute_mem_loss(
    mem_predictions_dict: Dict[str, torch.Tensor],
    true_labels_cat: torch.Tensor,    # Shape (batch, seq_len, num_cat_features)
    true_labels_cont: torch.Tensor,   # Shape (batch, seq_len, num_cont_features)
    masked_token_indices_in_batch: torch.Tensor, # Shape (total_masked, 2) [batch_idx, seq_idx]
    feature_info: Dict[str, Dict[str, Any]], # To know type of features
    categorical_feature_to_idx_in_tensor: Dict[str, int], # Maps feature_key to its column index in true_labels_cat
    continuous_feature_to_idx_in_tensor: Dict[str, int],   # Maps feature_key to its column index in true_labels_cont
    device: torch.device
) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=device)
    num_losses_computed = 0

    if masked_token_indices_in_batch.numel() == 0:
        return total_loss

    # These are the [batch_idx, seq_idx] pairs for all masked tokens in the batch
    batch_indices = masked_token_indices_in_batch[:, 0]
    sequence_indices = masked_token_indices_in_batch[:, 1]

    for feature_key, pred_tensor in mem_predictions_dict.items():
        if feature_key not in feature_info:
            logger.warning(f"MEM Loss: Feature key '{feature_key}' from predictions not found in feature_info. Skipping.")
            continue

        info = feature_info[feature_key]
        actual_labels_for_feature_at_masked_positions = None

        if info['type'] == 'categorical':
            if feature_key not in categorical_feature_to_idx_in_tensor:
                logger.warning(f"MEM Loss: Categorical feature key '{feature_key}' not in categorical_feature_to_idx_in_tensor map. Skipping.")
                continue

            if torch.isinf(pred_tensor).any() or torch.isnan(pred_tensor).any():
                logger.warning(f"MEM Loss: pred_tensor for CAT feature '{feature_key}' contains Inf/NaN BEFORE loss calculation.")

            col_idx_in_cat_tensor = categorical_feature_to_idx_in_tensor[feature_key]
            actual_labels = true_labels_cat[batch_indices, sequence_indices, col_idx_in_cat_tensor].long().to(device)

            if torch.isinf(actual_labels).any() or torch.isnan(actual_labels).any():
                 logger.warning(f"MEM Loss: actual_labels for CAT feature '{feature_key}' contains Inf/NaN.")

            loss_fn_cat = nn.CrossEntropyLoss(ignore_index=-100)
            try:
                loss = loss_fn_cat(pred_tensor, actual_labels)
                if torch.isinf(loss).any() or torch.isnan(loss).any():
                    logger.warning(f"MEM Loss: CAT Loss for '{feature_key}' is Inf/NaN. Predictions: {pred_tensor[:5]}, Labels: {actual_labels[:5]}")
                else:
                    total_loss += loss
                    num_losses_computed += 1
            except Exception as e:
                logger.error(f"MEM Loss: Error during CAT loss calculation for '{feature_key}': {e}. Preds shape: {pred_tensor.shape}, Labels shape: {actual_labels.shape}")


        elif info['type'] == 'continuous':
            if feature_key not in continuous_feature_to_idx_in_tensor:
                logger.warning(f"MEM Loss: Continuous feature key '{feature_key}' not in continuous_feature_to_idx_in_tensor map. Skipping.")
                continue

            if torch.isinf(pred_tensor).any() or torch.isnan(pred_tensor).any():
                logger.warning(f"MEM Loss: pred_tensor for CONT feature '{feature_key}' contains Inf/NaN BEFORE loss calculation.")

            col_idx_in_cont_tensor = continuous_feature_to_idx_in_tensor[feature_key]
            actual_labels = true_labels_cont[batch_indices, sequence_indices, col_idx_in_cont_tensor].float().to(device)

            if torch.isinf(actual_labels).any() or torch.isnan(actual_labels).any():
                logger.warning(f"MEM Loss: actual_labels for CONT feature '{feature_key}' contains Inf/NaN.")

            valid_continuous_mask = actual_labels != -100.0

            if valid_continuous_mask.any():
                loss_fn_cont = nn.MSELoss()
                try:
                    loss = loss_fn_cont(pred_tensor[valid_continuous_mask].squeeze(-1),
                                        actual_labels[valid_continuous_mask])
                    if torch.isinf(loss).any() or torch.isnan(loss).any():
                         logger.warning(f"MEM Loss: CONT Loss for '{feature_key}' is Inf/NaN. Predictions: {pred_tensor[valid_continuous_mask][:5]}, Labels: {actual_labels[valid_continuous_mask][:5]}")
                    else:
                        total_loss += loss
                        num_losses_computed += 1
                except Exception as e:
                    logger.error(f"MEM Loss: Error during CONT loss calculation for '{feature_key}': {e}. Preds shape: {pred_tensor[valid_continuous_mask].squeeze(-1).shape}, Labels shape: {actual_labels[valid_continuous_mask].shape}")

    return total_loss / num_losses_computed if num_losses_computed > 0 else torch.tensor(0.0, device=device)

def compute_contrastive_loss(cls_embedding_transformer: torch.Tensor,
                             cls_embedding_gnn: torch.Tensor,
                             temperature: float = 0.1):
    cls_embedding_transformer = F.normalize(cls_embedding_transformer, p=2, dim=1)
    cls_embedding_gnn = F.normalize(cls_embedding_gnn, p=2, dim=1)
    logits = torch.matmul(cls_embedding_transformer, cls_embedding_gnn.T) / temperature
    batch_size = cls_embedding_transformer.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_j) / 2.0

def main(config_path: str = "config/pretrain_config.yaml", dry_run: bool = False):
    logger.info("Starting TFM Pre-training script...")
    config = load_config(config_path)
    if config is None: return

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # JULES: Original
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # JULES: Original
    # device = torch.device("cpu") # JULES: Forcing CPU for debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # JULES: Reverted to original, map creation moved
    logger.info(f"Using device: {device}")
    amp_dtype, use_amp = configure_cuda_runtime(config, device)

    data_cfg = config.get('data', {})
    graph_path = data_cfg.get('graph_path')
    sequences_path = data_cfg.get('sequences_path')

    if not graph_path or not sequences_path:
        logger.error("Graph or sequences path not specified in config. Exiting.")
        return

    doc_id_to_gnn_idx_map = {} # Initialize map

    try:
        logger.info("Loading graph data (on CPU first for robust ID mapping)...")
        # Load graph on CPU first to safely access python list attributes like original_ids_list
        graph_data_cpu = torch.load(graph_path, weights_only=False)
        logger.info("Graph data loaded to CPU.")

        # <JULES_CPU_GRAPH_INSPECTION_START>
        logger.info(f"JULES_CPU_INSPECT: Type of graph_data_cpu: {type(graph_data_cpu)}")
        if hasattr(graph_data_cpu, 'node_types'):
            logger.info(f"JULES_CPU_INSPECT: graph_data_cpu.node_types: {graph_data_cpu.node_types}")
        else:
            logger.info("JULES_CPU_INSPECT: graph_data_cpu does not have 'node_types' attribute.")

        if hasattr(graph_data_cpu, 'keys'):
            logger.info(f"JULES_CPU_INSPECT: graph_data_cpu.keys(): {graph_data_cpu.keys()}")

        try:
            doc_node_store_cpu_direct = graph_data_cpu['Document']
            logger.info(f"JULES_CPU_INSPECT: Successfully accessed graph_data_cpu['Document']. Type: {type(doc_node_store_cpu_direct)}")
            if hasattr(doc_node_store_cpu_direct, 'original_ids_list'):
                logger.info(f"JULES_CPU_INSPECT: graph_data_cpu['Document'] has 'original_ids_list'.")
                if doc_node_store_cpu_direct.original_ids_list is not None:
                    logger.info(f"JULES_CPU_INSPECT: original_ids_list is not None. Length: {len(doc_node_store_cpu_direct.original_ids_list)}")
                else:
                    logger.info("JULES_CPU_INSPECT: original_ids_list IS None.")
            else:
                logger.info("JULES_CPU_INSPECT: graph_data_cpu['Document'] does NOT have 'original_ids_list'.")
        except KeyError:
            logger.warning("JULES_CPU_INSPECT: KeyError when trying to access graph_data_cpu['Document']. 'Document' node type likely missing or not accessible via dict-style access at this stage.")
        except Exception as e_inspect:
            logger.error(f"JULES_CPU_INSPECT: An unexpected error occurred during CPU graph inspection: {e_inspect}")
        # </JULES_CPU_GRAPH_INSPECTION_END>

        # Create doc_id_to_gnn_idx_map from CPU copy of the graph
        map_created_successfully = False
        try:
            # JULES: Using direct dictionary access as validated by JULES_CPU_INSPECT logs
            doc_node_store_cpu = graph_data_cpu['Document']
            logger.info("Successfully accessed graph_data_cpu['Document'] for map creation.")

            original_ids = doc_node_store_cpu.original_ids_list
            if original_ids is not None:
                doc_id_to_gnn_idx_map = {
                    orig_id: i for i, orig_id in enumerate(original_ids)
                }
                logger.info(f"Created doc_id_to_gnn_idx_map from graph's 'Document.original_ids_list' on CPU ({len(doc_id_to_gnn_idx_map)} entries).")
                map_created_successfully = True
                if len(original_ids) == 0:
                    logger.warning("Graph 'Document.original_ids_list' was found but is empty.")
            else:
                logger.warning("Graph data (CPU copy) 'Document.original_ids_list' IS NONE, though 'Document' node store was accessed.")
        except KeyError:
            logger.warning("Graph data (CPU copy) 'Document' node type not found via direct key access (KeyError).")
        except AttributeError:
            logger.warning("Graph data (CPU copy) 'Document' node type was accessed, but does not have 'original_ids_list' attribute (AttributeError).")
        except Exception as e_map: # Catch any other unexpected errors during map creation
            logger.error(f"Unexpected error during map creation from original_ids_list on CPU: {e_map}")

        if not map_created_successfully:
            map_path = os.path.join(os.path.dirname(graph_path), 'doc_id_to_graph_idx.pt')
            if os.path.exists(map_path):
                doc_id_to_gnn_idx_map = torch.load(map_path, weights_only=False)
                logger.info(f"Loaded canonical doc_id_to_gnn_idx_map from {map_path} ({len(doc_id_to_gnn_idx_map)} entries).")
                map_created_successfully = True
            else:
                logger.warning("Failed to create map from original_ids_list and no canonical map found. Attempting sequence-order fallback map.")
                sequences_peek_data_for_fallback = torch.load(sequences_path, weights_only=False) # Load only if needed
                all_doc_ids_in_sequences = sorted(list(sequences_peek_data_for_fallback.keys()))
                if not all_doc_ids_in_sequences:
                    logger.error("No document IDs in sequences for fallback map. Cannot proceed.")
                    return
                doc_id_to_gnn_idx_map = {doc_id: i for i, doc_id in enumerate(all_doc_ids_in_sequences)}

                num_doc_nodes_graph_fallback_check = graph_data_cpu['Document'].num_nodes if 'Document' in graph_data_cpu.node_types and hasattr(graph_data_cpu['Document'], 'num_nodes') else 0
                if num_doc_nodes_graph_fallback_check != len(all_doc_ids_in_sequences):
                    logger.error(f"Fallback map size ({len(all_doc_ids_in_sequences)}) mismatch with graph 'Document' nodes ({num_doc_nodes_graph_fallback_check}). Map is unreliable.")
                    if num_doc_nodes_graph_fallback_check > 0: return
                logger.info(f"Created fallback doc_id_to_gnn_idx_map with {len(doc_id_to_gnn_idx_map)} entries.")
                del sequences_peek_data_for_fallback

        # Now move graph to the target device
        logger.info(f"Moving graph data to device: {device}...")
        graph_data_for_collate = graph_data_cpu.to(device)
        logger.info("Graph data moved to device.")
        del graph_data_cpu # Clean up CPU copy

        logger.info("Loading sequence data for doc_id mapping setup...") # This is just for peeking, map already created if possible
        sequences_peek_data = torch.load(sequences_path, weights_only=False) # Keep this for other setup if needed by dataset init
        logger.info("Initial data loaded successfully for setup.")

        # <JULES_DEBUG_PRINT_START>
        # Inspect graph_data_for_collate (now on target device) to see if original_ids_list is still there
        # (it might be gone after .to(device) if it's a Python list)
        # This is mostly for curiosity now, as we've already used it from the CPU copy.
        logger.info("--- JULES: Graph Data Inspection (after .to(device)) START ---")
        try:
            logger.info(f"JULES: graph_data_for_collate type: {type(graph_data_for_collate)}")
            logger.info(f"JULES: graph_data_for_collate metadata: {graph_data_for_collate.metadata()}")
            doc_type_present = 'Document' in graph_data_for_collate.node_types
            logger.info(f"JULES: 'Document' in graph_data_for_collate.node_types: {doc_type_present}")

            if doc_type_present:
                doc_node_store = graph_data_for_collate['Document']
                logger.info(f"JULES: Type of graph_data_for_collate['Document']: {type(doc_node_store)}")

                has_orig_ids = hasattr(doc_node_store, 'original_ids_list')
                logger.info(f"JULES: hasattr(graph_data_for_collate['Document'], 'original_ids_list'): {has_orig_ids}")

                if has_orig_ids and doc_node_store.original_ids_list is not None:
                    logger.info(f"JULES: graph_data_for_collate['Document'].original_ids_list IS NOT NONE")
                    logger.info(f"JULES: Length of original_ids_list: {len(doc_node_store.original_ids_list)}")
                    logger.info(f"JULES: First 5 original_ids: {doc_node_store.original_ids_list[:5]}")
                elif has_orig_ids:
                    logger.info(f"JULES: graph_data_for_collate['Document'].original_ids_list IS NONE")
                else:
                    logger.info(f"JULES: graph_data_for_collate['Document'] does not have attribute 'original_ids_list'")

                if hasattr(doc_node_store, 'num_nodes') and doc_node_store.num_nodes is not None:
                    logger.info(f"JULES: graph_data_for_collate['Document'].num_nodes: {doc_node_store.num_nodes}")
                else:
                    logger.info(f"JULES: graph_data_for_collate['Document'].num_nodes is not available or None.")

                if hasattr(doc_node_store, 'x') and doc_node_store.x is not None:
                    logger.info(f"JULES: graph_data_for_collate['Document'].x.shape: {doc_node_store.x.shape}")
                else:
                    logger.info(f"JULES: graph_data_for_collate['Document'].x is not available or None.")
            else:
                logger.info(f"JULES: Node type 'Document' not found in graph_data_for_collate.")
        except Exception as e_debug:
            logger.error(f"JULES: Error during debug printing: {e_debug}")
            traceback.print_exc()
        logger.info("--- JULES: Graph Data Inspection END ---")
        # </JULES_DEBUG_PRINT_END>

    except Exception as e:
        logger.error(f"Failed to load initial data for setup: {e}")
        traceback.print_exc()
        return

    # doc_id_to_gnn_idx_map is now definitively created above (lines ~150-175)
    # using graph_data_cpu before it's moved to device.
    # The following block that re-checked or created it using graph_data_for_collate
    # is now removed to prevent conflicts or erroneous warnings.
    # JULES: Removed original block for doc_id_to_gnn_idx_map creation which was here.
    # It started with:
    # # doc_id_to_gnn_idx_map = {}
    # # doc_node_data_from_graph = graph_data_for_collate.get('Document', None)
    # # if doc_node_data_from_graph is not None and hasattr... etc.
    # This logic is now handled robustly before graph_data_cpu.to(device).

    if not doc_id_to_gnn_idx_map:
        logger.error("doc_id_to_gnn_idx_map is empty after attempting creation. This should not happen if graph or sequences are valid. Exiting.")
        return

    # <JULES_CONSISTENCY_CHECKS_START>
    logger.info("--- JULES: Data Consistency Checks START ---")
    num_doc_nodes_in_graph = 0
    if 'Document' in graph_data_for_collate.node_types and hasattr(graph_data_for_collate['Document'], 'num_nodes'):
        num_doc_nodes_in_graph = graph_data_for_collate['Document'].num_nodes
        logger.info(f"JULES_CONSISTENCY: Number of 'Document' nodes in graph_data_for_collate: {num_doc_nodes_in_graph}")
    else:
        logger.warning("JULES_CONSISTENCY: 'Document' node type not found in graph_data_for_collate or num_nodes not available for it.")

    logger.info(f"JULES_CONSISTENCY: Number of entries in doc_id_to_gnn_idx_map: {len(doc_id_to_gnn_idx_map)}")

    if num_doc_nodes_in_graph != len(doc_id_to_gnn_idx_map):
        # This warning is most relevant if map_created_successfully is True (i.e., map came from graph's original_ids_list)
        logger.warning(f"JULES_CONSISTENCY_ALERT: Mismatch between graph document nodes ({num_doc_nodes_in_graph}) and map size ({len(doc_id_to_gnn_idx_map)}). This is problematic if the map was derived from the graph.")

    # sequences_peek_data was loaded earlier for fallback map creation, may or may not exist if fallback wasn't needed.
    # Its keys are now expected to be original string document IDs.
    if 'sequences_peek_data' in locals() and sequences_peek_data is not None:
        sequence_doc_ids_set = set(sequences_peek_data.keys())
        num_docs_in_sequences_peek = len(sequence_doc_ids_set)
        logger.info(f"JULES_CONSISTENCY: Number of unique document IDs in (peeked) sequence data: {num_docs_in_sequences_peek}")

        map_keys_set = set(doc_id_to_gnn_idx_map.keys())

        if map_created_successfully: # Map from graph's original_ids_list
            if num_doc_nodes_in_graph != num_docs_in_sequences_peek: # Compare graph node count with sequence doc count
                logger.warning(f"JULES_CONSISTENCY_ALERT: Graph 'Document' nodes count ({num_doc_nodes_in_graph}) differs from unique sequence document ID count ({num_docs_in_sequences_peek}). Ensure data generation scripts are consistent.")

            if map_keys_set != sequence_doc_ids_set: # Compare the actual sets of IDs
                logger.warning("JULES_CONSISTENCY_ALERT: Document IDs in graph's original_ids_list (map keys) do not perfectly match document IDs in sequence data.")
                missing_in_sequences = map_keys_set - sequence_doc_ids_set
                if missing_in_sequences:
                    logger.warning(f"  Docs in graph map but not in sequences (first 5): {list(missing_in_sequences)[:5]}")
                missing_in_graph_map = sequence_doc_ids_set - map_keys_set
                if missing_in_graph_map:
                    logger.warning(f"  Docs in sequences but not in graph map (first 5): {list(missing_in_graph_map)[:5]}")
        else: # Map from fallback (sequences)
            if map_keys_set != sequence_doc_ids_set:
                 logger.error(f"JULES_CONSISTENCY_ERROR: Fallback map keys do not match sequence data keys. This is a critical internal error. Fallback map keys count: {len(map_keys_set)}, Sequence data keys count: {len(sequence_doc_ids_set)}")
            # The following is redundant if the above set comparison is done, but kept for explicitness on counts
            if len(doc_id_to_gnn_idx_map) != num_docs_in_sequences_peek:
                 logger.warning(f"JULES_CONSISTENCY_ALERT: Fallback map size ({len(doc_id_to_gnn_idx_map)}) differs from sequence file document count ({num_docs_in_sequences_peek}).")
    else:
        logger.info("JULES_CONSISTENCY: sequences_peek_data was not loaded (or available) prior to this check. Dataset will load it. Cannot compare sequence keys directly here.")
    logger.info("--- JULES: Data Consistency Checks END ---")
    # </JULES_CONSISTENCY_CHECKS_END>

    dataset = TFMPretrainDataset(graph_path, sequences_path, config) # Dataset loads data internally
    from functools import partial
    collate_fn_with_args = partial(tfm_pretrain_collate_fn, graph_data=graph_data_for_collate, doc_id_to_gnn_idx_map=doc_id_to_gnn_idx_map)

    dataloader = DataLoader(
        dataset, batch_size=config.get('training', {}).get('batch_size', 32), shuffle=True,
        num_workers=0, # JULES: Temporarily set to 0 for debugging CUDA + multiprocessing issues
        # num_workers=data_cfg.get('num_workers', 0), # Original line
        collate_fn=collate_fn_with_args,
        pin_memory=False # JULES: Set to False as graph_data in batch is already on target device
        # pin_memory=True if device.type == 'cuda' else False # Original line
    )

    gnn_input_dims = {ntype: graph_data_for_collate[ntype].x.size(1)
                      if hasattr(graph_data_for_collate[ntype], 'x') and graph_data_for_collate[ntype].x is not None else 0
                      for ntype in graph_data_for_collate.node_types}

    tf_cfg = config.get('transformer_model', {})

    model_event_facet_vocab_sizes = {
        feat_name: info['vocab_size']
        for feat_name, info in dataset.feature_info.items() if info['type'] == 'categorical'
    }
    model_categorical_feature_order = [
        key for key in dataset.feature_keys_ordered
        if dataset.feature_info.get(key, {}).get('type') == 'categorical'
    ]

    model = TFM(
        gnn_config=config.get('gnn_model',{}),
        transformer_config=tf_cfg,
        gnn_input_feature_dims=gnn_input_dims,
        gnn_node_types=graph_data_for_collate.node_types,
        gnn_edge_types_tuples=graph_data_for_collate.edge_types,
        event_facet_vocab_sizes=model_event_facet_vocab_sizes,
        event_facet_embedding_dims=tf_cfg.get('event_facet_embedding_dims', {}),
        num_continuous_event_features=len(dataset.continuous_feature_indices_in_tensor),
        categorical_event_feature_order=model_categorical_feature_order,
        mem_feature_info=dataset.feature_info
    ).to(device)

    optimizer = get_optimizer(model, config.get('training', {}))
    num_epochs = config.get('training', {}).get('num_epochs', 10)
    steps_per_epoch = len(dataloader)
    # Pass the 'scheduler_params' sub-config to get_scheduler
    scheduler = get_scheduler(optimizer, config.get('training', {}).get('scheduler_params', {}), num_epochs, steps_per_epoch)

    if dry_run:
        logger.info(
            "Dry run complete: dataset=%d docs, steps_per_epoch=%d, epochs=%d, batch_size=%s, grad_accum=%s",
            len(dataset),
            steps_per_epoch,
            num_epochs,
            config.get('training', {}).get('batch_size'),
            config.get('training', {}).get('gradient_accumulation_steps', 1),
        )
        return

    loss_weights = config.get('training', {}).get('loss_weights', {})
    mem_w = loss_weights.get('loss_mem_weight', 1.0)
    con_w = loss_weights.get('loss_contrastive_weight', 0.5)

    checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', 'checkpoints/pretrain')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Initialize best_loss for saving the best model ---
    best_loss = float('inf')
    best_pending_checkpoint = None  # Holds a deferred best snapshot until the next save interval
    # --- Load best_loss if resuming from a checkpoint ---
    resume_training = config.get('training', {}).get('resume_from_checkpoint', False)
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_model.pt")
    start_epoch = 0

    # --- Initialize lists for storing metrics for plotting ---
    epoch_losses, mem_losses, con_losses, learning_rates = [], [], [], []

    if resume_training and os.path.exists(last_checkpoint_path):
        checkpoint = None
        loaded_checkpoint_path = None
        resume_candidates = [
            last_checkpoint_path,
            os.path.join(checkpoint_dir, "best_model.pt"),
        ]
        for candidate_path in resume_candidates:
            if not os.path.exists(candidate_path):
                continue
            try:
                logger.info(f"Attempting to resume training from checkpoint: {candidate_path}")
                checkpoint = torch.load(candidate_path, map_location=device)
                loaded_checkpoint_path = candidate_path
                break
            except Exception as ckpt_error:
                logger.warning(f"Could not load checkpoint {candidate_path}: {ckpt_error}")

        if checkpoint is None:
            logger.warning("Resume training was set, but no valid checkpoint could be loaded. Starting from scratch.")
            resume_training = False
        else:
            logger.info(f"Resuming training from checkpoint: {loaded_checkpoint_path}")
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logger.warning("Checkpoint has no optimizer_state_dict; optimizer will restart from config.")
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif scheduler:
                logger.warning("Checkpoint has no scheduler_state_dict; scheduler will restart from config.")
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf'))) # Load best_loss

            # Load saved metrics if available in checkpoint (optional, good for exact plot continuation)
            if 'epoch_losses' in checkpoint: epoch_losses = checkpoint['epoch_losses']
            if 'mem_losses' in checkpoint: mem_losses = checkpoint['mem_losses']
            if 'con_losses' in checkpoint: con_losses = checkpoint['con_losses']
            if 'learning_rates' in checkpoint: learning_rates = checkpoint['learning_rates']
            logger.info(f"Resumed from epoch {start_epoch}. Current best loss: {best_loss:.4f}. Loaded {len(epoch_losses)} previous loss entries.")
    if not resume_training:
        if config.get('training', {}).get('resume_from_checkpoint', False):
            logger.warning(f"Resume training was set, but checkpoint {last_checkpoint_path} not found. Starting from scratch.")
        logger.info("Starting training from scratch.")


    logger.info(f"Starting training for {num_epochs} epochs...")
    grad_accum_steps = int(config.get('training', {}).get('gradient_accumulation_steps', 1))
    grad_accum_steps = max(1, grad_accum_steps)
    for epoch in range(start_epoch, num_epochs): # Adjusted loop to account for start_epoch
        model.train()
        total_loss_epoch, total_mem_loss_epoch, total_con_loss_epoch = 0.0, 0.0, 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):

            # Updated batch unpacking
            event_seq_cat = batch['event_sequences_cat'].to(device)
            event_seq_cont = batch['event_sequences_cont'].to(device)
            doc_indices = batch['document_node_indices_in_graph'].to(device)
            pad_mask = batch['padding_mask_transformer'].to(device)
            lbl_cat = batch['labels_cat'].to(device) # New labels
            lbl_cont = batch['labels_cont'].to(device) # New labels

            # Determine masked token indices for MEM.
            # A token is masked if any of its categorical labels are not -100.
            # (Assuming continuous labels are also -100 for non-masked tokens,
            # or this check is sufficient based on __getitem__ logic)
            is_target_token = torch.any(lbl_cat != -100, dim=-1)
            masked_token_indices_for_model = is_target_token.nonzero(as_tuple=False).to(device)

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                tfm_outputs_dict = model(
                    graph_data=batch['graph_data'], # This is already on device from collate_fn if graph_data_for_collate was
                    event_sequences_cat=event_seq_cat,
                    event_sequences_cont=event_seq_cont,
                    document_node_indices_in_graph=doc_indices,
                    padding_mask=pad_mask,
                    masked_token_indices_in_sequence=masked_token_indices_for_model
                )

                loss_mem = torch.tensor(0.0, device=device)
                if 'mem_predictions' in tfm_outputs_dict and tfm_outputs_dict['mem_predictions'] is not None and masked_token_indices_for_model.numel() > 0 :
                    # This call will need to be updated once compute_mem_loss is refactored
                    loss_mem = compute_mem_loss( # TODO: Refactor compute_mem_loss and this call
                        mem_predictions_dict=tfm_outputs_dict['mem_predictions'],
                        # true_labels_full_sequence=lbl_mem, # Old way
                        true_labels_cat=lbl_cat,             # New
                        true_labels_cont=lbl_cont,           # New
                        masked_token_indices_in_batch=masked_token_indices_for_model,
                        feature_info=dataset.feature_info,
                        # feature_keys_ordered=dataset.feature_keys_ordered, # May not be needed directly by new compute_mem_loss
                        categorical_feature_to_idx_in_tensor=dataset.categorical_feature_to_idx_in_tensor, # Needs to be added to dataset
                        continuous_feature_to_idx_in_tensor=dataset.continuous_feature_to_idx_in_tensor,   # Needs to be added to dataset
                        device=device
                    )

                loss_con = torch.tensor(0.0, device=device)
                if 'cls_embedding_transformer' in tfm_outputs_dict and 'cls_embedding_gnn' in tfm_outputs_dict:
                    loss_con = compute_contrastive_loss(
                        cls_embedding_transformer=tfm_outputs_dict['cls_embedding_transformer'],
                        cls_embedding_gnn=tfm_outputs_dict['cls_embedding_gnn'],
                        temperature=tf_cfg.get('contrastive_temp', 0.1)
                    )

                combined_loss = (mem_w * loss_mem) + (con_w * loss_con)
            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                logger.warning(f"NaN or Inf loss detected: MEM={loss_mem.item()}, CON={loss_con.item()}. Skipping backward pass.")
            else:
                (combined_loss / grad_accum_steps).backward()
                clip_norm = config.get('training', {}).get('clip_grad_norm')
                should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if should_step:
                    if clip_norm and float(clip_norm) > 0: # Ensure clip_norm is positive
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
                    optimizer.step()
                    optimizer.zero_grad()

            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and (((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))):
                scheduler.step()

            total_loss_epoch += combined_loss.item() if not (torch.isnan(combined_loss) or torch.isinf(combined_loss)) else 0
            total_mem_loss_epoch += loss_mem.item() if not (torch.isnan(loss_mem) or torch.isinf(loss_mem)) else 0
            total_con_loss_epoch += loss_con.item() if not (torch.isnan(loss_con) or torch.isinf(loss_con)) else 0
            progress_bar.set_postfix({'L': combined_loss.item(), 'MEM': loss_mem.item(), 'CON': loss_con.item(), 'LR': optimizer.param_groups[0]['lr']})

        avg_loss = total_loss_epoch / len(dataloader) if len(dataloader) > 0 else 0
        avg_mem = total_mem_loss_epoch / len(dataloader) if len(dataloader) > 0 else 0
        avg_con = total_con_loss_epoch / len(dataloader) if len(dataloader) > 0 else 0
        logger.info(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f} (MEM: {avg_mem:.4f}, CON: {avg_con:.4f})")

        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)

        # --- Updated Checkpoint Saving Logic ---
        # 1. Save the last model checkpoint (for resuming)
        current_epoch_num = epoch + 1

        # --- Store metrics for plotting ---
        epoch_losses.append(avg_loss)
        mem_losses.append(avg_mem)
        con_losses.append(avg_con)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        # --- End store metrics ---

        last_ckpt_interval = int(config.get('logging', {}).get('last_checkpoint_interval', 1))
        last_ckpt_interval = max(1, last_ckpt_interval)
        should_save_last = (current_epoch_num % last_ckpt_interval == 0) or (current_epoch_num == num_epochs)
        if should_save_last:
            last_ckpt_path = os.path.join(checkpoint_dir, "last_model.pt")
            save_checkpoint_atomic({
                'epoch': current_epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,  # Save current best_loss
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': config, # Save the original config used for this run
                # Save collected metrics in the checkpoint for exact plot continuation on resume
                'epoch_losses': epoch_losses,
                'mem_losses': mem_losses,
                'con_losses': con_losses,
                'learning_rates': learning_rates
            }, last_ckpt_path)
            logger.info(f"Last checkpoint for epoch {current_epoch_num} saved to {last_ckpt_path}")

        # 2. Save the best model checkpoint (for fine-tuning/evaluation)
        best_ckpt_interval = int(config.get('logging', {}).get('best_checkpoint_interval', 1))
        best_ckpt_interval = max(1, best_ckpt_interval)
        should_save_best = (current_epoch_num % best_ckpt_interval == 0) or (current_epoch_num == num_epochs)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_pending_checkpoint = {
                'epoch': current_epoch_num,
                'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'loss': best_loss,
                'config': config # Save config for reference
            }
            if not should_save_best:
                logger.info(f"New best loss observed but checkpoint deferred to interval: {best_loss:.4f} at epoch {current_epoch_num}")
        if should_save_best and best_pending_checkpoint is not None:
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint_atomic(best_pending_checkpoint, best_ckpt_path)
            logger.info(f"🎉 New best model saved to {best_ckpt_path} with loss: {best_pending_checkpoint['loss']:.4f} at epoch {best_pending_checkpoint['epoch']}")
            best_pending_checkpoint = None

        # 3. (Optional) Keep periodic saving if checkpoint_interval is explicitly set to something other than 1
        checkpoint_interval = config.get('logging', {}).get('checkpoint_interval', 1)
        # Avoid double-saving if interval is 1 (covered by last_model.pt), unless specifically desired for epoch-numbered files
        if checkpoint_interval > 1 and current_epoch_num % checkpoint_interval == 0:
            periodic_ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{current_epoch_num}.pt")
            save_checkpoint_atomic({
                'epoch': current_epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': config
            }, periodic_ckpt_path)
            logger.info(f"Periodic checkpoint for epoch {current_epoch_num} saved to {periodic_ckpt_path}")
        elif checkpoint_interval == 1 and not os.path.exists(os.path.join(checkpoint_dir, f"model_epoch_{current_epoch_num}.pt")):
            # If interval is 1, but user still expects model_epoch_X.pt files, create them as copies of last_model.pt
            # This part is optional and depends on whether distinct files per epoch are strictly needed when interval is 1.
            # For simplicity, the current logic prioritizes last_model.pt and best_model.pt.
            # If distinct tfm_epoch_{epoch+1}.pt are needed even with interval=1, the old logic can be re-added here.
            # For now, assuming last_model.pt and best_model.pt are primary. The old line was:
            # ckpt_path = os.path.join(checkpoint_dir, f"tfm_epoch_{epoch+1}.pt")
            # The new "last_model.pt" serves the purpose of "tfm_epoch_{epoch+1}.pt" when interval is 1.
            pass


    logger.info("Pre-training finished.")

    # --- Plotting loss evolution ---
    try:
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(epoch_losses) + 1)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot losses on the first y-axis
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs_range, epoch_losses, color=color, linestyle='-', label='Total Loss')
        ax1.plot(epochs_range, mem_losses, color='tab:orange', linestyle='--', label='MEM Loss')
        ax1.plot(epochs_range, con_losses, color='tab:green', linestyle=':', label='CON Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Create a second y-axis for the learning rate
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(epochs_range, learning_rates, color=color, linestyle='-.', label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout() # otherwise the right y-label is slightly clipped
        plt.title('Training Loss and Learning Rate Evolution')

        plot_path = os.path.join(checkpoint_dir, "loss_evolution.png")
        plt.savefig(plot_path)
        logger.info(f"Loss evolution plot saved to {plot_path}")
        plt.close(fig) # Close the figure to free memory

    except ImportError:
        logger.warning("Matplotlib not found. Skipping loss plot generation. Please install matplotlib.")
    except Exception as e:
        logger.error(f"An error occurred while generating the loss plot: {e}")
        traceback.print_exc()
    # --- End plotting ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train the TFM backbone.")
    parser.add_argument("--config", default="config/pretrain_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model/optimizer and exit before training.")
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
