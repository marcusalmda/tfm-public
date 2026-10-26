import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def positive_class_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)
    if probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1]
    if probs.ndim == 1:
        return probs
    raise ValueError(f"Binary probabilities must have shape (n,) or (n, 2), got {probs.shape}")


def calculate_metrics_classification(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_names: List[str],
    num_classes: int = 2,
    average_method_for_multiclass: str = "macro",
    threshold: float = 0.5,
) -> Dict[str, float]:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    results: Dict[str, float] = {}

    if num_classes == 2:
        score = positive_class_probs(probs)
        preds = (score >= threshold).astype(int)
        auc_probs = score
    else:
        preds = np.argmax(probs, axis=1)
        auc_probs = probs

    for name in metric_names:
        try:
            if name == "accuracy":
                results[name] = accuracy_score(labels, preds)
            elif name == "precision":
                avg = average_method_for_multiclass if num_classes > 2 else "binary"
                results[name] = precision_score(labels, preds, average=avg, zero_division=0)
            elif name == "recall":
                avg = average_method_for_multiclass if num_classes > 2 else "binary"
                results[name] = recall_score(labels, preds, average=avg, zero_division=0)
            elif name == "f1":
                avg = average_method_for_multiclass if num_classes > 2 else "binary"
                results[name] = f1_score(labels, preds, average=avg, zero_division=0)
            elif name == "auc":
                if num_classes == 2:
                    results[name] = roc_auc_score(labels, auc_probs)
                else:
                    results[name] = roc_auc_score(
                        labels,
                        auc_probs,
                        multi_class="ovr",
                        average=average_method_for_multiclass,
                    )
        except ValueError as exc:
            logger.warning("Could not compute metric %s: %s. Setting to 0.0.", name, exc)
            results[name] = 0.0

    if num_classes == 2:
        results["threshold"] = float(threshold)
    return results


def tune_binary_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",
    min_precision: Optional[float] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    labels = np.asarray(labels)
    score = positive_class_probs(np.asarray(probs))
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_value = -np.inf
    best_metrics: Dict[str, float] = {}

    for threshold in thresholds:
        preds = (score >= threshold).astype(int)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        if min_precision is not None and precision < min_precision:
            continue
        if metric == "f2":
            beta2 = 4.0
            value = (1 + beta2) * precision * recall / ((beta2 * precision) + recall) if precision + recall > 0 else 0.0
        elif metric == "recall":
            value = recall
        elif metric == "precision":
            value = precision
        else:
            value = f1

        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = {
                "threshold": best_threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                f"optimized_{metric}": float(value),
            }

    if not best_metrics:
        best_metrics = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, f"optimized_{metric}": 0.0}
    return best_threshold, best_metrics


def create_doc_id_splits(
    document_ids: Sequence[Any],
    labels_dict: Dict[Any, int],
    test_size: float,
    validation_size: float,
    seed: int,
    stratify: bool = True,
) -> Tuple[List[Any], List[Any], List[Any]]:
    document_ids = list(document_ids)
    labels = [labels_dict[doc_id] for doc_id in document_ids]
    stratify_initial = labels if stratify and len(set(labels)) > 1 else None

    train_val_ids, test_ids = train_test_split(
        document_ids,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_initial,
    )

    train_val_labels = [labels_dict[doc_id] for doc_id in train_val_ids]
    stratify_train = train_val_labels if stratify and len(set(train_val_labels)) > 1 else None
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=validation_size,
        random_state=seed,
        stratify=stratify_train,
    )
    return list(train_ids), list(val_ids), list(test_ids)


def compute_class_weights(labels: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(list(labels), dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_classification_loss(
    labels: Iterable[int],
    num_classes: int,
    device: torch.device,
    loss_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    loss_config = loss_config or {}
    loss_type = str(loss_config.get("type", "cross_entropy")).lower()
    class_weight_cfg = loss_config.get("class_weight", None)
    weight = None
    if class_weight_cfg == "balanced":
        weight = compute_class_weights(labels, num_classes, device)
        logger.info("Using balanced class weights: %s", weight.detach().cpu().tolist())
    elif isinstance(class_weight_cfg, (list, tuple)):
        weight = torch.tensor(class_weight_cfg, dtype=torch.float32, device=device)
        logger.info("Using configured class weights: %s", weight.detach().cpu().tolist())

    if loss_type == "focal":
        return FocalLoss(gamma=float(loss_config.get("focal_gamma", 2.0)), weight=weight)
    return nn.CrossEntropyLoss(weight=weight)


def confusion_matrix_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    preds = (positive_class_probs(np.asarray(probs)) >= threshold).astype(int)
    return confusion_matrix(labels, preds)
