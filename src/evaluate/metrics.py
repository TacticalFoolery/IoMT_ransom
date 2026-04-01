import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

SIM_CLASS_NAMES = ["normal", "encryption_heavy", "exfiltration_first", "wiper", "slow_burn"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    y_true : binary ground-truth labels (N,)
    y_pred : binary predicted labels (N,)
    y_prob : predicted probabilities for the positive class (N,)
    """
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_metrics(metrics: dict, dataset_label: str = "Test") -> None:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*40}")
    print(f" {dataset_label} Results")
    print(f"{'='*40}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  :  {tn:>6}   {fp:>6}")
    print(f"  Actual 1  :  {fn:>6}   {tp:>6}")
    print(f"{'='*40}\n")


def compute_metrics_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict:
    """Multi-class metrics (no AUC-ROC — requires probability scores per class)."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "macro_f1":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None,      zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ),
    }


def print_metrics_multiclass(
    metrics: dict,
    dataset_label: str = "Test",
    class_names: list | None = None,
) -> None:
    print(f"\n{'='*50}")
    print(f" {dataset_label} Results (multi-class)")
    print(f"{'='*50}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
    if class_names and metrics.get("per_class_f1") is not None:
        print(f"\n  Per-class F1:")
        for name, f1 in zip(class_names, metrics["per_class_f1"]):
            print(f"    {name:22s}: {f1:.4f}")
    print(f"\n  Classification Report:")
    print(metrics["classification_report"])
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    cm = metrics["confusion_matrix"]
    header = "".join(f"  {i:>6}" for i in range(cm.shape[1]))
    print(f"       {header}")
    for i, row in enumerate(cm):
        label = class_names[i] if class_names else str(i)
        print(f"  {label[:5]:5s}  {'  '.join(f'{v:>6}' for v in row)}")
    print(f"{'='*50}\n")
