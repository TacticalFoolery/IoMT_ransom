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
