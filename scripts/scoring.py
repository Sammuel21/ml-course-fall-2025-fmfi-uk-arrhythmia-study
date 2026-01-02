# Evaluation system

import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
    confusion_matrix,
    multilabel_confusion_matrix,
    classification_report,
)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def sigmoid_np(logits_np: np.ndarray) -> np.ndarray:
    # stable sigmoid
    logits_np = logits_np.astype(np.float64)
    out = np.empty_like(logits_np, dtype=np.float64)
    pos = logits_np >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits_np[pos]))
    expx = np.exp(logits_np[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out.astype(np.float32)

def _filter_labels_with_both_classes(y_true: np.ndarray):
    # AUROC/AP can be undefined if a label has only 0s or only 1s in y_true.
    pos = y_true.sum(axis=0)
    keep = (pos > 0) & (pos < y_true.shape[0])
    return keep

def multilabel_eval(Y_true, logits, threshold=0.5, label_names=None, return_reports=False):
    Y_true = _to_numpy(Y_true).astype(int)
    logits = _to_numpy(logits).astype(np.float32)
    P = sigmoid_np(logits)
    Y_pred = (P >= threshold).astype(int)

    out = {}

    # Thresholded metrics
    out["f1_micro"] = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    out["f1_macro"] = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    out["f1_samples"] = f1_score(Y_true, Y_pred, average="samples", zero_division=0)
    out["hamming_loss"] = hamming_loss(Y_true, Y_pred)

    # Threshold-free metrics (skip labels that are constant in Y_true)
    keep = _filter_labels_with_both_classes(Y_true)
    if keep.any():
        out["auroc_micro"] = roc_auc_score(Y_true[:, keep], P[:, keep], average="micro")
        out["auroc_macro"] = roc_auc_score(Y_true[:, keep], P[:, keep], average="macro")
        out["ap_micro"] = average_precision_score(Y_true[:, keep], P[:, keep], average="micro")
        out["ap_macro"] = average_precision_score(Y_true[:, keep], P[:, keep], average="macro")
    else:
        out["auroc_micro"] = np.nan
        out["auroc_macro"] = np.nan
        out["ap_micro"] = np.nan
        out["ap_macro"] = np.nan

    if return_reports:
        out["per_label_confusion"] = multilabel_confusion_matrix(Y_true, Y_pred)
        out["classification_report"] = classification_report(
            Y_true, Y_pred,
            target_names=label_names,
            zero_division=0
        )

    return out, P, Y_pred

def rhythm_conditions_eval(Y_true, logits, rhythm_idx, cond_idx, threshold=0.5,
                           rhythm_label_names=None, cond_label_names=None):
    Y_true = _to_numpy(Y_true).astype(int)
    logits = _to_numpy(logits).astype(np.float32)
    P = sigmoid_np(logits)

    # ---- overall (all labels)
    overall, _, _ = multilabel_eval(Y_true, logits, threshold=threshold)

    # ---- conditions (multilabel)
    cond_metrics, _, _ = multilabel_eval(
        Y_true[:, cond_idx], logits[:, cond_idx],
        threshold=threshold, label_names=cond_label_names
    )

    # ---- rhythm (try multiclass if one-hot)
    Y_r = Y_true[:, rhythm_idx]
    P_r = P[:, rhythm_idx]

    one_hot = np.all((Y_r.sum(axis=1) == 1))
    if one_hot:
        y_true_r = Y_r.argmax(axis=1)
        y_pred_r = P_r.argmax(axis=1)

        rhythm = {
            "acc": float((y_true_r == y_pred_r).mean()),
            "f1_macro": f1_score(y_true_r, y_pred_r, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(
                y_true_r, y_pred_r,
                labels=np.arange(len(rhythm_idx))
            ),
            "used_argmax_multiclass": True,
        }
    else:
        # fall back to multilabel rhythm evaluation
        rhythm_metrics, _, _ = multilabel_eval(
            Y_r, logits[:, rhythm_idx],
            threshold=threshold, label_names=rhythm_label_names
        )
        rhythm = dict(rhythm_metrics)
        rhythm["used_argmax_multiclass"] = False

    return {
        "overall_multilabel": overall,
        "conditions_multilabel": cond_metrics,
        "rhythm": rhythm,
    }
