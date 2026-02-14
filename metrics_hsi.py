from typing import Dict

import numpy as np


# 构建混淆矩阵：行是真实类别，列是预测类别。
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# OA: Overall Accuracy，总体正确率。
def compute_oa(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


# AA: Average Accuracy，各类别召回率的平均。
def compute_aa(cm: np.ndarray) -> float:
    per_class_acc = []
    for i in range(cm.shape[0]):
        denom = cm[i, :].sum()
        per_class_acc.append(0.0 if denom == 0 else cm[i, i] / denom)
    return float(np.mean(per_class_acc)) if per_class_acc else 0.0


# Kappa: Cohen's Kappa，校正随机一致性后的指标。
def compute_kappa(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    p0 = np.trace(cm) / total
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = np.sum(row * col) / (total * total)
    denom = 1.0 - pe
    if denom == 0:
        return 0.0
    return float((p0 - pe) / denom)


# 一次性返回完整评估结果，便于主程序统一落盘。
def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    per_class = []
    for i in range(num_classes):
        denom = cm[i, :].sum()
        per_class.append(0.0 if denom == 0 else float(cm[i, i] / denom))
    return {
        "oa": compute_oa(cm),
        "aa": compute_aa(cm),
        "kappa": compute_kappa(cm),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class,
    }
