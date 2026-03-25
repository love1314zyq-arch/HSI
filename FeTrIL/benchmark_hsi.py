import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from benchmarks.visualizations import save_task_prediction_artifact
from hsi_data import HSIPatchDataset
from hybrid_hsi_lite import HybridHSILiteBackbone


DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def parse_args():
    parser = argparse.ArgumentParser(description="FeTrIL HSI benchmark")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task-split", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--patches", type=int, default=11)
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--base-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-tol", type=float, default=1e-4)
    return parser.parse_args()


def task_classes(task_split):
    classes = []
    start = 0
    for size in task_split:
        classes.append(list(range(start, start + int(size))))
        start += int(size)
    return classes


def seen_classes(task_groups, session):
    labels = []
    for idx in range(session + 1):
        labels.extend(task_groups[idx])
    return labels


def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def extract_features(backbone, dataset, batch_size, device):
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []
    backbone.eval()
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            feat = backbone(images).cpu().numpy()
            features.append(feat)
            labels.append(target.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def train_base(backbone, head, loader, device, epochs, lr, weight_decay):
    model = nn.Sequential(backbone, head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            count += labels.size(0)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"[fetril_hsi] base epoch {epoch + 1}/{epochs} | loss={loss_sum / max(count, 1):.4f}")
    return backbone


def compute_centroids(features, labels, classes):
    centroids = {}
    for cls in classes:
        cls_features = features[labels == cls]
        centroids[int(cls)] = cls_features.mean(axis=0)
    return centroids


def build_classifier_features(old_centroids, new_features, new_labels, current_classes):
    class_to_features = {int(cls): new_features[new_labels == cls] for cls in current_classes}
    new_centroids = {int(cls): feats.mean(axis=0) for cls, feats in class_to_features.items()}

    pseudo_features = []
    pseudo_labels = []
    if old_centroids:
        current_centroid_matrix = np.stack([new_centroids[int(cls)] for cls in current_classes], axis=0)
        current_class_ids = [int(cls) for cls in current_classes]
        for old_cls, old_centroid in old_centroids.items():
            dists = np.linalg.norm(current_centroid_matrix - old_centroid[None, :], axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_cls = current_class_ids[nearest_idx]
            translated = class_to_features[nearest_cls] - new_centroids[nearest_cls][None, :] + old_centroid[None, :]
            pseudo_features.append(translated)
            pseudo_labels.append(np.full(translated.shape[0], int(old_cls), dtype=np.int64))

    real_new_features = [class_to_features[int(cls)] for cls in current_classes]
    real_new_labels = [np.full(class_to_features[int(cls)].shape[0], int(cls), dtype=np.int64) for cls in current_classes]
    parts_x = pseudo_features + real_new_features
    parts_y = pseudo_labels + real_new_labels
    train_x = np.concatenate(parts_x, axis=0)
    train_y = np.concatenate(parts_y, axis=0)
    return train_x, train_y, new_centroids


def fit_linear_classifier(train_x, train_y, c_value, tol):
    norms = np.linalg.norm(train_x, axis=1, keepdims=True)
    train_x = train_x / np.clip(norms, 1e-12, None)
    clf = LinearSVC(C=float(c_value), tol=float(tol), dual=False, multi_class="ovr")
    clf.fit(train_x, train_y)
    return clf


def evaluate_classifier(backbone, classifier, dataset, batch_size, device, labels_order):
    feats, labels = extract_features(backbone, dataset, batch_size=batch_size, device=device)
    feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    preds = classifier.predict(feats)
    cm = confusion_matrix(labels, preds, labels=labels_order)
    oa = float((preds == labels).mean()) if len(labels) else 0.0
    per_class = []
    for idx in range(len(labels_order)):
        denom = cm[idx, :].sum()
        per_class.append(float(cm[idx, idx] / denom) if denom > 0 else 0.0)
    aa = float(np.mean(per_class)) if per_class else 0.0
    try:
        kappa = float(cohen_kappa_score(labels, preds, labels=labels_order))
    except Exception:
        kappa = 0.0
    if np.isnan(kappa):
        kappa = 0.0
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "labels": labels,
        "preds": preds,
    }


def compute_group_accuracy(labels, preds, task_groups, session):
    result = []
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    for idx in range(session + 1):
        group = np.asarray(task_groups[idx], dtype=np.int64)
        mask = np.isin(labels, group)
        if mask.sum() == 0:
            result.append(0.0)
        else:
            result.append(float((preds[mask] == labels[mask]).mean()))
    return result


def compute_average_forgetting(history):
    if len(history) <= 1:
        return 0.0
    latest = history[-1]
    forgetting = []
    for group_idx in range(len(latest) - 1):
        best_prev = max(step[group_idx] for step in history[:-1] if group_idx < len(step))
        forgetting.append(best_prev - latest[group_idx])
    return float(np.mean(forgetting)) if forgetting else 0.0


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    print(
        f"[fetril_hsi] dataset={args.dataset} | task_split={args.task_split} | seed={args.seed} | "
        f"device={args.device} | output_root={args.output_root}"
    )
    print(
        f"[fetril_hsi] patches={args.patches} | pca_dim={args.pca_dim} | "
        f"feature_dim={args.feature_dim} | base_epochs={args.base_epochs}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task_groups = task_classes(args.task_split)

    base_train = HSIPatchDataset(args.data_root, args.dataset, True, args.seed, task_groups[0], args.patches, args.pca_dim)
    input_channels = int(base_train.input_channels)
    backbone = HybridHSILiteBackbone(in_channels=input_channels, feature_dim=args.feature_dim).to(device)
    head = nn.Linear(args.feature_dim, len(task_groups[0]))
    base_loader = make_loader(base_train, args.batch_size, shuffle=True)

    print(
        f"[fetril_hsi] session 1/{len(task_groups)} | current_labels={task_groups[0]} | "
        f"train_samples={len(base_train)} | input_channels={input_channels}"
    )
    train_base(backbone, head, base_loader, device, args.base_epochs, args.lr, args.weight_decay)
    frozen_backbone = copy.deepcopy(backbone).to(device)
    frozen_backbone.eval()
    for param in frozen_backbone.parameters():
        param.requires_grad = False

    seen_metrics = []
    group_history = []
    centroids = {}

    for session, current_classes in enumerate(task_groups):
        current_seen = seen_classes(task_groups, session)
        train_dataset = HSIPatchDataset(args.data_root, args.dataset, True, args.seed, current_classes, args.patches, args.pca_dim)
        test_dataset = HSIPatchDataset(args.data_root, args.dataset, False, args.seed, current_seen, args.patches, args.pca_dim)
        train_feats, train_labels = extract_features(frozen_backbone, train_dataset, args.batch_size, device)

        if session == 0:
            centroids = compute_centroids(train_feats, train_labels, current_classes)
            classifier = fit_linear_classifier(train_feats, train_labels, args.svm_c, args.svm_tol)
        else:
            print(
                f"[fetril_hsi] session {session + 1}/{len(task_groups)} | current_labels={current_classes} | "
                f"seen_labels={current_seen} | train_samples={len(train_dataset)}"
            )
            old_centroids = {cls: centroids[cls] for cls in seen_classes(task_groups, session - 1)}
            classifier_x, classifier_y, new_centroids = build_classifier_features(old_centroids, train_feats, train_labels, current_classes)
            centroids.update(new_centroids)
            classifier = fit_linear_classifier(classifier_x, classifier_y, args.svm_c, args.svm_tol)

        metrics = evaluate_classifier(frozen_backbone, classifier, test_dataset, args.batch_size, device, current_seen)
        group_acc = compute_group_accuracy(metrics["labels"], metrics["preds"], task_groups, session)
        group_history.append(group_acc)
        avg_forgetting = compute_average_forgetting(group_history)
        coords = np.asarray(test_dataset.coords, dtype=np.int64)
        save_task_prediction_artifact(
            args.output_root,
            session,
            current_seen,
            coords[:, 0],
            coords[:, 1],
            metrics["preds"],
            metrics["labels"],
        )
        print(
            f"[fetril_hsi] eval session {session + 1}/{len(task_groups)} | seen_classes={len(current_seen)} | "
            f"OA={metrics['oa']:.4f} | AA={metrics['aa']:.4f} | Kappa={metrics['kappa']:.4f}"
        )
        print(f"[fetril_hsi] per-task group accuracy: {group_acc}")
        if session > 0:
            print(f"[fetril_hsi] average_forgetting={avg_forgetting:.4f}")
        seen_metrics.append(
            {
                "session": session,
                "seen_classes": len(current_seen),
                "oa": metrics["oa"],
                "aa": metrics["aa"],
                "kappa": metrics["kappa"],
                "per_task_group_accuracy": group_acc,
            }
        )

    result = {
        "algorithm": "fetril",
        "dataset": args.dataset,
        "task_split": [int(x) for x in args.task_split],
        "seed": int(args.seed),
        "seen_metrics": seen_metrics,
        "average_forgetting": compute_average_forgetting(group_history),
    }
    result_path = Path(args.output_root) / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[fetril_hsi] result saved to {result_path}")


if __name__ == "__main__":
    main()
