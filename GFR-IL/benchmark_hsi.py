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
from torch.utils.data import DataLoader

from benchmarks.visualizations import save_task_prediction_artifact
from hsi_data import HSIPixelDataset
from resnet_pixel import ResNetPixelBackbone


DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def parse_args():
    parser = argparse.ArgumentParser(description="GFR-IL HSI benchmark")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task-split", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--patches", type=int, default=1)
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--base-epochs", type=int, default=60)
    parser.add_argument("--inc-epochs", type=int, default=100)
    parser.add_argument("--classifier-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tradeoff", type=float, default=1.0)
    parser.add_argument("--replay-factor", type=int, default=20)
    return parser.parse_args()


def task_classes(task_split):
    groups = []
    start = 0
    for size in task_split:
        groups.append(list(range(start, start + int(size))))
        start += int(size)
    return groups


def seen_classes(task_groups, session):
    labels = []
    for idx in range(session + 1):
        labels.extend(task_groups[idx])
    return labels


def make_loader(dataset, batch_size, shuffle=False):
    drop_last = bool(shuffle and len(dataset) > 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=drop_last,
    )


class IncrementalClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def expand(self, out_dim):
        old_weight = self.fc.weight.data.clone()
        old_bias = self.fc.bias.data.clone()
        in_dim = self.fc.in_features
        old_out = self.fc.out_features
        self.fc = nn.Linear(in_dim, out_dim).to(old_weight.device)
        with torch.no_grad():
            self.fc.weight[:old_out] = old_weight
            self.fc.bias[:old_out] = old_bias

    def forward(self, x):
        return self.fc(x)


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


def compute_stats(features, labels, classes):
    stats = {}
    for cls in classes:
        cls_features = features[labels == cls]
        mean = cls_features.mean(axis=0)
        std = cls_features.std(axis=0)
        std = np.maximum(std, 1e-4)
        stats[int(cls)] = {"mean": mean, "std": std}
    return stats


def evaluate(backbone, classifier, dataset, batch_size, device, labels_order):
    feats, labels = extract_features(backbone, dataset, batch_size=batch_size, device=device)
    with torch.no_grad():
        logits = classifier(torch.from_numpy(feats).float().to(device)).cpu().numpy()
    preds = np.argmax(logits, axis=1)
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
    return {"oa": oa, "aa": aa, "kappa": kappa, "labels": labels, "preds": preds}


def compute_group_accuracy(labels, preds, task_groups, session):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    values = []
    for idx in range(session + 1):
        group = np.asarray(task_groups[idx], dtype=np.int64)
        mask = np.isin(labels, group)
        values.append(float((preds[mask] == labels[mask]).mean()) if mask.sum() else 0.0)
    return values


def compute_average_forgetting(history):
    if len(history) <= 1:
        return 0.0
    latest = history[-1]
    forgetting = []
    for group_idx in range(len(latest) - 1):
        best_prev = max(step[group_idx] for step in history[:-1] if group_idx < len(step))
        forgetting.append(best_prev - latest[group_idx])
    return float(np.mean(forgetting)) if forgetting else 0.0


def train_backbone(backbone, classifier, loader, device, epochs, lr, weight_decay, current_classes, frozen_old=None, tradeoff=1.0):
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        backbone.train()
        classifier.train()
        loss_sum = 0.0
        count = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            feats = backbone(images)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            if frozen_old is not None:
                with torch.no_grad():
                    old_feats = frozen_old(images)
                loss = loss + tradeoff * torch.dist(feats, old_feats, 2) / max(labels.size(0), 1)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            count += labels.size(0)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"[gfril_hsi] train classes={current_classes} | "
                f"epoch {epoch + 1}/{epochs} | loss={loss_sum / max(count, 1):.4f}"
            )


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def sample_replay(stats, replay_factor):
    replay_x = []
    replay_y = []
    if not stats:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    for cls, item in stats.items():
        mean = item["mean"]
        std = item["std"]
        num = max(1, replay_factor)
        samples = mean[None, :] + np.random.randn(num, mean.shape[0]).astype(np.float32) * std[None, :]
        replay_x.append(samples.astype(np.float32))
        replay_y.append(np.full(num, int(cls), dtype=np.int64))
    return np.concatenate(replay_x, axis=0), np.concatenate(replay_y, axis=0)


def train_classifier(classifier, train_x, train_y, device, epochs, lr, weight_decay):
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    tensor_x = torch.from_numpy(train_x).float()
    tensor_y = torch.from_numpy(train_y).long()
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True, num_workers=0)
    for epoch in range(epochs):
        classifier.train()
        loss_sum = 0.0
        count = 0
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            count += labels.size(0)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"[gfril_hsi] replay classifier epoch {epoch + 1}/{epochs} | loss={loss_sum / max(count, 1):.4f}")


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    print(
        f"[gfril_hsi] dataset={args.dataset} | task_split={args.task_split} | seed={args.seed} | "
        f"device={args.device} | output_root={args.output_root}"
    )
    print(
        f"[gfril_hsi] input_mode=pixel | patches={args.patches} | pca_dim={args.pca_dim} | feature_dim={args.feature_dim} | "
        f"base_epochs={args.base_epochs} | inc_epochs={args.inc_epochs} | classifier_epochs={args.classifier_epochs}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    groups = task_classes(args.task_split)

    base_train = HSIPixelDataset(args.data_root, args.dataset, True, args.seed, groups[0], args.patches, args.pca_dim)
    input_channels = int(base_train.input_channels)
    backbone = ResNetPixelBackbone(in_channels=input_channels, feature_dim=args.feature_dim).to(device)
    classifier = IncrementalClassifier(backbone.feature_dim, len(groups[0])).to(device)
    print(
        f"[gfril_hsi] session 1/{len(groups)} | current_labels={groups[0]} | "
        f"train_samples={len(base_train)} | input_channels={input_channels}"
    )
    train_backbone(
        backbone,
        classifier,
        make_loader(base_train, args.batch_size, shuffle=True),
        device,
        args.base_epochs,
        args.lr,
        args.weight_decay,
        groups[0],
    )

    seen_metrics = []
    history = []
    old_stats = {}

    for session, current_classes in enumerate(groups):
        current_seen = seen_classes(groups, session)
        train_dataset = HSIPixelDataset(args.data_root, args.dataset, True, args.seed, current_classes, args.patches, args.pca_dim)
        test_dataset = HSIPixelDataset(args.data_root, args.dataset, False, args.seed, current_seen, args.patches, args.pca_dim)

        if session > 0:
            print(
                f"[gfril_hsi] session {session + 1}/{len(groups)} | current_labels={current_classes} | "
                f"seen_labels={current_seen} | train_samples={len(train_dataset)}"
            )
            classifier.expand(len(current_seen))
            backbone = freeze_module(backbone.to(device))
            print("[gfril_hsi] backbone frozen after base session; incremental stage trains classifier only")
            replay_x, replay_y = sample_replay(old_stats, args.replay_factor)
            new_feats, new_labels = extract_features(backbone, train_dataset, args.batch_size, device)
            if replay_x.size > 0:
                train_x = np.concatenate([new_feats, replay_x], axis=0).astype(np.float32)
                train_y = np.concatenate([new_labels, replay_y], axis=0).astype(np.int64)
            else:
                train_x = new_feats.astype(np.float32)
                train_y = new_labels.astype(np.int64)
            train_classifier(classifier, train_x, train_y, device, args.classifier_epochs, args.classifier_lr, args.weight_decay)

        seen_train = HSIPixelDataset(args.data_root, args.dataset, True, args.seed, current_seen, args.patches, args.pca_dim)
        feats, labels = extract_features(backbone, seen_train, args.batch_size, device)
        old_stats = compute_stats(feats, labels, current_seen)

        metrics = evaluate(backbone, classifier, test_dataset, args.batch_size, device, current_seen)
        group_acc = compute_group_accuracy(metrics["labels"], metrics["preds"], groups, session)
        history.append(group_acc)
        avg_forgetting = compute_average_forgetting(history)
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
            f"[gfril_hsi] eval session {session + 1}/{len(groups)} | seen_classes={len(current_seen)} | "
            f"OA={metrics['oa']:.4f} | AA={metrics['aa']:.4f} | Kappa={metrics['kappa']:.4f}"
        )
        print(f"[gfril_hsi] per-task group accuracy: {group_acc}")
        if session > 0:
            print(f"[gfril_hsi] average_forgetting={avg_forgetting:.4f}")
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
        "algorithm": "gfr_il",
        "dataset": args.dataset,
        "task_split": [int(x) for x in args.task_split],
        "seed": int(args.seed),
        "seen_metrics": seen_metrics,
        "average_forgetting": compute_average_forgetting(history),
    }
    result_path = Path(args.output_root) / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[gfril_hsi] result saved to {result_path}")


if __name__ == "__main__":
    main()
