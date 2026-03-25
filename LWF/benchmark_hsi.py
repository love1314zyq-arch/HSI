import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn import metrics

from benchmarks.visualizations import save_task_prediction_artifact
from hsi_data import HSIPatchDataset
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark wrapper for LwF on HSI class-incremental scenarios.")
    parser.add_argument("--dataset", required=True, choices=["paviau", "salinas", "houston", "indianpines"])
    parser.add_argument("--task-split", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--inc-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--init-lr", type=float, default=0.1)
    parser.add_argument("--incremental-lr", type=float, default=0.01)
    parser.add_argument("--patches", type=int, default=11)
    parser.add_argument("--pca-dim", type=int, default=30)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cumulative_task_labels(task_split):
    labels = []
    start = 0
    per_task = []
    for size in task_split:
        current = list(range(start, start + int(size)))
        per_task.append(current)
        labels.extend(current)
        start += int(size)
    return per_task


def compute_metrics(y_true, y_pred, seen_labels):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=seen_labels)
    total = np.sum(cm)
    if total == 0:
        return 0.0, 0.0, 0.0, []
    acc_class = np.divide(
        np.diag(cm),
        np.sum(cm, axis=1),
        out=np.zeros(len(seen_labels), dtype=np.float32),
        where=np.sum(cm, axis=1) != 0,
    )
    oa = float(np.sum(np.diag(cm)) / total)
    aa = float(np.mean(acc_class)) if len(acc_class) > 0 else 0.0
    pe = float(np.sum(np.sum(cm, axis=1) * np.sum(cm, axis=0)) / (total ** 2))
    denom = 1.0 - pe
    kappa = 0.0 if abs(denom) < 1e-12 else float((oa - pe) / denom)
    return oa, aa, kappa, acc_class.tolist()


def average_forgetting(stage_metrics):
    if len(stage_metrics) <= 1:
        return 0.0
    final = stage_metrics[-1]["task_group_accuracy"]
    forget = 0.0
    count = 0
    for idx in range(len(final) - 1):
        best = max(stage["task_group_accuracy"][idx] for stage in stage_metrics[idx:])
        forget += best - final[idx]
        count += 1
    return float(forget / max(count, 1))


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    print(
        f"[lwf_benchmark] dataset={args.dataset} | task_split={args.task_split} | seed={args.seed} | "
        f"device={args.device} | output_root={output_root}"
    )
    set_seed(int(args.seed))

    task_labels = cumulative_task_labels(args.task_split)
    all_labels = [label for task in task_labels for label in task]
    class_map = {label: idx for idx, label in enumerate(all_labels)}
    map_reverse = {idx: label for label, idx in class_map.items()}

    bootstrap_dataset = HSIPatchDataset(
        repo_root / "data",
        args.dataset,
        True,
        int(args.seed),
        task_labels[0],
        patch_size=int(args.patches),
        pca_dim=int(args.pca_dim),
        input_mode="patch",
    )
    args.input_channels = bootstrap_dataset.input_channels
    args.device = args.device
    args.batch_size = int(args.batch_size)
    args.init_lr = float(args.init_lr)
    args.incremental_lr = float(args.incremental_lr)
    args.num_epochs = int(args.num_epochs)
    args.inc_epochs = int(args.inc_epochs)
    args.backbone_name = "hybrid_hsi_lite"
    args.freeze_feature_extractor_on_increment = True

    init_classes = max(int(args.task_split[0]), 1)
    model = Model(init_classes, class_map, args)
    if torch.cuda.is_available() and args.device != "cpu":
        model.cuda()

    stage_metrics = []
    for session, current_labels in enumerate(task_labels):
        seen_labels = [label for task in task_labels[: session + 1] for label in task]
        print(
            f"[lwf_benchmark] session {session + 1}/{len(task_labels)} | "
            f"current_labels={current_labels} | seen_labels={seen_labels}"
        )
        train_set = HSIPatchDataset(
            repo_root / "data",
            args.dataset,
            True,
            int(args.seed),
            current_labels,
            patch_size=int(args.patches),
            pca_dim=int(args.pca_dim),
            input_mode="patch",
        )
        print(
            f"[lwf_benchmark] train samples={len(train_set)} | input_channels={train_set.input_channels} | "
            f"input_mode=patch | patches={args.patches} | pca_dim={args.pca_dim} | "
            f"epochs={(args.num_epochs if session == 0 else args.inc_epochs)}"
        )
        model.update(train_set, class_map, args)
        model.n_known = model.n_classes
        model.eval()

        eval_set = HSIPatchDataset(
            repo_root / "data",
            args.dataset,
            False,
            int(args.seed),
            seen_labels,
            patch_size=int(args.patches),
            pca_dim=int(args.pca_dim),
            input_mode="patch",
        )
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for _, images, labels in eval_loader:
                if torch.cuda.is_available() and args.device != "cpu":
                    images = images.cuda()
                preds = model.classify(images)
                preds = [map_reverse[int(pred)] for pred in preds.cpu().numpy()]
                preds_all.extend(preds)
                labels_all.extend(labels.numpy().tolist())

        oa, aa, kappa, acc_class = compute_metrics(labels_all, preds_all, seen_labels)
        task_group_accuracy = []
        for group in task_labels[: session + 1]:
            group_mask = np.isin(labels_all, group)
            if not np.any(group_mask):
                task_group_accuracy.append(0.0)
                continue
            y_true = np.asarray(labels_all)[group_mask]
            y_pred = np.asarray(preds_all)[group_mask]
            task_group_accuracy.append(float(np.mean(y_true == y_pred)))

        metric = {
            "task_id": session,
            "seen_classes": len(seen_labels),
            "oa": oa,
            "aa": aa,
            "kappa": kappa,
            "acc_class": acc_class,
            "task_group_accuracy": task_group_accuracy,
        }
        stage_metrics.append(metric)
        coords = np.asarray(eval_set.coords, dtype=np.int64)
        save_task_prediction_artifact(
            str(output_root),
            session,
            seen_labels,
            coords[:, 0],
            coords[:, 1],
            np.asarray(preds_all, dtype=np.int64),
            np.asarray(labels_all, dtype=np.int64),
        )
        print(
            f"[lwf_benchmark] eval session {session + 1}/{len(task_labels)} | "
            f"seen_classes={len(seen_labels)} | OA={oa:.4f} | AA={aa:.4f} | Kappa={kappa:.4f}"
        )
        print(f"[lwf_benchmark] per-task group accuracy: {task_group_accuracy}")

    result = {
        "algorithm": "lwf",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "task_split": [int(x) for x in args.task_split],
        "seen_metrics": stage_metrics,
        "average_forgetting": average_forgetting(stage_metrics),
    }
    with open(output_root / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[lwf_benchmark] result saved to {output_root / 'result.json'}")


if __name__ == "__main__":
    main()
