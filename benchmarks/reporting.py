import csv
import io
import os
import pickle
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

from benchmarks.common import Scenario, cumulative_seen_classes
from benchmarks.visualizations import DATASET_ROOTS, load_display_spec, load_processed_scene, save_train_filled_task_figure
from utils_hsi import ensure_dir, save_json


def _mean(values: List[float]):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _std(values: List[float]):
    vals = [float(v) for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0 if vals else None
    return float(statistics.stdev(vals))


def _group_by_algorithm(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for item in results:
        grouped[item["algorithm"]].append(item)
    return dict(grouped)


def write_scenario_report(results: List[Dict[str, Any]], scenario: Scenario, output_root: str) -> None:
    report_root = os.path.join(output_root, "reports", scenario.name)
    ensure_dir(report_root)
    save_json(os.path.join(report_root, "benchmark_raw.json"), {"scenario": scenario.to_dict(), "results": results})

    grouped = _group_by_algorithm(results)
    csv_path = os.path.join(report_root, f"{scenario.name}_benchmark_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "status",
                "num_runs",
                "final_oa_mean",
                "final_oa_std",
                "final_aa_mean",
                "final_kappa_mean",
                "average_forgetting_mean",
                "reason",
            ]
        )
        for algorithm, rows in grouped.items():
            statuses = {row["status"] for row in rows}
            if statuses == {"success"}:
                status = "success"
                reason = ""
            else:
                status = ",".join(sorted(statuses))
                reason = " | ".join(sorted({row.get("reason", "") for row in rows if row.get("reason")}))
            writer.writerow(
                [
                    algorithm,
                    status,
                    len(rows),
                    _mean([row.get("final_oa") for row in rows]),
                    _std([row.get("final_oa") for row in rows]),
                    _mean([row.get("final_aa") for row in rows]),
                    _mean([row.get("final_kappa") for row in rows]),
                    _mean([row.get("average_forgetting") for row in rows]),
                    reason,
                ]
            )

    write_task_metric_tables(grouped, scenario, report_root)
    generate_task_visualizations(grouped, scenario, report_root)
    plot_incremental_compare(grouped, scenario, os.path.join(report_root, f"{scenario.name}_incremental_compare.png"))


def write_task_metric_tables(grouped: Dict[str, List[Dict[str, Any]]], scenario: Scenario, report_root: str) -> None:
    detailed_path = os.path.join(report_root, f"{scenario.name}_task_metrics_detailed.csv")
    with open(detailed_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "algorithm", "seed", "task_id", "seen_classes", "oa", "aa", "kappa", "status"])
        for algorithm, rows in grouped.items():
            for row in rows:
                metrics = row.get("seen_metrics", [])
                if not metrics:
                    writer.writerow([scenario.name, algorithm, row["seed"], "", "", "", "", "", row["status"]])
                    continue
                for task_metric in metrics:
                    writer.writerow(
                        [
                            scenario.name,
                            algorithm,
                            row["seed"],
                            task_metric.get("task_id", task_metric.get("session")),
                            task_metric.get("seen_classes"),
                            task_metric.get("oa"),
                            task_metric.get("aa"),
                            task_metric.get("kappa"),
                            row["status"],
                        ]
                    )

    mean_path = os.path.join(report_root, f"{scenario.name}_task_metrics_mean.csv")
    with open(mean_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "algorithm", "task_id", "seen_classes", "oa_mean", "oa_std", "aa_mean", "aa_std", "kappa_mean", "kappa_std"])
        for algorithm, rows in grouped.items():
            success_rows = [row for row in rows if row["status"] == "success" and row.get("seen_metrics")]
            if not success_rows:
                continue
            task_count = max(len(row["seen_metrics"]) for row in success_rows)
            for task_id in range(task_count):
                task_rows = [row["seen_metrics"][task_id] for row in success_rows if len(row["seen_metrics"]) > task_id]
                writer.writerow(
                    [
                        scenario.name,
                        algorithm,
                        task_id,
                        task_rows[0].get("seen_classes") if task_rows else None,
                        _mean([item.get("oa") for item in task_rows]),
                        _std([item.get("oa") for item in task_rows]),
                        _mean([item.get("aa") for item in task_rows]),
                        _std([item.get("aa") for item in task_rows]),
                        _mean([item.get("kappa") for item in task_rows]),
                        _std([item.get("kappa") for item in task_rows]),
                    ]
                )


def _dataset_display_name(dataset: str) -> str:
    return {
        "paviau": "PaviaU",
        "salinas": "Salinas",
        "houston": "Houston",
        "indianpines": "IndianPines",
    }[dataset]


def _iter_prediction_files(raw_result_path: str):
    if not raw_result_path:
        return []
    raw_path = Path(raw_result_path)
    base_dir = raw_path if raw_path.is_dir() else raw_path.parent
    pred_dir = base_dir / "task_predictions"
    if not pred_dir.exists():
        return []
    return sorted(pred_dir.glob("task_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))


FEICA_RAW_GT_FILES = {
    "paviau": "PaviaU_gt.mat",
    "salinas": "Salinas_gt.mat",
    "houston": "Houston_gt.mat",
    "indianpines": "Indian_pines_gt.mat",
}


def _load_raw_gt(dataset: str) -> np.ndarray:
    dataset_dir = DATASET_ROOTS[dataset]
    gt_filename = FEICA_RAW_GT_FILES[dataset]
    gt_path = Path(os.getcwd()) / "data" / dataset_dir / "raw" / gt_filename
    mat = loadmat(gt_path)
    gt_key = next(k for k in mat.keys() if not k.startswith("__"))
    return np.asarray(mat[gt_key], dtype=np.int64)


def _reconstruct_feica_split_coords(dataset: str, seed: int, split_rate: float = 0.2):
    processed_gt, _, _ = load_processed_scene(os.path.join(os.getcwd(), "data"), dataset, seed)
    raw_gt = _load_raw_gt(dataset)
    train_coords = {}
    test_coords = {}
    label_map = {}
    all_coords = np.argwhere(raw_gt > 0).astype(np.int64)
    all_raw_labels = raw_gt[all_coords[:, 0], all_coords[:, 1]].astype(np.int64)
    raw_class_ids = np.unique(all_raw_labels)
    standardized = np.empty_like(all_raw_labels)
    for feica_cls_id, raw_cls_id in enumerate(raw_class_ids):
        standardized[all_raw_labels == raw_cls_id] = feica_cls_id
        class_coords = all_coords[all_raw_labels == raw_cls_id]
        proc_labels = processed_gt[class_coords[:, 0], class_coords[:, 1]]
        proc_labels = proc_labels[proc_labels >= 0]
        label_map[feica_cls_id] = int(np.bincount(proc_labels).argmax())

    for feica_cls_id in np.unique(standardized):
        class_indices = np.nonzero(standardized == feica_cls_id)[0]
        np.random.seed(42)
        np.random.shuffle(class_indices)
        each_train_size = int(np.floor(split_rate * len(class_indices)))
        train_indices = class_indices[:each_train_size]
        test_indices = class_indices[each_train_size:]
        train_coords[int(feica_cls_id)] = all_coords[train_indices]
        test_coords[int(feica_cls_id)] = all_coords[test_indices]
    return train_coords, test_coords, label_map


def _ensure_feica_prediction_artifacts(row: Dict[str, Any], scenario: Scenario):
    raw_result_path = row.get("raw_result_path")
    if not raw_result_path or not os.path.exists(raw_result_path):
        return []
    run_dir = os.path.dirname(raw_result_path)
    pred_dir = os.path.join(run_dir, "predictions_0")
    if not os.path.isdir(pred_dir):
        return []

    dataset_gt, train_mask_unused, test_mask_unused = load_processed_scene(os.path.join(os.getcwd(), "data"), scenario.dataset, row["seed"])
    _ = train_mask_unused, test_mask_unused
    _, test_coords_by_class, label_map = _reconstruct_feica_split_coords(scenario.dataset, row["seed"], split_rate=0.2)
    seen_counts = cumulative_seen_classes(scenario.task_split)
    out_dir = os.path.join(run_dir, "task_predictions")
    ensure_dir(out_dir)
    for stale_file in Path(out_dir).glob("task_*.npz"):
        stale_file.unlink()

    for task_id, seen_count in enumerate(seen_counts):
        candidates = [
            os.path.join(pred_dir, f"{task_id:03d}.pkl"),
            os.path.join(pred_dir, f"{task_id:02d}.pkl"),
            os.path.join(pred_dir, f"{task_id}.pkl"),
        ]
        pkl_path = next((path for path in candidates if os.path.exists(path)), None)
        if pkl_path is None:
            continue
        with open(pkl_path, "rb") as f:
            original = torch.storage._load_from_bytes

            def _cpu_load(b):
                return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

            try:
                torch.storage._load_from_bytes = _cpu_load
                preds, labels = pickle.load(f)
            finally:
                torch.storage._load_from_bytes = original
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()
        if hasattr(labels, "cpu"):
            labels = labels.cpu().numpy()
        preds = np.asarray(preds, dtype=np.int64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        coords = []
        for cls_id in range(seen_count):
            coords.append(test_coords_by_class[int(cls_id)])
        coords = np.concatenate(coords, axis=0) if coords else np.empty((0, 2), dtype=np.int64)
        if len(coords) != len(preds):
            continue
        mapped_preds = np.asarray([label_map[int(p)] for p in preds], dtype=np.int64)
        mapped_labels = np.asarray([label_map[int(y)] for y in labels], dtype=np.int64)
        mapped_seen_classes = np.asarray([label_map[int(cls_id)] for cls_id in range(seen_count)], dtype=np.int64)
        np.savez_compressed(
            os.path.join(out_dir, f"task_{task_id}.npz"),
            task_id=np.asarray([task_id], dtype=np.int64),
            seen_classes=mapped_seen_classes,
            rows=coords[:, 0],
            cols=coords[:, 1],
            preds=mapped_preds,
            labels=mapped_labels,
        )
    return _iter_prediction_files(run_dir)


def generate_task_visualizations(grouped: Dict[str, List[Dict[str, Any]]], scenario: Scenario, report_root: str) -> None:
    dataset_name, class_names, palette = load_display_spec(os.path.join(os.getcwd(), "data"), scenario.dataset)
    vis_root = os.path.join(report_root, "task_visualizations")
    ensure_dir(vis_root)

    for algorithm, rows in grouped.items():
        for row in rows:
            if row["status"] != "success":
                continue
            gt, train_mask, test_mask = load_processed_scene(os.path.join(os.getcwd(), "data"), scenario.dataset, row["seed"])
            if algorithm == "feica_cil":
                pred_files = _ensure_feica_prediction_artifacts(row, scenario)
                train_coords_by_class, test_coords_by_class, _ = _reconstruct_feica_split_coords(scenario.dataset, row["seed"], split_rate=0.2)
                visible_train = np.zeros_like(train_mask)
                visible_test = np.zeros_like(test_mask)
                for seen_cls in range(sum(scenario.task_split)):
                    if seen_cls in train_coords_by_class:
                        coords = train_coords_by_class[seen_cls]
                        visible_train[coords[:, 0], coords[:, 1]] = True
                    if seen_cls in test_coords_by_class:
                        coords = test_coords_by_class[seen_cls]
                        visible_test[coords[:, 0], coords[:, 1]] = True
                train_mask_use = visible_train
                test_mask_use = visible_test
            else:
                pred_files = _iter_prediction_files(row.get("raw_result_path"))
                train_mask_use = train_mask
                test_mask_use = test_mask
            if not pred_files:
                continue
            seed_dir = os.path.join(vis_root, algorithm, f"seed_{row['seed']}")
            ensure_dir(seed_dir)
            for pred_file in pred_files:
                data = np.load(pred_file)
                task_id = int(data["task_id"][0])
                save_train_filled_task_figure(
                    out_dir=seed_dir,
                    algorithm=algorithm,
                    scenario_name=scenario.name,
                    task_id=task_id,
                    dataset_name=dataset_name,
                    class_names=class_names,
                    palette=palette,
                    gt=gt,
                    train_mask=train_mask_use,
                    test_mask=test_mask_use,
                    seen_classes=data["seen_classes"].tolist(),
                    rows=data["rows"],
                    cols=data["cols"],
                    preds=data["preds"],
                )


def plot_incremental_compare(grouped: Dict[str, List[Dict[str, Any]]], scenario: Scenario, out_png: str) -> None:
    seen_classes = cumulative_seen_classes(scenario.task_split)
    plt.figure(figsize=(8.2, 5.4))
    plotted = False
    for algorithm, rows in grouped.items():
        success_rows = [row for row in rows if row["status"] == "success" and row.get("seen_metrics")]
        if not success_rows:
            continue
        task_count = len(success_rows[0]["seen_metrics"])
        means = []
        for task_id in range(task_count):
            vals = [row["seen_metrics"][task_id]["oa"] * 100.0 for row in success_rows]
            means.append(statistics.mean(vals))
        xs = seen_classes[:task_count]
        plt.plot(xs, means, marker="o", linewidth=2.2, label=algorithm)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Number of seen classes", fontsize=12)
    plt.ylabel("Overall Accuracy (%)", fontsize=12)
    plt.title(f"Incremental comparison on {scenario.dataset}", fontsize=13)
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
