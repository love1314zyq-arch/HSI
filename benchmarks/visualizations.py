import os
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from task_visualize_hsi import PALETTE_MAP, _get_palette, _label_map_to_rgb
from utils_hsi import ensure_dir


DATASET_ROOTS = {
    "paviau": "PaviaU",
    "salinas": "Salinas",
    "houston": "Houston",
    "indianpines": "IndianPines",
}


def load_display_spec(data_root: str, dataset: str) -> Tuple[str, List[str], np.ndarray]:
    dataset_dir = DATASET_ROOTS[dataset]
    dataset_name = dataset_dir
    metadata_path = Path(data_root) / dataset_dir / "metadata" / "dataset_info.json"

    if not metadata_path.exists():
        class_names = [f"Class {i+1}" for i in range(32)]
        return dataset_name, class_names, _get_palette(dataset_name, len(class_names))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = metadata.get("class_order_names")
    class_order_original_labels = metadata.get("class_order_original_labels")
    if not class_names or not class_order_original_labels:
        class_names = [f"Class {i+1}" for i in range(32)]
        return dataset_name, class_names, _get_palette(dataset_name, len(class_names))

    base_palette = PALETTE_MAP.get(dataset_name)
    if base_palette is None:
        return dataset_name, class_names, _get_palette(dataset_name, len(class_names))

    reordered_palette = np.zeros((len(class_names) + 1, 3), dtype=np.uint8)
    reordered_palette[0] = base_palette[0]
    for new_idx, original_label in enumerate(class_order_original_labels, start=1):
        reordered_palette[new_idx] = base_palette[int(original_label)]
    return dataset_name, class_names, reordered_palette


def load_processed_scene(data_root: str, dataset: str, seed: int):
    dataset_dir = DATASET_ROOTS[dataset]
    processed_root = Path(data_root) / dataset_dir / "processed"
    gt = np.load(processed_root / "gt.npy").astype(np.int64)
    train_mask = np.load(processed_root / f"train_mask_seed{seed}.npy").astype(bool)
    test_mask = np.load(processed_root / f"test_mask_seed{seed}.npy").astype(bool)
    return gt, train_mask, test_mask


def make_seen_dense_gt_map(gt: np.ndarray, seen_classes: Iterable[int], train_mask: np.ndarray, test_mask: np.ndarray):
    seen_set = set(int(c) for c in seen_classes)
    out = np.full_like(gt, fill_value=-1)
    visible = np.logical_or(train_mask, test_mask)
    for cls in seen_set:
        cls_mask = np.logical_and(visible, gt == cls)
        out[cls_mask] = cls
    return out


def make_train_filled_pred_map(
    gt: np.ndarray,
    train_mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    preds: np.ndarray,
    seen_classes: Iterable[int],
):
    seen_set = set(int(c) for c in seen_classes)
    out = np.full_like(gt, fill_value=-1)
    for cls in seen_set:
        cls_mask = np.logical_and(train_mask, gt == cls)
        out[cls_mask] = cls
    for row, col, pred in zip(rows.tolist(), cols.tolist(), preds.tolist()):
        if int(pred) in seen_set:
            out[int(row), int(col)] = int(pred)
    return out


def save_train_filled_task_figure(
    out_dir: str,
    algorithm: str,
    scenario_name: str,
    task_id: int,
    dataset_name: str,
    class_names: List[str],
    palette: np.ndarray,
    gt: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seen_classes: List[int],
    rows: np.ndarray,
    cols: np.ndarray,
    preds: np.ndarray,
):
    ensure_dir(out_dir)
    gt_dense = make_seen_dense_gt_map(gt, seen_classes, train_mask, test_mask)
    pred_dense = make_train_filled_pred_map(gt, train_mask, rows, cols, preds, seen_classes)

    gt_rgb = _label_map_to_rgb(gt_dense, palette)
    pred_rgb = _label_map_to_rgb(pred_dense, palette)

    fig = plt.figure(figsize=(12, 5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_rgb)
    ax1.set_axis_off()
    ax1.set_title(f"(a) Task {task_id} GT (Seen)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred_rgb)
    ax2.set_axis_off()
    ax2.set_title(f"(b) Task {task_id} Pred (Train+Test)")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    top = 0.94 if len(class_names) > 12 else 0.92
    step = 0.055 if len(class_names) > 12 else 0.09
    font_size = 10 if len(class_names) > 12 else 12
    for i, (name, color) in enumerate(zip(class_names, palette[1:]), start=1):
        y = top - (i - 1) * step
        ax3.add_patch(
            plt.Rectangle((0.05, y - 0.055), 0.22, 0.08, color=color / 255.0, transform=ax3.transAxes, clip_on=False)
        )
        text_color = "white" if float(np.mean(color)) < 110 else "black"
        ax3.text(0.16, y - 0.015, f"{i}", ha="center", va="center", fontsize=14, color=text_color)
        ax3.text(0.31, y - 0.015, name, ha="left", va="center", fontsize=font_size)

    fig.text(0.5, 0.01, f"{algorithm} | {scenario_name} | Task {task_id}", ha="center", fontsize=14)
    stem = f"{scenario_name}_{algorithm}_task{task_id}"
    fig.savefig(os.path.join(out_dir, f"{stem}_gt_pred_trainfilled_compare.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(os.path.join(out_dir, f"{stem}_gt_seen.png"), gt_rgb)
    plt.imsave(os.path.join(out_dir, f"{stem}_pred_trainfilled.png"), pred_rgb)


def save_task_prediction_artifact(
    output_root: str,
    task_id: int,
    seen_classes: List[int],
    rows: np.ndarray,
    cols: np.ndarray,
    preds: np.ndarray,
    labels: Optional[np.ndarray] = None,
):
    task_dir = Path(output_root) / "task_predictions"
    task_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "task_id": np.asarray([int(task_id)], dtype=np.int64),
        "seen_classes": np.asarray(seen_classes, dtype=np.int64),
        "rows": np.asarray(rows, dtype=np.int64),
        "cols": np.asarray(cols, dtype=np.int64),
        "preds": np.asarray(preds, dtype=np.int64),
    }
    if labels is not None:
        save_dict["labels"] = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(task_dir / f"task_{task_id}.npz", **save_dict)
