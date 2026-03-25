import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils_hsi import ensure_dir


PALETTE_MAP = {
    "PaviaU": np.array(
        [
            [255, 255, 255],
            [0, 0, 0],
            [0, 255, 0],
            [0, 139, 0],
            [139, 90, 43],
            [128, 0, 128],
            [255, 0, 0],
            [255, 255, 0],
            [128, 128, 128],
            [0, 0, 180],
        ],
        dtype=np.uint8,
    ),
    "Salinas": np.array(
        [
            [255, 255, 255],
            [255, 0, 0],
            [0, 104, 122],
            [132, 216, 239],
            [38, 17, 193],
            [34, 255, 34],
            [187, 187, 187],
            [162, 163, 33],
            [236, 248, 33],
            [58, 48, 232],
            [125, 7, 112],
            [244, 161, 239],
            [255, 107, 226],
            [194, 69, 52],
            [106, 23, 19],
            [77, 47, 117],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    ),
    "IndianPines": np.array(
        [
            [255, 255, 255],
            [208, 140, 40],   # Alfalfa
            [188, 235, 74],   # Corn-notill
            [150, 168, 54],   # Corn-mintill
            [120, 255, 36],   # Corn
            [201, 92, 68],    # Grass-pasture
            [218, 165, 58],   # Grass-trees
            [205, 32, 32],    # Grass-pasture-mowed
            [192, 138, 42],   # Hay-windrowed
            [255, 78, 255],   # Oats
            [22, 126, 126],   # Soybean-notill
            [88, 70, 178],    # Soybean-mintill
            [184, 230, 150],  # Soybean-clean
            [184, 204, 36],   # Wheat
            [53, 126, 53],    # Woods
            [92, 224, 255],   # Buildings-Grass-Trees-Drives
            [96, 124, 124],   # Stone-Steel-Towers
        ],
        dtype=np.uint8,
    ),
    "Houston": np.array(
        [
            [0, 0, 0],
            [220, 38, 54],    # Healthy grass
            [26, 107, 126],   # Stressed grass
            [117, 217, 247],  # Synthetic grass
            [74, 55, 191],    # Trees
            [50, 255, 50],    # Soil
            [42, 166, 42],    # Water
            [166, 170, 47],   # Residential
            [247, 240, 54],   # Commercial
            [79, 65, 241],    # Road
            [159, 58, 217],   # Highway
            [113, 34, 79],    # Railway
            [244, 146, 236],  # Parking Lot 1
            [192, 69, 50],    # Parking Lot 2
            [156, 107, 53],   # Tennis court
            [152, 152, 152],  # Running track
        ],
        dtype=np.uint8,
    ),
}


def _get_palette(dataset_name: str, class_count: int) -> np.ndarray:
    base_palette = PALETTE_MAP.get(dataset_name)
    if base_palette is None or base_palette.shape[0] != class_count + 1:
        palette = np.zeros((class_count + 1, 3), dtype=np.uint8)
        palette[0] = np.array([255, 255, 255], dtype=np.uint8)
        for index in range(1, class_count + 1):
            palette[index] = np.array(
                [
                    (37 * index) % 256,
                    (97 * index) % 256,
                    (157 * index) % 256,
                ],
                dtype=np.uint8,
            )
        return palette
    return base_palette


def _label_map_to_rgb(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    idx_map = np.clip(label_map + 1, 0, palette.shape[0] - 1)
    return palette[idx_map]


def _make_seen_gt_map(gt: np.ndarray, seen_classes: Iterable[int]) -> np.ndarray:
    seen_set = set(int(c) for c in seen_classes)
    out = np.full_like(gt, fill_value=-1)
    for c in seen_set:
        out[gt == c] = c
    return out


def _make_pred_map(
    h: int, w: int, rows: np.ndarray, cols: np.ndarray, preds: np.ndarray, seen_classes: Iterable[int]
) -> np.ndarray:
    out = np.full((h, w), fill_value=-1, dtype=np.int64)
    seen_set = set(int(c) for c in seen_classes)
    for r, c, p in zip(rows.tolist(), cols.tolist(), preds.tolist()):
        if int(p) in seen_set:
            out[int(r), int(c)] = int(p)
    return out


def _make_sparse_gt_map(
    h: int, w: int, gt: np.ndarray, rows: np.ndarray, cols: np.ndarray, seen_classes: Iterable[int]
) -> np.ndarray:
    """
    Build a sparse GT map aligned with a subset of pixels (e.g., test pixels).

    Only pixels listed in (rows, cols) are filled. Others are set to -1.
    Unseen classes are also set to -1 to match the "Seen" visualization semantics.
    """
    out = np.full((h, w), fill_value=-1, dtype=np.int64)
    seen_set = set(int(c) for c in seen_classes)
    for r, c in zip(rows.tolist(), cols.tolist()):
        rr, cc = int(r), int(c)
        lab = int(gt[rr, cc])
        if lab in seen_set:
            out[rr, cc] = lab
    return out


def save_task_comparison_figure(
    out_dir: str,
    task_id: int,
    gt: np.ndarray,
    seen_classes: List[int],
    rows: np.ndarray,
    cols: np.ndarray,
    preds: np.ndarray,
    dataset_name: str,
    class_names: List[str],
):
    ensure_dir(out_dir)
    h, w = gt.shape
    palette = _get_palette(dataset_name, len(class_names))
    gt_seen = _make_seen_gt_map(gt, seen_classes)
    pred_map = _make_pred_map(h, w, rows, cols, preds, seen_classes)

    gt_rgb = _label_map_to_rgb(gt_seen, palette)
    pred_rgb = _label_map_to_rgb(pred_map, palette)

    fig = plt.figure(figsize=(12, 5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_rgb)
    ax1.set_axis_off()
    ax1.set_title(f"(a) Task {task_id} GT (Seen)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred_rgb)
    ax2.set_axis_off()
    ax2.set_title(f"(b) Task {task_id} Pred")

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

    fig.text(0.5, 0.01, f"{dataset_name} Task {task_id} GT vs Prediction", ha="center", fontsize=14)
    out_path = os.path.join(out_dir, f"task_{task_id}_gt_pred_compare.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Also save raw maps for further post-processing.
    plt.imsave(os.path.join(out_dir, f"task_{task_id}_gt_seen.png"), gt_rgb)
    plt.imsave(os.path.join(out_dir, f"task_{task_id}_pred.png"), pred_rgb)


def save_task_test_aligned_comparison_figure(
    out_dir: str,
    task_id: int,
    gt: np.ndarray,
    seen_classes: List[int],
    rows: np.ndarray,
    cols: np.ndarray,
    preds: np.ndarray,
    dataset_name: str,
    class_names: List[str],
):
    """
    GT vs Pred visualization aligned to the same subset of pixels (usually the test set).

    Compared to save_task_comparison_figure(), this version masks the GT using the
    provided (rows, cols) positions so both GT and Pred are directly comparable.
    """
    ensure_dir(out_dir)
    h, w = gt.shape
    palette = _get_palette(dataset_name, len(class_names))

    gt_test = _make_sparse_gt_map(h, w, gt, rows, cols, seen_classes)
    pred_map = _make_pred_map(h, w, rows, cols, preds, seen_classes)

    gt_rgb = _label_map_to_rgb(gt_test, palette)
    pred_rgb = _label_map_to_rgb(pred_map, palette)

    fig = plt.figure(figsize=(12, 5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_rgb)
    ax1.set_axis_off()
    ax1.set_title(f"(a) Task {task_id} GT (Test)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred_rgb)
    ax2.set_axis_off()
    ax2.set_title(f"(b) Task {task_id} Pred (Test)")

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

    fig.text(0.5, 0.01, f"{dataset_name} Task {task_id} GT(Test) vs Prediction(Test)", ha="center", fontsize=14)
    out_path = os.path.join(out_dir, f"task_{task_id}_gt_test_pred_test_compare.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(os.path.join(out_dir, f"task_{task_id}_gt_test.png"), gt_rgb)
    plt.imsave(os.path.join(out_dir, f"task_{task_id}_pred_test.png"), pred_rgb)
