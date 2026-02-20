import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils_hsi import ensure_dir


CLASS_NAMES = [
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Metal sheets",
    "Bare Soil",
    "Bitumen",
    "Bricks",
    "Shadows",
]

# Background + 9 classes.
PALETTE = np.array(
    [
        [255, 255, 255],  # background / unknown
        [0, 0, 0],  # 1 Asphalt
        [0, 255, 0],  # 2 Meadows
        [0, 139, 0],  # 3 Gravel
        [139, 90, 43],  # 4 Trees
        [128, 0, 128],  # 5 Metal sheets
        [255, 0, 0],  # 6 Bare Soil
        [255, 255, 0],  # 7 Bitumen
        [128, 128, 128],  # 8 Bricks
        [0, 0, 180],  # 9 Shadows
    ],
    dtype=np.uint8,
)


def _label_map_to_rgb(label_map: np.ndarray) -> np.ndarray:
    # -1 for background/unknown, 0..8 for classes.
    idx_map = np.clip(label_map + 1, 0, 9)
    return PALETTE[idx_map]


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
):
    ensure_dir(out_dir)
    h, w = gt.shape
    gt_seen = _make_seen_gt_map(gt, seen_classes)
    pred_map = _make_pred_map(h, w, rows, cols, preds, seen_classes)

    gt_rgb = _label_map_to_rgb(gt_seen)
    pred_rgb = _label_map_to_rgb(pred_map)

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
    top = 0.92
    step = 0.09
    for i, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE[1:]), start=1):
        y = top - (i - 1) * step
        ax3.add_patch(
            plt.Rectangle((0.05, y - 0.055), 0.22, 0.08, color=color / 255.0, transform=ax3.transAxes, clip_on=False)
        )
        text_color = "white" if i in [1, 5, 9] else "black"
        ax3.text(0.16, y - 0.015, f"{i}", ha="center", va="center", fontsize=14, color=text_color)
        ax3.text(0.31, y - 0.015, name, ha="left", va="center", fontsize=12)

    fig.text(0.5, 0.01, f"PaviaU Task {task_id} GT vs Prediction", ha="center", fontsize=14)
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
):
    """
    GT vs Pred visualization aligned to the same subset of pixels (usually the test set).

    Compared to save_task_comparison_figure(), this version masks the GT using the
    provided (rows, cols) positions so both GT and Pred are directly comparable.
    """
    ensure_dir(out_dir)
    h, w = gt.shape

    gt_test = _make_sparse_gt_map(h, w, gt, rows, cols, seen_classes)
    pred_map = _make_pred_map(h, w, rows, cols, preds, seen_classes)

    gt_rgb = _label_map_to_rgb(gt_test)
    pred_rgb = _label_map_to_rgb(pred_map)

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
    top = 0.92
    step = 0.09
    for i, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE[1:]), start=1):
        y = top - (i - 1) * step
        ax3.add_patch(
            plt.Rectangle((0.05, y - 0.055), 0.22, 0.08, color=color / 255.0, transform=ax3.transAxes, clip_on=False)
        )
        text_color = "white" if i in [1, 5, 9] else "black"
        ax3.text(0.16, y - 0.015, f"{i}", ha="center", va="center", fontsize=14, color=text_color)
        ax3.text(0.31, y - 0.015, name, ha="left", va="center", fontsize=12)

    fig.text(0.5, 0.01, f"PaviaU Task {task_id} GT(Test) vs Prediction(Test)", ha="center", fontsize=14)
    out_path = os.path.join(out_dir, f"task_{task_id}_gt_test_pred_test_compare.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(os.path.join(out_dir, f"task_{task_id}_gt_test.png"), gt_rgb)
    plt.imsave(os.path.join(out_dir, f"task_{task_id}_pred_test.png"), pred_rgb)
