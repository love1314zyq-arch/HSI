import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.io import loadmat


def main():
    root = "data/IndianPines"
    out_dir = "outputs/visualizations/indianpines"
    os.makedirs(out_dir, exist_ok=True)

    cube = loadmat(os.path.join(root, "raw", "Indian_pines_corrected.mat"))["indian_pines_corrected"].astype(np.float32)
    mapped_gt = np.load(os.path.join(root, "processed", "gt.npy"))
    meta = json.load(open(os.path.join(root, "metadata", "dataset_info.json"), "r", encoding="utf-8"))
    class_names = meta["class_order_names"]

    colors = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
        (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    ]

    # False color image using a simple NIR-Red-Green style band combination.
    bands = (29, 19, 9)
    rgb = np.stack([cube[:, :, b] for b in bands], axis=-1)
    for i in range(3):
        channel = rgb[:, :, i]
        lo = np.percentile(channel, 1)
        hi = np.percentile(channel, 99)
        rgb[:, :, i] = np.clip((channel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(out_dir, "indianpines_false_color.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    gt_rgb = np.ones((mapped_gt.shape[0], mapped_gt.shape[1], 3), dtype=np.float32)
    for cls_id, color in enumerate(colors, start=1):
        gt_rgb[mapped_gt == cls_id] = np.array(color, dtype=np.float32) / 255.0

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(gt_rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(out_dir, "indianpines_ground_truth.png"), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows_per_col = 8
    start_y = 0.935
    box_h = 0.085
    step = box_h
    col_specs = [
        {"patch_x": 0.09, "text_x": 0.17, "index_x": 0.125},
        {"patch_x": 0.47, "text_x": 0.55, "index_x": 0.505},
    ]
    for i, (name, color) in enumerate(zip(class_names, colors), start=1):
        col_idx = (i - 1) // rows_per_col
        row_idx = (i - 1) % rows_per_col
        spec = col_specs[col_idx]
        y = start_y - row_idx * step
        ax.add_patch(
            Rectangle(
                (spec["patch_x"], y - box_h / 2),
                0.075,
                box_h,
                color=np.array(color) / 255.0,
                transform=ax.transAxes,
            )
        )
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        txt_color = "white" if luminance < 120 else "black"
        ax.text(
            spec["index_x"],
            y,
            str(i),
            ha="center",
            va="center",
            fontsize=9,
            color=txt_color,
            transform=ax.transAxes,
        )
        ax.text(
            spec["text_x"],
            y,
            name,
            ha="left",
            va="center",
            fontsize=9.5,
            color="black",
            transform=ax.transAxes,
        )
    fig.tight_layout(pad=0.05)
    fig.savefig(os.path.join(out_dir, "indianpines_legend.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    for name in ("indianpines_false_color.png", "indianpines_ground_truth.png", "indianpines_legend.png"):
        print(os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
