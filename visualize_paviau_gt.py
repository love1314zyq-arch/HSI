import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


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

# Match common PaviaU visualization colors.
CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 0),  # 1 Asphalt
    (0, 255, 0),  # 2 Meadows
    (0, 139, 0),  # 3 Gravel
    (139, 90, 43),  # 4 Trees
    (128, 0, 128),  # 5 Metal sheets
    (255, 0, 0),  # 6 Bare Soil
    (255, 255, 0),  # 7 Bitumen
    (128, 128, 128),  # 8 Bricks
    (0, 0, 180),  # 9 Shadows
]


def _load_cube_and_gt(data_root: str):
    raw_dir = os.path.join(data_root, "raw")
    cube_mat = loadmat(os.path.join(raw_dir, "PaviaU.mat"))
    gt_mat = loadmat(os.path.join(raw_dir, "PaviaU_gt.mat"))

    cube = cube_mat.get("paviaU", None)
    gt = gt_mat.get("paviaU_gt", None)
    if cube is None:
        cube = next(v for v in cube_mat.values() if isinstance(v, np.ndarray) and v.ndim == 3)
    if gt is None:
        gt = next(v for v in gt_mat.values() if isinstance(v, np.ndarray) and v.ndim == 2)
    return cube.astype(np.float32), gt.astype(np.int64)


def _percentile_stretch(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[2]):
        ch = x[:, :, i]
        lo = np.percentile(ch, low)
        hi = np.percentile(ch, high)
        ch = np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out[:, :, i] = ch
    return out


def _make_false_color(cube: np.ndarray, rgb_bands: Tuple[int, int, int]) -> np.ndarray:
    c = cube.shape[2]
    r, g, b = [max(0, min(c - 1, k)) for k in rgb_bands]
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    return _percentile_stretch(rgb)


def _make_gt_rgb(gt: np.ndarray) -> np.ndarray:
    h, w = gt.shape
    out = np.ones((h, w, 3), dtype=np.float32)
    for cls_id in range(1, 10):
        color = np.array(CLASS_COLORS[cls_id - 1], dtype=np.float32) / 255.0
        out[gt == cls_id] = color
    return out


def _pick_font():
    # Try common CJK fonts; fallback to default if unavailable.
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def render_paviau_figure(data_root: str, out_path: str, rgb_bands: Tuple[int, int, int]):
    cube, gt = _load_cube_and_gt(data_root)
    false_color = _make_false_color(cube, rgb_bands=rgb_bands)
    gt_rgb = _make_gt_rgb(gt)

    _pick_font()
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2], wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(false_color)
    ax1.set_axis_off()
    ax1.set_title("(a) 伪彩色图", y=-0.11, fontsize=18)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_rgb)
    ax2.set_axis_off()
    ax2.set_title("(b) 真值图", y=-0.11, fontsize=18)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.text(0.15, -0.02, "(c) 颜色标识", fontsize=18)

    top = 0.90
    h = 0.085
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS), start=1):
        y = top - (i - 1) * h
        ax3.add_patch(
            plt.Rectangle(
                (0.06, y - h + 0.005),
                0.22,
                h,
                color=np.array(color) / 255.0,
                transform=ax3.transAxes,
                clip_on=False,
            )
        )
        txt_color = "white" if i in [1, 5, 9] else "black"
        ax3.text(0.17, y - h / 2, f"{i}", ha="center", va="center", fontsize=20, color=txt_color, fontfamily="serif")
        ax3.text(0.31, y - h / 2, name, ha="left", va="center", fontsize=16, color="black", fontfamily="serif")

    fig.text(0.5, 0.035, "图 2.3   PaviaU 高光谱图像数据集示意图", ha="center", fontsize=22)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Render PaviaU false-color + GT + legend figure.")
    parser.add_argument("--data_root", type=str, default="data/PaviaU", help="Path to dataset root containing raw/")
    parser.add_argument("--out", type=str, default="outputs/visualizations/paviau_dataset_overview.png")
    parser.add_argument(
        "--bands",
        type=int,
        nargs=3,
        default=[55, 30, 5],
        help="RGB band indices for false-color image (0-based).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_paviau_figure(args.data_root, args.out, tuple(args.bands))
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
