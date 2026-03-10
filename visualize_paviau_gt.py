import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from matplotlib import font_manager

from preprocess_hsi import DATASET_SPECS, infer_dataset_name


DATASET_COLORS: Dict[str, List[Tuple[int, int, int]]] = {
    "PaviaU": [
        (0, 0, 0),
        (0, 255, 0),
        (0, 139, 0),
        (139, 90, 43),
        (128, 0, 128),
        (255, 0, 0),
        (255, 255, 0),
        (128, 128, 128),
        (0, 0, 180),
    ],
    "Salinas": [
        (255, 0, 0),
        (0, 104, 122),
        (132, 216, 239),
        (38, 17, 193),
        (34, 255, 34),
        (187, 187, 187),
        (162, 163, 33),
        (236, 248, 33),
        (58, 48, 232),
        (125, 7, 112),
        (244, 161, 239),
        (255, 107, 226),
        (194, 69, 52),
        (106, 23, 19),
        (77, 47, 117),
        (0, 0, 0),
    ],
}

DEFAULT_RGB_BANDS = {
    "PaviaU": (55, 30, 5),
    "Salinas": (24, 13, 6),
}


def _load_cube_and_gt(data_root: str):
    dataset_name = infer_dataset_name(data_root)
    spec = DATASET_SPECS[dataset_name]
    raw_dir = os.path.join(data_root, "raw")

    cube_mat = loadmat(os.path.join(raw_dir, spec["cube_file"]))
    gt_mat = loadmat(os.path.join(raw_dir, spec["gt_file"]))

    cube = cube_mat.get(spec["cube_key"], None)
    gt = gt_mat.get(spec["gt_key"], None)
    if cube is None:
        cube = next(v for v in cube_mat.values() if isinstance(v, np.ndarray) and v.ndim == 3)
    if gt is None:
        gt = next(v for v in gt_mat.values() if isinstance(v, np.ndarray) and v.ndim == 2)
    return dataset_name, cube.astype(np.float32), gt.astype(np.int64)


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


def _make_gt_rgb(gt: np.ndarray, class_colors: List[Tuple[int, int, int]]) -> np.ndarray:
    h, w = gt.shape
    out = np.ones((h, w, 3), dtype=np.float32)
    for cls_id, color in enumerate(class_colors, start=1):
        out[gt == cls_id] = np.array(color, dtype=np.float32) / 255.0
    return out


def _pick_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"


def _has_cjk_font() -> bool:
    candidates = {"SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC", "Noto Sans CJK JP"}
    for path in font_manager.findSystemFonts(fontpaths=None, fontext="ttf"):
        try:
            if font_manager.FontProperties(fname=path).get_name() in candidates:
                return True
        except Exception:
            continue
    return False


def _text_color_for_patch(color: Tuple[int, int, int]) -> str:
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return "white" if luminance < 110 else "black"


def render_dataset_figure(data_root: str, out_path: str, rgb_bands: Tuple[int, int, int] = None):
    dataset_name, cube, gt = _load_cube_and_gt(data_root)
    spec = DATASET_SPECS[dataset_name]
    class_names = spec["class_names"]
    class_colors = DATASET_COLORS[dataset_name]
    if len(class_names) != len(class_colors):
        raise ValueError(f"{dataset_name} class_names and class_colors length mismatch")

    rgb_bands = DEFAULT_RGB_BANDS[dataset_name] if rgb_bands is None else rgb_bands
    false_color = _make_false_color(cube, rgb_bands=rgb_bands)
    gt_rgb = _make_gt_rgb(gt, class_colors)

    _pick_font()
    use_chinese = _has_cjk_font()
    fig = plt.figure(figsize=(13, 7.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(false_color)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_rgb)
    ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    top = 0.94 if len(class_names) > 12 else 0.90
    step = 0.055 if len(class_names) > 12 else 0.085
    rect_h = 0.048 if len(class_names) > 12 else 0.08
    num_fs = 10 if len(class_names) > 12 else 20
    txt_fs = 10 if len(class_names) > 12 else 16

    for i, (name, color) in enumerate(zip(class_names, class_colors), start=1):
        y = top - (i - 1) * step
        ax3.add_patch(
            plt.Rectangle(
                (0.06, y - rect_h + 0.004),
                0.18,
                rect_h,
                color=np.array(color) / 255.0,
                transform=ax3.transAxes,
                clip_on=False,
            )
        )
        txt_color = _text_color_for_patch(color)
        ax3.text(0.15, y - rect_h / 2, f"{i}", ha="center", va="center", fontsize=num_fs, color=txt_color)
        ax3.text(0.28, y - rect_h / 2, name, ha="left", va="center", fontsize=txt_fs, color="black")

    if use_chinese:
        caption_a = "(a) 伪彩色图"
        caption_b = "(b) 真值图"
        caption_c = "(c) 颜色标识"
        title = f"{dataset_name} 高光谱图像数据集示意图"
    else:
        caption_a = "(a) False-color"
        caption_b = "(b) Ground Truth"
        caption_c = "(c) Legend"
        title = f"{dataset_name} HSI Dataset Overview"

    fig.text(0.19, 0.08, caption_a, ha="center", fontsize=18)
    fig.text(0.48, 0.08, caption_b, ha="center", fontsize=18)
    fig.text(0.80, 0.08, caption_c, ha="center", fontsize=18)
    fig.text(0.5, 0.02, title, ha="center", fontsize=22)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.subplots_adjust(bottom=0.16, top=0.98)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def render_paviau_figure(data_root: str, out_path: str, rgb_bands: Tuple[int, int, int]):
    render_dataset_figure(data_root, out_path, rgb_bands)


def parse_args():
    parser = argparse.ArgumentParser(description="Render HSI false-color + GT + legend figure.")
    parser.add_argument("--data_root", type=str, default="data/PaviaU", help="Path to dataset root containing raw/")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--bands",
        type=int,
        nargs=3,
        default=None,
        help="RGB band indices for false-color image (0-based). Default depends on dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = infer_dataset_name(args.data_root)
    out_path = args.out
    if out_path is None:
        out_path = os.path.join("outputs", "visualizations", f"{dataset_name.lower()}_dataset_overview.png")
    bands = None if args.bands is None else tuple(args.bands)
    render_dataset_figure(args.data_root, out_path, bands)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
