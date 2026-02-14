import argparse
import os
import ssl
import urllib.request
from typing import Dict, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

from utils_hsi import ensure_dir, save_json, set_seed


# PaviaU 官方源 + 备用源。
PAVIAU_URLS = {
    "PaviaU.mat": [
        "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "https://raw.githubusercontent.com/eecn/Hyperspectral-Classification/master/Data/PaviaU.mat",
    ],
    "PaviaU_gt.mat": [
        "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "https://raw.githubusercontent.com/eecn/Hyperspectral-Classification/master/Data/PaviaU_gt.mat",
    ],
}

# PaviaU 9 个类别名称（不包含背景）。
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


# 尝试多个 URL 下载同一文件，直到成功。
def _download_file(urls, dst_path):
    if os.path.exists(dst_path):
        return

    last_err = None
    # 某些 Windows 环境证书链不完整，使用非验证上下文提高可用性。
    ssl_ctx = ssl._create_unverified_context()

    for url in urls:
        try:
            print(f"[download] {url} -> {dst_path}")
            with urllib.request.urlopen(url, context=ssl_ctx, timeout=120) as response:
                content = response.read()
            with open(dst_path, "wb") as f:
                f.write(content)
            if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
                return
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Failed to download {dst_path}: {last_err}")


# 下载原始数据文件到 raw 目录。
def download_paviau(root_dir: str):
    raw_dir = os.path.join(root_dir, "raw")
    ensure_dir(raw_dir)
    for name, urls in PAVIAU_URLS.items():
        _download_file(urls, os.path.join(raw_dir, name))


# 读取 .mat 文件并提取数据立方体与标签图。
def _extract_cube_and_gt(raw_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    cube_mat = loadmat(os.path.join(raw_dir, "PaviaU.mat"))
    gt_mat = loadmat(os.path.join(raw_dir, "PaviaU_gt.mat"))

    cube = cube_mat.get("paviaU", None)
    gt = gt_mat.get("paviaU_gt", None)

    # 兼容不同命名风格的 mat 文件。
    if cube is None:
        cube = next(v for v in cube_mat.values() if isinstance(v, np.ndarray) and v.ndim == 3)
    if gt is None:
        gt = next(v for v in gt_mat.values() if isinstance(v, np.ndarray) and v.ndim == 2)

    return cube.astype(np.float32), gt.astype(np.int64)


# 只基于有标签像素统计均值/方差进行标准化。
def _normalize_labeled_pixels(cube: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h, w, c = cube.shape
    flat = cube.reshape(-1, c)
    label_mask = gt.reshape(-1) > 0

    labeled = flat[label_mask]
    mean = labeled.mean(axis=0, keepdims=True)
    std = labeled.std(axis=0, keepdims=True) + 1e-8

    flat_norm = (flat - mean) / std
    return flat_norm.reshape(h, w, c)


# 将原始标签（1~9）映射为连续标签（0~8），背景置为 -1。
def _remap_labels(gt: np.ndarray, class_order_original: np.ndarray) -> np.ndarray:
    mapped_gt = np.full_like(gt, fill_value=-1)
    for new_idx, original_label in enumerate(class_order_original):
        mapped_gt[gt == original_label] = new_idx
    return mapped_gt


# 为每个类别按比例切分 train/test 掩码。
def _build_train_test_masks(mapped_gt: np.ndarray, train_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    h, w = mapped_gt.shape
    train_mask = np.zeros((h, w), dtype=np.bool_)
    test_mask = np.zeros((h, w), dtype=np.bool_)

    for cls in np.unique(mapped_gt):
        if cls < 0:
            continue
        pos = np.argwhere(mapped_gt == cls)
        rng.shuffle(pos)
        n_train = max(1, int(len(pos) * train_ratio))
        train_pos = pos[:n_train]
        test_pos = pos[n_train:]
        train_mask[train_pos[:, 0], train_pos[:, 1]] = True
        test_mask[test_pos[:, 0], test_pos[:, 1]] = True

    return train_mask, test_mask


# 完整预处理流程：读取 -> 标准化 -> PCA -> 切分 -> 保存。
def prepare_processed_data(root_dir: str, pca_dim: int, train_ratio: float, seed: int):
    raw_dir = os.path.join(root_dir, "raw")
    processed_dir = os.path.join(root_dir, "processed")
    splits_dir = os.path.join(root_dir, "splits")
    metadata_dir = os.path.join(root_dir, "metadata")
    ensure_dir(processed_dir)
    ensure_dir(splits_dir)
    ensure_dir(metadata_dir)

    cube, gt = _extract_cube_and_gt(raw_dir)
    cube_norm = _normalize_labeled_pixels(cube, gt)

    # 固定随机顺序，决定增量学习类别出现顺序。
    rng = np.random.default_rng(seed)
    class_order_original = np.arange(1, 10)
    rng.shuffle(class_order_original)

    mapped_gt = _remap_labels(gt, class_order_original)

    h, w, c = cube_norm.shape
    flat = cube_norm.reshape(-1, c)

    # PCA 降维，减少谱维冗余，降低模型复杂度。
    pca = PCA(n_components=pca_dim, random_state=seed)
    flat_pca = pca.fit_transform(flat)
    cube_pca = flat_pca.reshape(h, w, pca_dim).astype(np.float32)

    train_mask, test_mask = _build_train_test_masks(mapped_gt, train_ratio=train_ratio, seed=seed)

    np.save(os.path.join(processed_dir, f"pca{pca_dim}_cube.npy"), cube_pca)
    np.save(os.path.join(processed_dir, "gt.npy"), mapped_gt)
    np.save(os.path.join(processed_dir, f"train_mask_seed{seed}.npy"), train_mask)
    np.save(os.path.join(processed_dir, f"test_mask_seed{seed}.npy"), test_mask)

    # 保存每类样本统计，便于论文中描述数据划分。
    split_info: Dict[str, Dict[str, int]] = {}
    for cls in range(9):
        n_train = int(np.sum(np.logical_and(mapped_gt == cls, train_mask)))
        n_test = int(np.sum(np.logical_and(mapped_gt == cls, test_mask)))
        split_info[str(cls)] = {"train": n_train, "test": n_test}

    save_json(
        os.path.join(splits_dir, f"split_seed{seed}_train20.json"),
        {
            "seed": seed,
            "train_ratio": train_ratio,
            "split_per_class": split_info,
        },
    )

    ordered_class_names = [CLASS_NAMES[idx - 1] for idx in class_order_original]
    save_json(
        os.path.join(metadata_dir, "dataset_info.json"),
        {
            "dataset": "PaviaU",
            "shape": [int(h), int(w), int(c)],
            "pca_dim": pca_dim,
            "num_classes": 9,
            "class_order_original_labels": class_order_original.tolist(),
            "class_order_names": ordered_class_names,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare PaviaU dataset for PASS-HSI")
    parser.add_argument("--root", type=str, default="data/PaviaU")
    parser.add_argument("--pca_dim", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if not args.skip_download:
        download_paviau(args.root)

    prepare_processed_data(args.root, pca_dim=args.pca_dim, train_ratio=args.train_ratio, seed=args.seed)
    print("PaviaU preprocessing completed.")


if __name__ == "__main__":
    main()


