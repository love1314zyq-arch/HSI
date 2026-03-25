from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_ROOTS = {
    "paviau": "PaviaU",
    "salinas": "Salinas",
    "houston": "Houston",
    "indianpines": "IndianPines",
}


class HSIPatchDataset(Dataset):
    def __init__(self, root, dataset_key, train, seed, classes, patch_size=11, pca_dim=30):
        super().__init__()
        dataset_dir = DATASET_ROOTS[dataset_key]
        processed_root = Path(root) / dataset_dir / "processed"
        cube_path = processed_root / f"pca{pca_dim}_cube.npy"
        if not cube_path.exists():
            cube_path = processed_root / "pca30_cube.npy"
        gt_path = processed_root / "gt.npy"
        mask_path = processed_root / (f"train_mask_seed{seed}.npy" if train else f"test_mask_seed{seed}.npy")

        cube = np.load(cube_path).astype(np.float32)  # [H, W, C]
        cube = np.transpose(cube, (2, 0, 1))  # [C, H, W]
        gt = np.load(gt_path).astype(np.int64)
        mask = np.load(mask_path).astype(bool)

        labels = [int(x) for x in classes]
        select = np.logical_and(mask, np.isin(gt, labels))
        coords = np.argwhere(select)

        self.patch_size = int(patch_size)
        self.pad = self.patch_size // 2
        self.cube = np.pad(cube, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="reflect")
        self.coords = [(int(row), int(col)) for row, col in coords]
        self.targets = np.asarray([int(gt[int(row), int(col)]) for row, col in coords], dtype=np.int64)
        self.input_channels = int(cube.shape[0])

    def __getitem__(self, index):
        row, col = self.coords[index]
        row_pad = row + self.pad
        col_pad = col + self.pad
        image = self.cube[
            :,
            row_pad - self.pad : row_pad + self.pad + 1,
            col_pad - self.pad : col_pad + self.pad + 1,
        ]
        return torch.from_numpy(image).float(), int(self.targets[index])

    def __len__(self):
        return len(self.targets)
