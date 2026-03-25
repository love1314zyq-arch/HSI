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


class HSIPixelDataset(Dataset):
    def __init__(self, root, dataset_key, train, seed, classes, patch_size=1, pca_dim=30):
        super().__init__()
        dataset_dir = DATASET_ROOTS[dataset_key]
        processed_root = Path(root) / dataset_dir / "processed"
        cube_path = processed_root / "full_cube.npy"
        if not cube_path.exists():
            cube_path = processed_root / f"pca{pca_dim}_cube.npy"
            if not cube_path.exists():
                cube_path = processed_root / "pca30_cube.npy"
        gt_path = processed_root / "gt.npy"
        mask_path = processed_root / (f"train_mask_seed{seed}.npy" if train else f"test_mask_seed{seed}.npy")

        cube = np.load(cube_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.int64)
        mask = np.load(mask_path).astype(bool)

        labels = [int(x) for x in classes]
        select = np.logical_and(mask, np.isin(gt, labels))
        coords = np.argwhere(select)

        self.patch_size = int(patch_size)
        self.cube = cube
        self.coords = [(int(row), int(col)) for row, col in coords]
        self.targets = np.asarray([int(gt[int(row), int(col)]) for row, col in coords], dtype=np.int64)
        self.input_channels = int(cube.shape[-1])

    def __getitem__(self, index):
        row, col = self.coords[index]
        spectrum = self.cube[row, col, :]
        image = np.asarray(spectrum, dtype=np.float32)[:, None, None]
        return torch.from_numpy(image).float(), int(self.targets[index])

    def __len__(self):
        return len(self.targets)
