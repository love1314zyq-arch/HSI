from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_ROOTS = {
    "PaviaU": "PaviaU",
    "Salinas": "Salinas",
    "Houston": "Houston",
    "IndianPines": "IndianPines",
}


def _extract_patch(cube, row, col, patch_size):
    pad = patch_size // 2
    padded = np.pad(cube, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    row_pad = row + pad
    col_pad = col + pad
    return padded[:, row_pad - pad : row_pad + pad + 1, col_pad - pad : col_pad + pad + 1]


class HSIPatchDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, args=None, download=False):
        super().__init__()
        _ = download
        if args is None:
            raise ValueError("HSIPatchDataset requires args.")
        dataset_dir = DATASET_ROOTS.get(args.dataset_name)
        if dataset_dir is None:
            raise KeyError(f"Unsupported HSI dataset for SSRE: {args.dataset_name}")

        processed_root = Path(root) / dataset_dir / "processed"
        pca_dim = int(getattr(args, "pca_dim", 30))
        patch_size = int(getattr(args, "patch_size", getattr(args, "patches", 11)))
        input_mode = str(getattr(args, "hsi_input_mode", "pixel")).lower()
        if input_mode == "pixel":
            cube_path = processed_root / "full_cube.npy"
            if not cube_path.exists():
                cube_path = processed_root / f"pca{pca_dim}_cube.npy"
                if not cube_path.exists():
                    cube_path = processed_root / "pca30_cube.npy"
                print(
                    f"[ssre_hsi_dataset] warning: full_cube.npy not found for {args.dataset_name}, "
                    f"falling back to {cube_path.name}"
                )
        else:
            cube_path = processed_root / f"pca{pca_dim}_cube.npy"
            if not cube_path.exists():
                cube_path = processed_root / "pca30_cube.npy"
        gt_path = processed_root / "gt.npy"
        seed = int(args.seed)
        mask_path = processed_root / (f"train_mask_seed{seed}.npy" if train else f"test_mask_seed{seed}.npy")
        split_name = "train" if train else "test"
        print(
            f"[ssre_hsi_dataset] loading {args.dataset_name} {split_name} split | "
            f"cube={cube_path} | gt={gt_path} | mask={mask_path} | mode={input_mode} | patch={patch_size}"
        )

        cube = np.load(cube_path).astype(np.float32)  # [H, W, C]
        gt = np.load(gt_path).astype(np.int64)
        mask = np.load(mask_path).astype(bool)
        labeled = np.logical_and(mask, gt >= 0)
        self.input_mode = input_mode
        self.patch_size = patch_size
        if self.input_mode == "patch":
            cube = np.transpose(cube, (2, 0, 1))  # [C, H, W]
            pad = patch_size // 2
            self.cube = np.pad(cube, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
            self.pad = pad
            self.spectral_dim = int(self.cube.shape[0])
        else:
            self.cube = cube  # [H, W, C]
            self.pad = 0
            self.spectral_dim = int(self.cube.shape[-1])

        self.transform = transform
        self.target_transform = target_transform
        self.coords = []
        self.targets = []
        self.sub_indexes = {}

        class_ids = sorted(int(x) for x in np.unique(gt[labeled]))
        for cls_id in class_ids:
            coords = np.argwhere(np.logical_and(labeled, gt == cls_id))
            indexes = []
            for row, col in coords:
                indexes.append(len(self.coords))
                self.coords.append((int(row), int(col)))
                self.targets.append(int(cls_id))
            self.sub_indexes[int(cls_id)] = np.asarray(indexes, dtype=np.int64)

        self.targets = np.asarray(self.targets, dtype=np.int64)
        print(
            f"[ssre_hsi_dataset] prepared {args.dataset_name} {split_name} split | "
            f"samples={len(self.targets)} | classes={class_ids}"
        )
        self.TrainIndexes = np.asarray([], dtype=np.int64)
        self.TrainLabels = np.asarray([], dtype=np.int64)
        self.TestIndexes = np.asarray([], dtype=np.int64)
        self.TestLabels = np.asarray([], dtype=np.int64)

    def _normalize_classes(self, classes):
        if isinstance(classes, tuple):
            if len(classes) == 2 and all(isinstance(x, (int, np.integer)) for x in classes):
                return list(range(int(classes[0]), int(classes[1])))
            return [int(x) for x in classes]
        if isinstance(classes, (list, np.ndarray)):
            return [int(x) for x in classes]
        raise TypeError(f"Unsupported class specifier: {classes}")

    def getTrainData(self, classes):
        labels = self._normalize_classes(classes)
        mask = np.isin(self.targets, labels)
        self.TrainIndexes = np.where(mask)[0]
        self.TrainLabels = self.targets[mask]
        if self.input_mode == "patch":
            train_shape = (len(self.TrainIndexes), self.cube.shape[0], self.patch_size, self.patch_size)
        else:
            train_shape = (len(self.TrainIndexes), self.cube.shape[-1], 1, 1)
        print("the size of train set is %s" % (str(train_shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTestData_up2now(self, classes):
        labels = self._normalize_classes(classes)
        mask = np.isin(self.targets, labels)
        self.TestIndexes = np.where(mask)[0]
        self.TestLabels = self.targets[mask]
        if self.input_mode == "patch":
            test_shape = (len(self.TestIndexes), self.cube.shape[0], self.patch_size, self.patch_size)
        else:
            test_shape = (len(self.TestIndexes), self.cube.shape[-1], 1, 1)
        print("the size of test set is %s" % (str(test_shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def _get_patch(self, base_index):
        row, col = self.coords[int(base_index)]
        if self.input_mode == "patch":
            row_pad = row + self.pad
            col_pad = col + self.pad
            return self.cube[:, row_pad - self.pad : row_pad + self.pad + 1, col_pad - self.pad : col_pad + self.pad + 1]
        spectrum = self.cube[row, col, :]
        return np.asarray(spectrum, dtype=np.float32)[:, None, None]

    def getTrainItem(self, index):
        img = torch.from_numpy(self._get_patch(self.TrainIndexes[index])).float()
        target = int(self.TrainLabels[index])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def getTestItem(self, index):
        img = torch.from_numpy(self._get_patch(self.TestIndexes[index])).float()
        target = int(self.TestLabels[index])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainIndexes) != 0:
            return self.getTrainItem(index)
        if len(self.TestIndexes) != 0:
            return self.getTestItem(index)
        raise IndexError("Dataset view is empty; call getTrainData or getTestData_up2now first.")

    def __len__(self):
        if len(self.TrainIndexes) != 0:
            return len(self.TrainIndexes)
        if len(self.TestIndexes) != 0:
            return len(self.TestIndexes)
        return 0
