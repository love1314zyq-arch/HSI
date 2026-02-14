import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


# Patch 级数据集：
# 给定像素中心位置，从高光谱立方体中切出固定大小 patch。
class PatchDataset(Dataset):
    def __init__(self, cube: np.ndarray, gt: np.ndarray, positions: np.ndarray, patch_size: int):
        self.cube = cube
        self.gt = gt
        self.positions = positions
        self.patch_size = patch_size
        self.pad = patch_size // 2

        # 使用 reflect 填充边界，避免边缘像素无法取 patch。
        self.cube_pad = np.pad(cube, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        r, c = self.positions[idx]
        rp = r + self.pad
        cp = c + self.pad

        patch = self.cube_pad[rp - self.pad: rp + self.pad + 1, cp - self.pad: cp + self.pad + 1, :]
        patch = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float()
        label = int(self.gt[r, c])

        # 额外返回像素位置，便于后续可视化或误差分析。
        return patch, label, int(r), int(c)


# 数据管理器：
# 1) 读取预处理结果
# 2) 构建固定增量任务（5+2+2）
# 3) 根据任务返回训练/测试数据集
class PaviaUDataManager:
    def __init__(
        self,
        root: str,
        seed: int = 1993,
        patch_size: int = 11,
        base_classes: int = 5,
        task_num: int = 2,
        pca_dim: int = 30,
    ):
        self.root = root
        self.seed = seed
        self.patch_size = patch_size
        self.base_classes = base_classes
        self.task_num = task_num
        self.pca_dim = pca_dim

        processed_dir = os.path.join(root, "processed")
        metadata_path = os.path.join(root, "metadata", "dataset_info.json")

        cube_path = os.path.join(processed_dir, f"pca{pca_dim}_cube.npy")
        if not os.path.exists(cube_path):
            legacy_path = os.path.join(processed_dir, "pca30_cube.npy")
            if os.path.exists(legacy_path):
                cube_path = legacy_path
            else:
                raise FileNotFoundError(f"Missing PCA cube file: {cube_path}")
        self.cube = np.load(cube_path)
        self.gt = np.load(os.path.join(processed_dir, "gt.npy"))
        self.train_mask = np.load(os.path.join(processed_dir, f"train_mask_seed{seed}.npy"))
        self.test_mask = np.load(os.path.join(processed_dir, f"test_mask_seed{seed}.npy"))

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.class_order_names = meta["class_order_names"]
        self.num_classes = int(meta["num_classes"])

        self.tasks = self._build_tasks()

    # 构建增量任务划分：
    # 例如 total=9, base=5, task_num=2 -> [0..4], [5,6], [7,8]
    def _build_tasks(self) -> List[List[int]]:
        all_classes = list(range(self.num_classes))
        base = all_classes[: self.base_classes]
        remaining = all_classes[self.base_classes:]
        if self.task_num <= 0:
            return [base]

        # 将剩余类别尽量均匀分配到 task_num 个增量任务中。
        chunk_sizes = [len(remaining) // self.task_num] * self.task_num
        for i in range(len(remaining) % self.task_num):
            chunk_sizes[i] += 1

        tasks = [base]
        cursor = 0
        for sz in chunk_sizes:
            tasks.append(remaining[cursor: cursor + sz])
            cursor += sz
        return tasks

    def get_task_classes(self, task_id: int) -> List[int]:
        return self.tasks[task_id]

    def get_seen_classes(self, task_id: int) -> List[int]:
        seen = []
        for i in range(task_id + 1):
            seen.extend(self.tasks[i])
        return seen

    def get_seen_class_count(self, task_id: int) -> int:
        return len(self.get_seen_classes(task_id))

    # 按类别 + 掩码筛选像素坐标。
    def _collect_positions(self, classes: List[int], split: str) -> np.ndarray:
        if split == "train":
            mask = self.train_mask
        elif split == "test":
            mask = self.test_mask
        else:
            raise ValueError("split must be 'train' or 'test'")

        selected = np.zeros_like(mask, dtype=np.bool_)
        for cls in classes:
            selected |= (self.gt == cls)

        final_mask = np.logical_and(selected, mask)
        positions = np.argwhere(final_mask)
        return positions

    def get_task_dataset(self, task_id: int, split: str = "train") -> PatchDataset:
        classes = self.get_task_classes(task_id)
        positions = self._collect_positions(classes, split)
        return PatchDataset(self.cube, self.gt, positions, patch_size=self.patch_size)

    def get_seen_dataset(self, task_id: int, split: str = "test") -> PatchDataset:
        classes = self.get_seen_classes(task_id)
        positions = self._collect_positions(classes, split)
        return PatchDataset(self.cube, self.gt, positions, patch_size=self.patch_size)

    # 用于“按任务分别测试”场景。
    def get_taskwise_test_dataset(self, task_id: int, eval_task_id: int) -> PatchDataset:
        _ = task_id
        classes = self.get_task_classes(eval_task_id)
        positions = self._collect_positions(classes, split="test")
        return PatchDataset(self.cube, self.gt, positions, patch_size=self.patch_size)


