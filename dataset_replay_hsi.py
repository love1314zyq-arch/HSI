from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MixedReplayDataset(Dataset):
    """Concatenates current-task samples with raw replay exemplars."""

    def __init__(
        self,
        current_dataset: Dataset,
        memory_images: np.ndarray,
        memory_labels: np.ndarray,
        memory_rows: Optional[np.ndarray] = None,
        memory_cols: Optional[np.ndarray] = None,
    ):
        self.current_dataset = current_dataset
        self.memory_images = np.asarray(memory_images, dtype=np.float32)
        self.memory_labels = np.asarray(memory_labels, dtype=np.int64)
        self.memory_rows = (
            np.asarray(memory_rows, dtype=np.int64)
            if memory_rows is not None
            else np.full((len(self.memory_labels),), -1, dtype=np.int64)
        )
        self.memory_cols = (
            np.asarray(memory_cols, dtype=np.int64)
            if memory_cols is not None
            else np.full((len(self.memory_labels),), -1, dtype=np.int64)
        )

        if len(self.memory_images) != len(self.memory_labels):
            raise ValueError("memory_images and memory_labels must have the same length")

    def __len__(self) -> int:
        return len(self.current_dataset) + len(self.memory_labels)

    def __getitem__(self, idx: int):
        current_len = len(self.current_dataset)
        if idx < current_len:
            return self.current_dataset[idx]

        memory_idx = idx - current_len
        image = torch.from_numpy(self.memory_images[memory_idx]).float()
        label = int(self.memory_labels[memory_idx])
        row = int(self.memory_rows[memory_idx])
        col = int(self.memory_cols[memory_idx])
        return image, label, row, col
