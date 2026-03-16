from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import torch


class FeatureMemoryBank:
    """Stores class-balanced feature exemplars without raw image replay."""

    def __init__(self, memory_per_class: int = 40):
        self.memory_per_class = int(memory_per_class)
        self._bank: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.memory_per_class))

    def add(self, features: np.ndarray, labels: np.ndarray):
        for feat, cls in zip(features, labels):
            self._bank[int(cls)].append(np.asarray(feat, dtype=np.float32))

    def has_data(self) -> bool:
        return any(len(v) > 0 for v in self._bank.values())

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        available_classes = [cls for cls, feats in self._bank.items() if len(feats) > 0]
        if len(available_classes) == 0:
            return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)

        sampled_feats = []
        sampled_labels = []
        rng = np.random.default_rng()
        for _ in range(batch_size):
            cls = int(rng.choice(available_classes))
            feats = self._bank[cls]
            idx = int(rng.integers(0, len(feats)))
            sampled_feats.append(feats[idx])
            sampled_labels.append(cls)

        feat_t = torch.from_numpy(np.asarray(sampled_feats, dtype=np.float32)).to(device)
        label_t = torch.from_numpy(np.asarray(sampled_labels, dtype=np.int64)).to(device)
        return feat_t, label_t

    def state_dict(self):
        return {str(k): [x.tolist() for x in v] for k, v in self._bank.items()}

    def load_state_dict(self, state):
        self._bank.clear()
        for k, feats in state.items():
            cls = int(k)
            q = deque(maxlen=self.memory_per_class)
            for feat in feats:
                q.append(np.asarray(feat, dtype=np.float32))
            self._bank[cls] = q


class RawMemoryBank:
    """Stores class-balanced raw patch exemplars for rehearsal."""

    def __init__(self, memory_per_class: int = 20):
        self.memory_per_class = int(memory_per_class)
        self._bank: Dict[int, List[np.ndarray]] = defaultdict(list)

    def add(self, images: np.ndarray, labels: np.ndarray):
        for image, cls in zip(images, labels):
            cls = int(cls)
            self._bank[cls].append(np.asarray(image, dtype=np.float32))
            if len(self._bank[cls]) > self.memory_per_class:
                self._bank[cls] = self._bank[cls][-self.memory_per_class :]

    def set_class(self, cls: int, images: np.ndarray):
        self.set_class_with_limit(cls, images, self.memory_per_class)

    def set_class_with_limit(self, cls: int, images: np.ndarray, limit: int):
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 0:
            self._bank[int(cls)] = []
            return
        limit = max(0, int(limit))
        self._bank[int(cls)] = [img for img in images[:limit]]

    def get_class(self, cls: int) -> np.ndarray:
        images = self._bank.get(int(cls), [])
        if len(images) == 0:
            return np.empty((0,), dtype=np.float32)
        return np.asarray(images, dtype=np.float32)

    def trim_class(self, cls: int, limit: int):
        cls = int(cls)
        limit = max(0, int(limit))
        if cls not in self._bank:
            return
        self._bank[cls] = self._bank[cls][:limit]

    def classes(self) -> List[int]:
        return sorted(self._bank.keys())

    def export_all(self) -> Tuple[np.ndarray, np.ndarray]:
        exported_images = []
        exported_labels = []
        for cls in self.classes():
            cls_images = self._bank[cls]
            if len(cls_images) == 0:
                continue
            exported_images.extend(cls_images)
            exported_labels.extend([cls] * len(cls_images))

        if len(exported_images) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        return (
            np.asarray(exported_images, dtype=np.float32),
            np.asarray(exported_labels, dtype=np.int64),
        )

    def has_data(self) -> bool:
        return any(len(v) > 0 for v in self._bank.values())

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        available_classes = [cls for cls, images in self._bank.items() if len(images) > 0]
        if len(available_classes) == 0:
            return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)

        sampled_images = []
        sampled_labels = []
        rng = np.random.default_rng()
        for _ in range(batch_size):
            cls = int(rng.choice(available_classes))
            cls_images = self._bank[cls]
            idx = int(rng.integers(0, len(cls_images)))
            sampled_images.append(cls_images[idx])
            sampled_labels.append(cls)

        image_t = torch.from_numpy(np.asarray(sampled_images, dtype=np.float32)).to(device)
        label_t = torch.from_numpy(np.asarray(sampled_labels, dtype=np.int64)).to(device)
        return image_t, label_t

    def state_dict(self):
        return {str(k): [x.tolist() for x in v] for k, v in self._bank.items()}

    def load_state_dict(self, state):
        self._bank.clear()
        for k, images in state.items():
            cls = int(k)
            self._bank[cls] = [np.asarray(image, dtype=np.float32) for image in images[: self.memory_per_class]]
