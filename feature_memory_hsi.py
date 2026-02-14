from collections import defaultdict, deque
from typing import Dict, Tuple

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
