from typing import Dict, List, Tuple

import numpy as np


class PrototypeAugmentorHSI:
    """Prototype augmentation for HSI feature replay.

    Supported modes:
    - ``gaussian``: the original isotropic Gaussian perturbation with a global radius
    - ``spectral_diag``: class-wise diagonal covariance perturbation, better aligned
      with HSI feature anisotropy while keeping the original prototype pipeline intact
    """

    def __init__(
        self,
        mode: str = "gaussian",
        diag_floor_ratio: float = 0.15,
        diag_eps: float = 1e-6,
    ):
        self.mode = str(mode).strip().lower()
        if self.mode not in {"gaussian", "spectral_diag"}:
            raise ValueError(f"Unsupported prototype augmentation mode: {self.mode}")
        self.diag_floor_ratio = float(diag_floor_ratio)
        self.diag_eps = float(diag_eps)

    def build_statistics(
        self,
        feature_bank: Dict[int, List[np.ndarray]],
        current_task_id: int,
        feature_dim: int,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float]:
        """Build class prototypes, optional diagonal scales, and global radius."""

        prototype_dict: Dict[int, np.ndarray] = {}
        diag_scale_dict: Dict[int, np.ndarray] = {}
        radius_list = []

        for cls, feats in feature_bank.items():
            feats_arr = np.asarray(feats, dtype=np.float32)
            if feats_arr.ndim != 2 or feats_arr.shape[0] == 0:
                continue

            prototype_dict[int(cls)] = np.mean(feats_arr, axis=0).astype(np.float32)

            if feats_arr.shape[0] > 1:
                var = np.var(feats_arr, axis=0, ddof=1).astype(np.float32)
            else:
                var = np.zeros((feature_dim,), dtype=np.float32)

            diag_scale_dict[int(cls)] = np.sqrt(np.maximum(var, self.diag_eps)).astype(np.float32)

            if current_task_id == 0 and feats_arr.shape[0] > 1:
                cov = np.cov(feats_arr.T)
                radius_list.append(np.trace(cov) / feats_arr.shape[1])

        radius = 0.0
        if current_task_id == 0 and len(radius_list) > 0:
            radius = float(np.sqrt(np.mean(radius_list)))

        if self.mode == "spectral_diag" and radius > 0:
            floor = radius * self.diag_floor_ratio
            for cls, scale in list(diag_scale_dict.items()):
                diag_scale_dict[cls] = np.maximum(scale, floor).astype(np.float32)

        return prototype_dict, diag_scale_dict, radius

    def sample(
        self,
        prototype_dict: Dict[int, np.ndarray],
        diag_scale_dict: Dict[int, np.ndarray],
        radius: float,
        batch_size: int,
        old_class_count: int,
        feature_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample perturbed prototype features for old classes."""

        if old_class_count <= 0 or len(prototype_dict) == 0:
            return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

        proto_aug = []
        proto_aug_label = []
        available_classes = [cls for cls in range(old_class_count) if cls in prototype_dict]
        if len(available_classes) == 0:
            return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

        for _ in range(batch_size):
            cls = int(np.random.choice(available_classes))
            proto = prototype_dict[cls]

            if self.mode == "spectral_diag" and cls in diag_scale_dict:
                scale = diag_scale_dict[cls]
                noise = np.random.normal(0.0, 1.0, feature_dim).astype(np.float32) * scale
            else:
                noise = np.random.normal(0.0, 1.0, feature_dim).astype(np.float32) * float(radius)

            proto_aug.append((proto + noise).astype(np.float32))
            proto_aug_label.append(cls)

        return (
            np.asarray(proto_aug, dtype=np.float32),
            np.asarray(proto_aug_label, dtype=np.int64),
        )
