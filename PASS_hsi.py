import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from dataset_replay_hsi import MixedReplayDataset
from feature_memory_hsi import FeatureMemoryBank, RawMemoryBank
from metrics_hsi import evaluate_all
from prototype_augmentation_hsi import PrototypeAugmentorHSI
from replay_selection import icarl_selection
from task_visualize_hsi import save_task_test_aligned_comparison_figure
from utils_hsi import ensure_dir


class ProtoAugSSLHSI:
    def __init__(self, cfg: Dict, data_manager, model, device: torch.device):
        self.cfg = cfg
        self.data_manager = data_manager
        self.model = model
        self.device = device

        self.batch_size = int(cfg["batch_size"])
        self.num_workers = int(cfg["num_workers"])
        self.learning_rate = float(cfg["learning_rate"])
        self.weight_decay = float(cfg["weight_decay"])

        pass_cfg = cfg.get("pass", {})
        self.temp = float(pass_cfg.get("temp", 0.1))
        self.proto_weight = float(pass_cfg.get("protoAug_weight", 10.0))
        self.kd_weight = float(pass_cfg.get("kd_weight", 10.0))
        self.prototype_aug_mode = str(pass_cfg.get("prototype_aug_mode", "gaussian")).lower()
        self.prototype_aug_diag_floor_ratio = float(pass_cfg.get("prototype_aug_diag_floor_ratio", 0.15))
        self.ssl_mode = self._resolve_ssl_mode(pass_cfg)
        self.ssl_factor = self._ssl_factor_for_mode(self.ssl_mode)
        self.ssma_apply_prob = float(pass_cfg.get("ssma_apply_prob", 0.5))
        self.ssma_mask_prob = float(pass_cfg.get("ssma_mask_prob", 0.3))
        self.prototype_augmentor = PrototypeAugmentorHSI(
            mode=self.prototype_aug_mode,
            diag_floor_ratio=self.prototype_aug_diag_floor_ratio,
        )

        self.train_cfg = cfg.get("train", {})
        self.optimizer_name = str(self.train_cfg.get("optimizer", "adam")).lower()
        self.scheduler_name = str(self.train_cfg.get("scheduler", "step")).lower()

        selection_cfg = cfg.get("task_selection", {})
        self.select_best_model = bool(selection_cfg.get("enable_best_model", False))
        self.early_stop_min_epochs = int(selection_cfg.get("early_stop_min_epochs", 0))
        self.early_stop_drop_ratio = float(selection_cfg.get("early_stop_drop_ratio", 0.0))
        self.selection_metric = str(selection_cfg.get("metric", "oa")).lower()

        replay_cfg = cfg.get("replay", {})
        self.replay_enable = bool(replay_cfg.get("enable", False))
        self.replay_mode = str(replay_cfg.get("mode", "feature")).lower()
        self.replay_weight = float(replay_cfg.get("lambda_replay", 1.0))
        self.align_weight = float(replay_cfg.get("lambda_align", 0.0))
        self.memory_per_class = int(replay_cfg.get("memory_per_class", 40))
        self.replay_batch_size = int(replay_cfg.get("batch_size", self.batch_size))
        self.replay_strategy = str(replay_cfg.get("strategy", "loss")).lower()
        self.replay_selection = str(replay_cfg.get("selection", "fifo")).lower()
        self.replay_fixed_budget = bool(replay_cfg.get("fixed_budget", False))
        self.replay_memory_budget = int(replay_cfg.get("memory_budget", 0))
        self.replay_full_replay_below_train_count = int(replay_cfg.get("full_replay_below_train_count", 0))
        self.replay_min_memory_per_class = int(replay_cfg.get("min_memory_per_class", 0))
        if self.replay_strategy not in {"loss", "merged"}:
            raise ValueError(f"Unsupported replay.strategy: {self.replay_strategy}")
        if self.replay_selection not in {"fifo", "herding"}:
            raise ValueError(f"Unsupported replay.selection: {self.replay_selection}")
        self.memory_bank = FeatureMemoryBank(memory_per_class=self.memory_per_class)
        self.raw_memory_bank = RawMemoryBank(memory_per_class=self.memory_per_class)

        bias_cfg = cfg.get("bias_correction", {})
        self.bias_correction_enable = bool(bias_cfg.get("enable", False))

        ft_cfg = cfg.get("balanced_finetune", {})
        self.ft_enable = bool(ft_cfg.get("enable", False))
        self.ft_epochs = int(ft_cfg.get("epochs", 0))
        self.ft_lr = float(ft_cfg.get("learning_rate", self.learning_rate * 0.5))

        self.old_model = None
        self.prototype_dict: Dict[int, np.ndarray] = {}
        self.prototype_diag_scale_dict: Dict[int, np.ndarray] = {}
        self.radius = 0.0
        self.feature_dim = model.feature_dim

        self.train_loader = None
        self.test_loader = None
        self.current_task_id = 0
        self.current_seen_count = 0

    @staticmethod
    def _resolve_ssl_mode(pass_cfg: Dict) -> str:
        ssl_mode = str(pass_cfg.get("ssl_mode", "")).strip().lower()
        if ssl_mode:
            if ssl_mode not in {"none", "rotation4", "spectral3", "ssma", "auto"}:
                raise ValueError(f"Unsupported pass.ssl_mode: {ssl_mode}")
            if ssl_mode == "auto":
                raise ValueError("pass.ssl_mode=auto must be resolved in main_hsi.py before trainer creation")
            return ssl_mode

        if bool(pass_cfg.get("use_rotation_ssl", True)):
            return "rotation4"
        return "none"

    @staticmethod
    def _ssl_factor_for_mode(ssl_mode: str) -> int:
        if ssl_mode == "rotation4":
            return 4
        if ssl_mode == "spectral3":
            return 3
        return 1

    def _apply_ssma(self, images: torch.Tensor) -> torch.Tensor:
        # Source-style SSMA preprocessing: CenterCropResize + HorizontalFlip + MaskMixed.
        aug = images.clone()
        batch_size, channels, height, width = aug.shape
        if channels <= 0 or height <= 1 or width <= 1:
            return aug

        crop_candidates = []
        for ratio in (21 / 27, 27 / 27, 25 / 27, 23 / 27):
            crop_h = max(1, min(height, int(round(height * ratio))))
            crop_w = max(1, min(width, int(round(width * ratio))))
            crop_candidates.append((crop_h, crop_w))
        crop_candidates = list(dict.fromkeys(crop_candidates))
        keep_prob = 1.0 - self.ssma_mask_prob

        for idx in range(batch_size):
            crop_h, crop_w = crop_candidates[np.random.randint(0, len(crop_candidates))]
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)
            sample = aug[idx : idx + 1, :, top : top + crop_h, left : left + crop_w]
            if crop_h != height or crop_w != width:
                sample = F.interpolate(sample, size=(height, width), mode="bilinear", align_corners=False)

            if torch.rand(1, device=aug.device).item() < 0.5:
                sample = torch.flip(sample, dims=[3])

            if torch.rand(1, device=aug.device).item() < self.ssma_apply_prob:
                spatial_mask = torch.bernoulli(
                    torch.full((height, width), keep_prob, device=aug.device, dtype=sample.dtype)
                )
                spatial_mask = spatial_mask.unsqueeze(0).expand(channels, height, width)

                spectral_mask = torch.bernoulli(
                    torch.full((channels, 1, 1), keep_prob, device=aug.device, dtype=sample.dtype)
                )
                spectral_mask = spectral_mask.expand(channels, height, width)

                mixed_mask = (1.0 - spectral_mask) * spatial_mask + spectral_mask
                fill_value = torch.rand(1, device=aug.device, dtype=sample.dtype).item() * 0.5
                fill_tensor = (1.0 - mixed_mask) * fill_value
                sample = sample.squeeze(0) * mixed_mask + fill_tensor
                aug[idx] = sample
            else:
                aug[idx] = sample.squeeze(0)

        return aug

    def _augment_ssl(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ssl_mode == "rotation4":
            images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], dim=1)
            b, r, c, h, w = images.shape
            images = images.view(b * r, c, h, w)
            labels = torch.stack([labels * self.ssl_factor + k for k in range(self.ssl_factor)], dim=1).view(-1)
            return images, labels

        if self.ssl_mode == "spectral3":
            reverse_images = torch.flip(images, dims=[1])
            negate_images = -images + 1.0
            images = torch.stack((images, reverse_images, negate_images), dim=1)
            b, r, c, h, w = images.shape
            images = images.view(b * r, c, h, w)
            labels = torch.stack([labels * self.ssl_factor + k for k in range(self.ssl_factor)], dim=1).view(-1)
            return images, labels

        if self.ssl_mode == "ssma":
            return self._apply_ssma(images), labels

        return images, labels

    def _primary_logits(self, logits: torch.Tensor, seen_count: int) -> torch.Tensor:
        logits = logits[:, : seen_count * self.ssl_factor]
        if self.ssl_factor > 1:
            logits = logits[:, :: self.ssl_factor]
        return logits

    def _build_optimizer(self, params, lr: float = None):
        lr = self.learning_rate if lr is None else lr
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        return torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)

    def _build_scheduler(self, optimizer, epochs: int):
        if self.scheduler_name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        return StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)

    def _get_seen_train_class_counts(self, task_id: int) -> Dict[int, int]:
        counts = {}
        train_mask = self.data_manager.train_mask
        gt = self.data_manager.gt
        for cls in self.data_manager.get_seen_classes(task_id):
            counts[int(cls)] = int(np.logical_and(gt == cls, train_mask).sum())
        return counts

    def _compute_raw_memory_limits(self, task_id: int) -> Dict[int, int]:
        seen_classes = [int(cls) for cls in self.data_manager.get_seen_classes(task_id)]
        if not seen_classes:
            return {}

        if not self.replay_fixed_budget or self.replay_memory_budget <= 0:
            return {cls: self.memory_per_class for cls in seen_classes}

        counts = self._get_seen_train_class_counts(task_id)
        threshold = self.replay_full_replay_below_train_count
        min_memory = max(0, self.replay_min_memory_per_class)
        limits = {cls: 0 for cls in seen_classes}
        forced_budget = 0

        for cls in seen_classes:
            cls_count = max(0, counts.get(cls, 0))
            if threshold > 0 and cls_count < threshold:
                limits[cls] = cls_count
            elif min_memory > 0:
                limits[cls] = min(min_memory, cls_count)
            forced_budget += limits[cls]

        remaining_budget = max(0, self.replay_memory_budget - forced_budget)
        capacities = {
            cls: max(0, counts.get(cls, 0) - limits[cls])
            for cls in seen_classes
            if counts.get(cls, 0) - limits[cls] > 0
        }
        weights = {cls: max(1, counts.get(cls, 0)) for cls in capacities}

        while remaining_budget > 0 and capacities:
            total_weight = sum(weights.values())
            if total_weight <= 0:
                break

            increments = {cls: 0 for cls in capacities}
            remainders = []
            allocated_now = 0
            for cls in list(capacities):
                ideal = remaining_budget * weights[cls] / total_weight
                add = min(capacities[cls], int(np.floor(ideal)))
                if add > 0:
                    increments[cls] = add
                    allocated_now += add
                remainders.append((ideal - np.floor(ideal), cls))

            if allocated_now == 0:
                for _, cls in sorted(remainders, reverse=True):
                    if remaining_budget <= 0:
                        break
                    if capacities.get(cls, 0) <= 0:
                        continue
                    limits[cls] += 1
                    capacities[cls] -= 1
                    remaining_budget -= 1
                    if capacities[cls] == 0:
                        capacities.pop(cls, None)
                        weights.pop(cls, None)
                continue

            remaining_budget -= allocated_now
            for cls, add in increments.items():
                if add <= 0 or cls not in capacities:
                    continue
                limits[cls] += add
                capacities[cls] -= add
                if capacities[cls] == 0:
                    capacities.pop(cls, None)
                    weights.pop(cls, None)

        return limits

    def before_train(self, task_id: int):
        self.current_task_id = task_id
        self.current_seen_count = self.data_manager.get_seen_class_count(task_id)

        if task_id > 0:
            self.model.incremental_learning(self.current_seen_count * self.ssl_factor)

        train_set = self.data_manager.get_task_dataset(task_id, split="train")
        if (
            self.replay_enable
            and self.replay_mode == "raw"
            and self.replay_strategy == "merged"
            and self.raw_memory_bank.has_data()
        ):
            memory_images, memory_labels = self.raw_memory_bank.export_all()
            train_set = MixedReplayDataset(train_set, memory_images, memory_labels)
        test_set = self.data_manager.get_seen_dataset(task_id, split="test")

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        self.model.to(self.device)
        self.model.train()

    def train_task(self):
        epochs = int(self.cfg["epochs_base"] if self.current_task_id == 0 else self.cfg["epochs_inc"])
        opt = self._build_optimizer(self.model.parameters())
        scheduler = self._build_scheduler(opt, epochs)
        eval_interval = max(1, int(self.cfg["print_freq"]))
        best_score = float("-inf")
        best_state = None
        best_epoch = -1
        best_metrics = None

        for epoch in range(epochs):
            for _, (images, labels, _, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                images, labels_aug = self._augment_ssl(images, labels)

                loss = self._compute_loss(images, labels_aug)
                opt.zero_grad()
                loss.backward()
                opt.step()

            scheduler.step()

            should_eval = (epoch % eval_interval == 0) or (epoch == epochs - 1)
            if should_eval:
                metrics = self.evaluate_seen()
                if epoch % eval_interval == 0:
                    print(
                        f"Task {self.current_task_id} | Epoch {epoch} | "
                        f"OA {metrics['oa']:.4f} | AA {metrics['aa']:.4f} | Kappa {metrics['kappa']:.4f}"
                    )

                if self.select_best_model:
                    score = float(metrics.get(self.selection_metric, metrics["oa"]))
                    if score > best_score:
                        best_score = score
                        best_state = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_metrics = dict(metrics)

                    if (
                        self.early_stop_drop_ratio > 0
                        and best_score > float("-inf")
                        and (epoch + 1) >= self.early_stop_min_epochs
                        and score <= best_score * (1.0 - self.early_stop_drop_ratio)
                    ):
                        print(
                            f"Task {self.current_task_id} | Early stop triggered at epoch {epoch} | "
                            f"{self.selection_metric.upper()} dropped to {score:.4f} from best {best_score:.4f}"
                        )
                        break

        if self.select_best_model and best_state is not None:
            self.model.load_state_dict(best_state)
            print(
                f"Task {self.current_task_id} | Restored best checkpoint from epoch {best_epoch} | "
                f"OA {best_metrics['oa']:.4f} | AA {best_metrics['aa']:.4f} | Kappa {best_metrics['kappa']:.4f}"
            )

        self._save_prototypes_and_memory()
        if (
            self.ft_enable
            and self.current_task_id > 0
            and self.replay_enable
            and self.replay_mode != "raw"
            and self.memory_bank.has_data()
        ):
            self._balanced_finetune()

    def _compute_loss(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, feat = self.model(images)
        logits = logits[:, : self.current_seen_count * self.ssl_factor]
        loss_cls = nn.CrossEntropyLoss()(logits / self.temp, labels)

        if self.old_model is None:
            return loss_cls

        with torch.no_grad():
            _, feat_old = self.old_model(images)
        loss_kd = torch.dist(feat, feat_old, p=2)

        old_class_count = self.data_manager.get_seen_class_count(self.current_task_id - 1)
        loss_total = loss_cls + self.kd_weight * loss_kd

        if old_class_count > 0 and len(self.prototype_dict) > 0:
            proto_aug, proto_aug_label = self.prototype_augmentor.sample(
                prototype_dict=self.prototype_dict,
                diag_scale_dict=self.prototype_diag_scale_dict,
                radius=self.radius,
                batch_size=self.batch_size,
                old_class_count=old_class_count,
                feature_dim=self.feature_dim,
            )

            if len(proto_aug) > 0:
                proto_aug = torch.from_numpy(proto_aug).to(self.device)
                proto_aug_label = torch.from_numpy((proto_aug_label * self.ssl_factor).astype(np.int64)).to(self.device)

                soft_feat_aug = self.model.classify_from_feature(proto_aug)
                soft_feat_aug = soft_feat_aug[:, : self.current_seen_count * self.ssl_factor]
                loss_proto = nn.CrossEntropyLoss()(soft_feat_aug / self.temp, proto_aug_label)
                loss_total = loss_total + self.proto_weight * loss_proto

        if (
            self.replay_enable
            and self.replay_mode == "raw"
            and self.replay_strategy == "loss"
            and self.raw_memory_bank.has_data()
        ):
            mem_images, mem_label = self.raw_memory_bank.sample(self.replay_batch_size, self.device)
            if mem_images.numel() > 0:
                mem_logits, _ = self.model(mem_images)
                mem_logits = mem_logits[:, : self.current_seen_count * self.ssl_factor]
                mem_target = mem_label * self.ssl_factor
                loss_replay = nn.CrossEntropyLoss()(mem_logits / self.temp, mem_target)
                loss_total = loss_total + self.replay_weight * loss_replay

        if self.replay_enable and self.replay_mode != "raw" and self.memory_bank.has_data():
            mem_feat, mem_label = self.memory_bank.sample(self.replay_batch_size, self.device)
            if mem_feat.numel() > 0:
                mem_logits = self.model.classify_from_feature(mem_feat)
                mem_logits = mem_logits[:, : self.current_seen_count * self.ssl_factor]
                mem_target = mem_label * self.ssl_factor
                loss_replay = nn.CrossEntropyLoss()(mem_logits / self.temp, mem_target)
                loss_total = loss_total + self.replay_weight * loss_replay

                if self.align_weight > 0 and len(self.prototype_dict) > 0:
                    centers = []
                    for cls in mem_label.detach().cpu().numpy().tolist():
                        if cls in self.prototype_dict:
                            centers.append(self.prototype_dict[int(cls)])
                        else:
                            centers.append(np.zeros(self.feature_dim, dtype=np.float32))
                    centers = torch.from_numpy(np.asarray(centers, dtype=np.float32)).to(self.device)
                    loss_align = nn.MSELoss()(mem_feat, centers)
                    loss_total = loss_total + self.align_weight * loss_align

        return loss_total

    def _save_prototypes_and_memory(self):
        self.model.eval()

        feature_bank: Dict[int, List[np.ndarray]] = {}
        raw_bank: Dict[int, List[np.ndarray]] = {}
        with torch.no_grad():
            for _, (images, labels, _, _) in enumerate(self.train_loader):
                images_np = images.numpy()
                images = images.to(self.device)
                _, feat = self.model(images)
                feat_np = feat.cpu().numpy()
                labels_np = labels.numpy()
                for i, cls in enumerate(labels_np):
                    feature_bank.setdefault(int(cls), []).append(feat_np[i])
                    raw_bank.setdefault(int(cls), []).append(images_np[i])

        raw_limits = {}
        if self.replay_enable and self.replay_mode == "raw":
            raw_limits = self._compute_raw_memory_limits(self.current_task_id)
            for cls in self.raw_memory_bank.classes():
                if cls in raw_limits:
                    self.raw_memory_bank.trim_class(cls, raw_limits[cls])

        for cls, feats in feature_bank.items():
            feats_arr = np.asarray(feats, dtype=np.float32)
            if self.replay_enable and self.replay_mode != "raw":
                self.memory_bank.add(feats_arr, np.full(feats_arr.shape[0], cls, dtype=np.int64))
            if self.replay_enable and self.replay_mode == "raw":
                raw_arr = np.asarray(raw_bank.get(cls, []), dtype=np.float32)
                if raw_arr.size > 0:
                    cls_limit = raw_limits.get(cls, self.memory_per_class)
                    if self.replay_selection == "herding":
                        selected_indexes = icarl_selection(feats_arr, cls_limit)
                        self.raw_memory_bank.set_class_with_limit(cls, raw_arr[selected_indexes], cls_limit)
                    else:
                        self.raw_memory_bank.set_class_with_limit(cls, raw_arr[-cls_limit:], cls_limit)

        prototype_dict, diag_scale_dict, computed_radius = self.prototype_augmentor.build_statistics(
            feature_bank=feature_bank,
            current_task_id=self.current_task_id,
            feature_dim=self.feature_dim,
        )
        self.prototype_dict = prototype_dict
        self.prototype_diag_scale_dict = diag_scale_dict
        if self.current_task_id == 0 and computed_radius > 0:
            self.radius = computed_radius

        self.model.train()

    def _balanced_finetune(self):
        self.model.train()
        params = list(self.model.parameters())
        opt = self._build_optimizer(params, lr=self.ft_lr)

        for _ in range(self.ft_epochs):
            mem_feat, mem_label = self.memory_bank.sample(self.replay_batch_size, self.device)
            if mem_feat.numel() == 0:
                continue

            logits = self.model.classify_from_feature(mem_feat)
            logits = logits[:, : self.current_seen_count * self.ssl_factor]
            target = mem_label * self.ssl_factor
            loss = nn.CrossEntropyLoss()(logits / self.temp, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

    def after_train(self, exp_name: str):
        if self.bias_correction_enable and self.current_task_id > 0:
            old_class_count = self.data_manager.get_seen_class_count(self.current_task_id - 1)
            self.model.align_weights(old_count=old_class_count * self.ssl_factor)

        ckpt_dir = os.path.join(self.cfg["save_path"], exp_name)
        ensure_dir(ckpt_dir)

        seen_count = self.current_seen_count
        save_path = os.path.join(ckpt_dir, f"{seen_count}_model.pth")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "seen_count": seen_count,
                "prototype_dict": {k: v.tolist() for k, v in self.prototype_dict.items()},
                "prototype_diag_scale_dict": {k: v.tolist() for k, v in self.prototype_diag_scale_dict.items()},
                "radius": self.radius,
                "memory_bank": self.memory_bank.state_dict() if self.replay_enable and self.replay_mode != "raw" else {},
                "raw_memory_bank": self.raw_memory_bank.state_dict() if self.replay_enable and self.replay_mode == "raw" else {},
            },
            save_path,
        )

        self.old_model = copy.deepcopy(self.model)
        self.old_model.to(self.device)
        self.old_model.eval()

    def predict(self, dataloader: DataLoader, seen_count: int) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for images, labels, _, _ in dataloader:
                images = images.to(self.device)
                logits, _ = self.model(images)
                logits = self._primary_logits(logits, seen_count)
                pred = torch.argmax(logits, dim=1)
                ys.append(labels.numpy())
                ps.append(pred.cpu().numpy())
        self.model.train()
        return np.concatenate(ys), np.concatenate(ps)

    def predict_with_positions(
        self, dataloader: DataLoader, seen_count: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        ys = []
        ps = []
        rows = []
        cols = []
        with torch.no_grad():
            for images, labels, r, c in dataloader:
                images = images.to(self.device)
                logits, _ = self.model(images)
                logits = self._primary_logits(logits, seen_count)
                pred = torch.argmax(logits, dim=1)
                ys.append(labels.numpy())
                ps.append(pred.cpu().numpy())
                rows.append(r.numpy())
                cols.append(c.numpy())
        self.model.train()
        return np.concatenate(ys), np.concatenate(ps), np.concatenate(rows), np.concatenate(cols)

    def evaluate_seen(self):
        y_true, y_pred = self.predict(self.test_loader, self.current_seen_count)
        return evaluate_all(y_true, y_pred, num_classes=self.current_seen_count)

    def save_task_visualization(self, exp_dir: str, task_id: int):
        y_true, y_pred, rows, cols = self.predict_with_positions(self.test_loader, self.current_seen_count)
        _ = y_true  # y_true currently not used by renderer, kept for optional future checks.
        vis_dir = os.path.join(exp_dir, "task_visualizations")
        save_task_test_aligned_comparison_figure(
            out_dir=vis_dir,
            task_id=task_id,
            gt=self.data_manager.gt,
            seen_classes=self.data_manager.get_seen_classes(task_id),
            rows=rows,
            cols=cols,
            preds=y_pred,
            dataset_name=self.data_manager.dataset_name,
            class_names=self.data_manager.class_order_names,
        )
