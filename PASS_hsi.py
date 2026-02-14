import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from feature_memory_hsi import FeatureMemoryBank
from metrics_hsi import evaluate_all
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
        self.use_rotation_ssl = bool(pass_cfg.get("use_rotation_ssl", True))

        self.train_cfg = cfg.get("train", {})
        self.optimizer_name = str(self.train_cfg.get("optimizer", "adam")).lower()
        self.scheduler_name = str(self.train_cfg.get("scheduler", "step")).lower()

        replay_cfg = cfg.get("replay", {})
        self.replay_enable = bool(replay_cfg.get("enable", False))
        self.replay_weight = float(replay_cfg.get("lambda_replay", 1.0))
        self.align_weight = float(replay_cfg.get("lambda_align", 0.0))
        self.memory_per_class = int(replay_cfg.get("memory_per_class", 40))
        self.replay_batch_size = int(replay_cfg.get("batch_size", self.batch_size))
        self.memory_bank = FeatureMemoryBank(memory_per_class=self.memory_per_class)

        bias_cfg = cfg.get("bias_correction", {})
        self.bias_correction_enable = bool(bias_cfg.get("enable", False))

        ft_cfg = cfg.get("balanced_finetune", {})
        self.ft_enable = bool(ft_cfg.get("enable", False))
        self.ft_epochs = int(ft_cfg.get("epochs", 0))
        self.ft_lr = float(ft_cfg.get("learning_rate", self.learning_rate * 0.5))

        self.old_model = None
        self.prototype_dict: Dict[int, np.ndarray] = {}
        self.radius = 0.0
        self.feature_dim = model.feature_dim

        self.train_loader = None
        self.test_loader = None
        self.current_task_id = 0
        self.current_seen_count = 0

    def _build_optimizer(self, params, lr: float = None):
        lr = self.learning_rate if lr is None else lr
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        return torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)

    def _build_scheduler(self, optimizer, epochs: int):
        if self.scheduler_name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        return StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)

    def before_train(self, task_id: int):
        self.current_task_id = task_id
        self.current_seen_count = self.data_manager.get_seen_class_count(task_id)

        if task_id > 0:
            self.model.incremental_learning(self.current_seen_count * 4)

        train_set = self.data_manager.get_task_dataset(task_id, split="train")
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

        for epoch in range(epochs):
            for _, (images, labels, _, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.use_rotation_ssl:
                    images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], dim=1)
                    b, r, c, h, w = images.shape
                    images = images.view(b * r, c, h, w)
                    labels_aug = torch.stack([labels * 4 + k for k in range(4)], dim=1).view(-1)
                else:
                    labels_aug = labels

                loss = self._compute_loss(images, labels_aug)
                opt.zero_grad()
                loss.backward()
                opt.step()

            scheduler.step()

            if epoch % int(self.cfg["print_freq"]) == 0:
                metrics = self.evaluate_seen()
                print(
                    f"Task {self.current_task_id} | Epoch {epoch} | "
                    f"OA {metrics['oa']:.4f} | AA {metrics['aa']:.4f} | Kappa {metrics['kappa']:.4f}"
                )

        self._save_prototypes_and_memory()
        if self.ft_enable and self.current_task_id > 0 and self.replay_enable and self.memory_bank.has_data():
            self._balanced_finetune()

    def _compute_loss(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, feat = self.model(images)
        logits = logits[:, : self.current_seen_count * 4]
        loss_cls = nn.CrossEntropyLoss()(logits / self.temp, labels)

        if self.old_model is None:
            return loss_cls

        with torch.no_grad():
            _, feat_old = self.old_model(images)
        loss_kd = torch.dist(feat, feat_old, p=2)

        old_class_count = self.data_manager.get_seen_class_count(self.current_task_id - 1)
        loss_total = loss_cls + self.kd_weight * loss_kd

        if old_class_count > 0 and len(self.prototype_dict) > 0:
            proto_aug = []
            proto_aug_label = []
            for _ in range(self.batch_size):
                cls = np.random.randint(0, old_class_count)
                if cls not in self.prototype_dict:
                    continue
                noise = np.random.normal(0, 1, self.feature_dim).astype(np.float32) * float(self.radius)
                proto = self.prototype_dict[cls] + noise
                proto_aug.append(proto)
                proto_aug_label.append(4 * cls)

            if len(proto_aug) > 0:
                proto_aug = torch.from_numpy(np.asarray(proto_aug, dtype=np.float32)).to(self.device)
                proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label, dtype=np.int64)).to(self.device)

                soft_feat_aug = self.model.classify_from_feature(proto_aug)
                soft_feat_aug = soft_feat_aug[:, : self.current_seen_count * 4]
                loss_proto = nn.CrossEntropyLoss()(soft_feat_aug / self.temp, proto_aug_label)
                loss_total = loss_total + self.proto_weight * loss_proto

        if self.replay_enable and self.memory_bank.has_data():
            mem_feat, mem_label = self.memory_bank.sample(self.replay_batch_size, self.device)
            if mem_feat.numel() > 0:
                mem_logits = self.model.classify_from_feature(mem_feat)
                mem_logits = mem_logits[:, : self.current_seen_count * 4]
                mem_target = mem_label * 4
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
        with torch.no_grad():
            for _, (images, labels, _, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                _, feat = self.model(images)
                feat_np = feat.cpu().numpy()
                labels_np = labels.numpy()
                for i, cls in enumerate(labels_np):
                    feature_bank.setdefault(int(cls), []).append(feat_np[i])

        radius_list = []
        for cls, feats in feature_bank.items():
            feats_arr = np.asarray(feats, dtype=np.float32)
            self.prototype_dict[cls] = np.mean(feats_arr, axis=0)
            if self.replay_enable:
                self.memory_bank.add(feats_arr, np.full(feats_arr.shape[0], cls, dtype=np.int64))
            if self.current_task_id == 0 and feats_arr.shape[0] > 1:
                cov = np.cov(feats_arr.T)
                radius_list.append(np.trace(cov) / feats_arr.shape[1])

        if self.current_task_id == 0 and len(radius_list) > 0:
            self.radius = float(np.sqrt(np.mean(radius_list)))

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
            logits = logits[:, : self.current_seen_count * 4]
            target = mem_label * 4
            loss = nn.CrossEntropyLoss()(logits / self.temp, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

    def after_train(self, exp_name: str):
        if self.bias_correction_enable and self.current_task_id > 0:
            old_class_count = self.data_manager.get_seen_class_count(self.current_task_id - 1)
            self.model.align_weights(old_count=old_class_count * 4)

        ckpt_dir = os.path.join(self.cfg["save_path"], exp_name)
        ensure_dir(ckpt_dir)

        seen_count = self.current_seen_count
        save_path = os.path.join(ckpt_dir, f"{seen_count}_model.pth")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "seen_count": seen_count,
                "prototype_dict": {k: v.tolist() for k, v in self.prototype_dict.items()},
                "radius": self.radius,
                "memory_bank": self.memory_bank.state_dict() if self.replay_enable else {},
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
                logits = logits[:, : seen_count * 4]
                logits = logits[:, ::4]
                pred = torch.argmax(logits, dim=1)
                ys.append(labels.numpy())
                ps.append(pred.cpu().numpy())
        self.model.train()
        return np.concatenate(ys), np.concatenate(ps)

    def evaluate_seen(self):
        y_true, y_pred = self.predict(self.test_loader, self.current_seen_count)
        return evaluate_all(y_true, y_pred, num_classes=self.current_seen_count)
