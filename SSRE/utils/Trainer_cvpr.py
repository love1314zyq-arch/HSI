import copy
import builtins

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import logging
import time

from utils.model import prepare_model
from utils.data import datatypes
from utils.data import loadertypes
from utils.model_para import filter_para
from utils.loss import MyLosses
from utils.evaluate import Evaluator
from utils.lr import LR_Scheduler
from torch.utils.data import DataLoader, ConcatDataset
from utils.data.my_dataset import SingleClass
from utils.model.examplar import ExemplarDataset
from utils.model import backbones
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torchvision import transforms
from benchmarks.visualizations import save_task_prediction_artifact


# tiny
class Trainer(object):
    def __init__(self, args):
        self.args = args
        print(f"[ssre] trainer init start | dataset={args.dataset_name} | backbone={args.backbone_name}")

        # Define Saver
        # self.saver = Saver(args)
        # self.saver.save_experiment_config()

        # Define dataloader for train/test
        prepare_data = datatypes[args.dataset_type]
        self.dataset = prepare_data(args.dataset_name, args)
        self.dataloader = loadertypes[args.loader_type]
        print("[ssre] dataset objects prepared")
        if hasattr(args, "task_sizes") and args.task_sizes:
            start = 0
            self.label_per_task = []
            for size in args.task_sizes:
                end = start + int(size)
                self.label_per_task.append(list(range(start, end)))
                start = end
        else:
            self.label_per_task = [list(np.array(range(args.base_class)))] + [list(np.array(range(args.way)) +
                                                                                   args.way * task_id + args.base_class)
                                                                              for task_id in range(args.tasks)]

        # Define model and optimizer
        model = prepare_model(args)
        print("[ssre] model prepared")
        optim_para = filter_para(model, args, args.lr)
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(optim_para, weight_decay=args.weight_decay, nesterov=args.nesterov)
        else:
            optimizer = torch.optim.Adam(optim_para, weight_decay=args.weight_decay)

        self.criterion = MyLosses(weight=None, args=self.args).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(args.all_class)

        # Using cuda
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
        self.model, self.optimizer = model, optimizer
        self.old_model = None

        # Resuming checkpoint
        self.best_pred = 0.0
        self.test_out = None
        self.ii = 0

        # history of prediction
        self.acc_history = []
        self.forget_history = []
        self.stage_metrics = []
        self.last_metrics = None
        print("[ssre] trainer init finished")

    def _current_task_labels(self, session):
        return self.label_per_task[session]

    def _seen_labels(self, session):
        labels = []
        for task in range(session + 1):
            labels.extend(self.label_per_task[task])
        return labels

    def _compute_hsi_metrics(self, confusion_matrix, session_labels):
        cm = confusion_matrix[np.ix_(session_labels, session_labels)]
        total = np.sum(cm)
        if total == 0:
            return np.zeros(len(session_labels), dtype=np.float32), 0.0, 0.0, 0.0
        acc_class = np.divide(np.diag(cm), np.sum(cm, axis=1), out=np.zeros(len(session_labels), dtype=np.float32), where=np.sum(cm, axis=1) != 0)
        oa = float(np.sum(np.diag(cm)) / total)
        aa = float(np.mean(acc_class)) if len(acc_class) > 0 else 0.0
        pe = float(np.sum(np.sum(cm, axis=1) * np.sum(cm, axis=0)) / (total ** 2))
        denom = 1.0 - pe
        kappa = 0.0 if abs(denom) < 1e-12 else float((oa - pe) / denom)
        return acc_class, oa, aa, kappa

    def training(self, session):
        classes = self._current_task_labels(session)
        session_class = len(self._seen_labels(session))
        session_class_last = 0 if session == 0 else len(self._seen_labels(session - 1))
        train_dataset = self.dataset[0]
        train_dataset.getTrainData(classes)
        train_loader = self.dataloader(train_dataset, self.args)
        epochs = self.args.base_epochs if session == 0 else self.args.new_epochs
        lr = self.args.lr if session == 0 else self.args.lr_new
        seen_labels = self._seen_labels(session)

        print(
            f"[ssre] session {session + 1}/{self.args.session} | "
            f"current_classes={classes} | seen_classes={seen_labels} | "
            f"train_samples={len(train_dataset)} | epochs={epochs} | lr={lr}"
        )

        if session == 0 and self.args.val:
            self.model.eval()
            para = torch.load(self.args.model_path)
            para_dict = para['state_dict']
            para_dict_re = self.structure_reorganization(para_dict)
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in para_dict_re.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            if session > 0:
                self.model.eval()
                if self.args.backbone_name != 'hybrid_hsi_lite':
                    backbone = backbones[self.args.backbone_name](
                        mode='parallel_adapters',
                        in_channels=getattr(self.args, "input_channels", getattr(self.args, "pca_dim", 3)),
                    ).cuda()
                    model_dict = backbone.state_dict()
                    para_dict = self.model.backbone.state_dict()
                    state_dict = {k: v for k, v in para_dict.items() if k in model_dict.keys()}
                    model_dict.update(state_dict)
                    backbone.load_state_dict(model_dict)
                    self.model.backbone = backbone
                    self.model.fix_backbone_adapter()

            self.model.classifier.Incremental_learning(session_class)
            self.model = self.model.cuda()
            self.model.train()

            optim_para = filter_para(self.model, self.args, lr)
            if self.args.optim == 'sgd':
                self.optimizer = torch.optim.SGD(optim_para, weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
            else:
                self.optimizer = torch.optim.Adam(optim_para, weight_decay=self.args.weight_decay)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=45, gamma=0.1)

            for epoch in range(epochs):
                tbar = tqdm(train_loader)
                train_loss = 0.0
                # for i, sample in enumerate(train_loader):
                for i, sample in enumerate(tbar):
                    query_image, query_target = sample[1], sample[2]
                    if self.args.cuda:
                        query_image, query_target = query_image.cuda(), query_target.cuda()

                    self.optimizer.zero_grad()
                    loss = self._compute_loss(query_image, query_target, session_class_last)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    tbar.set_description(
                        "Session %d/%d | Epoch %d/%d | Train loss %.4f"
                        % (session + 1, self.args.session, epoch + 1, epochs, train_loss / (i + 1))
                    )
                if not self.args.no_lr_scheduler:
                    self.scheduler.step()

                if (epoch+1) % 10 == 0:
                    self.validation(session)
                    metrics = self.last_metrics or {}
                    print(
                        f"[ssre] session {session + 1}/{self.args.session} | "
                        f"epoch {epoch + 1}/{epochs} | "
                        f"seen_OA={metrics.get('oa', 0.0):.5f} | "
                        f"seen_AA={metrics.get('aa', 0.0):.5f} | "
                        f"seen_Kappa={metrics.get('kappa', 0.0):.5f}"
                    )
                    self.model.train()
        self.protoSave(self.model, train_loader, session)
        self.afterTrain(session)
        print(f"[ssre] session {session + 1}/{self.args.session} training finished")

    def validation(self, session, print=False):
        self.model.eval()
        self.evaluator.reset()
        forget_history = []

        session_labels = self._seen_labels(session)

        test_dataset = self.dataset[1]
        test_dataset.getTestData_up2now(session_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size_test, shuffle=False,
                                 num_workers=getattr(self.args, "num_workers", 0))
        all_targets = []
        all_preds = []

        for i, sample in enumerate(test_loader):
            image, target = sample[1], sample[2]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            all_targets.append(target)
            all_preds.append(pred)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Test
        confusion_matrix = self.evaluator.confusion_matrix.copy()
        Acc_class, OA, AA, Kappa = self._compute_hsi_metrics(confusion_matrix, session_labels)
        Acc = OA
        logging.info('Validation:')
        logging.info('[Session: %d, numImages: %5d]' % (session, i * self.args.batch_size + image.data.shape[0]))
        logging.info("Acc_class:{}, Acc:{} OA:{} AA:{} Kappa:{} \n".format(Acc_class, Acc, OA, AA, Kappa))
        self.last_metrics = {
            "task_id": session,
            "seen_classes": len(session_labels),
            "oa": float(OA),
            "aa": float(AA),
            "kappa": float(Kappa),
            "acc_class": np.asarray(Acc_class).tolist(),
        }
        if print:
            flat_targets = np.concatenate(all_targets, axis=0) if all_targets else np.asarray([], dtype=np.int64)
            flat_preds = np.concatenate(all_preds, axis=0) if all_preds else np.asarray([], dtype=np.int64)
            coords = np.asarray([test_dataset.coords[int(idx)] for idx in test_dataset.TestIndexes], dtype=np.int64)
            save_task_prediction_artifact(
                self.args.output_root,
                session,
                session_labels,
                coords[:, 0] if len(coords) else np.asarray([], dtype=np.int64),
                coords[:, 1] if len(coords) else np.asarray([], dtype=np.int64),
                flat_preds,
                flat_targets,
            )
            self.acc_history.append(Acc)
            for j in range(session+1):
                task_labels = self.label_per_task[j]
                task_group_acc = np.mean([Acc_class[session_labels.index(lbl)] for lbl in task_labels if lbl in session_labels])
                forget_history.append(np.around(task_group_acc, decimals=4))
            self.forget_history.append(forget_history)
            self.stage_metrics.append(self.last_metrics)
            builtins.print(
                f"[ssre] eval session {session + 1}/{self.args.session} | "
                f"seen_classes={len(session_labels)} | "
                f"OA={OA:.4f} | AA={AA:.4f} | Kappa={Kappa:.4f}"
            )
            builtins.print(f"[ssre] per-task group accuracy: {forget_history}")
            if session == (self.args.session-1):
                forget_avg = 0
                for ff in range(self.args.session-1):
                    forget_avg = forget_avg + self.forget_history[ff][ff] - self.forget_history[self.args.session-1][ff]
                forget_avg /= (self.args.session-1)
                logging.info("Avg_for:{} \n".format(forget_avg))
                logging.info("Avg_acc:{} \n".format(np.mean(self.acc_history)))
                logging.info("Acc:{} \n".format(list(self.acc_history)))
                builtins.print(f"[ssre] average_forgetting={forget_avg:.4f}")
                builtins.print(f"[ssre] accuracy_history={list(self.acc_history)}")

        return Acc

    def afterTrain(self, current_task):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(self.args.dir_name,
                                'model_' + self.args.dataset_name + '_' + str(self.args.way) + '_' + str(
                                    current_task) + '.pth'))
        self.old_model = self.model.copy()
        self.old_model.cuda()
        self.old_model.eval()

        if current_task > 0 and self.args.backbone_name != 'hybrid_hsi_lite':
            model_dict = self.model.state_dict()
            for k, v in model_dict.items():
                if 'adapter' in k:
                    k_conv3 = k.replace('adapter', 'conv')
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
            self.model.load_state_dict(model_dict)

    def _compute_loss(self, imgs, target, old_class=0):
        if self.old_model is None:
            output = self.model(imgs)
            loss_cls = nn.CrossEntropyLoss()(output / 0.1, target)
            return loss_cls
        else:
            feature = self.model.feature_extractor(imgs)
            feature_old = self.old_model.feature_extractor(imgs)

            proto = torch.from_numpy(np.array(self.prototype)).t().cuda()
            proto_nor = torch.nn.functional.normalize(proto, p=2, dim=0, eps=1e-12)
            feature_nor = torch.nn.functional.normalize(feature, p=2, dim=-1, eps=1e-12)
            cos_dist = feature_nor @ proto_nor
            cos_dist = torch.max(cos_dist, dim=-1).values
            cos_dist2 = 1 - cos_dist
            output = self.model(imgs)
            loss_cls = nn.CrossEntropyLoss(reduce=False)(output / 0.1, target)
            loss_cls = torch.mean(loss_cls*cos_dist2, dim=0)

            loss_kd = torch.norm(feature-feature_old, p=2, dim=1)
            loss_kd = torch.sum(loss_kd*cos_dist, dim=0)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            sample_count = min(self.args.batch_size, max(old_class, 1))
            for _ in range(sample_count):
                np.random.shuffle(index)
                temp = self.prototype[index[0]]
                proto_aug.append(temp)
                proto_aug_label.append(self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().cuda()
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).cuda()
            soft_feat_aug = self.model.classifier(proto_aug, 0)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / 0.1, proto_aug_label)
            return loss_cls + 10 * loss_protoAug + loss_kd

    def structure_reorganization(self, para_dict):
        para_dict_re = copy.deepcopy(para_dict)
        for k, v in para_dict.items():
            if 'bn.weight' in k or 'bn1.weight' in k or 'downsample.1.weight' in k:
                if 'bn.weight' in k:
                    k_conv3 = k.replace('bn', 'conv')
                elif 'bn1.weight' in k:
                    k_conv3 = k.replace('bn1', 'conv1')
                elif 'downsample.1.weight' in k:
                    k_conv3 = k.replace('1', '0')
                k_conv3_bias = k_conv3.replace('weight', 'bias')
                k_bn_bias = k.replace('weight', 'bias')
                k_bn_mean = k.replace('weight', 'running_mean')
                k_bn_var = k.replace('weight', 'running_var')

                gamma = para_dict[k]
                beta = para_dict[k_bn_bias]
                running_mean = para_dict[k_bn_mean]
                running_var = para_dict[k_bn_var]
                eps = 1e-5
                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                para_dict_re[k_conv3] *= t
                para_dict_re[k_conv3_bias] = beta - running_mean * gamma / std
        return para_dict_re

    def protoSave(self, model, loader, current_task):
        # if current_task > 0:
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (_, images, target) in enumerate(loader):
                feature = model.feature_extractor(images.cuda())
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        if not labels or not features:
            return
        labels = np.concatenate(labels, axis=0)
        features = np.concatenate(features, axis=0)
        labels_set = np.unique(labels)

        prototype = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))

        if current_task == 0:
            self.prototype = prototype
            self.class_label = class_label
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)

    def average_forgetting(self):
        if len(self.forget_history) <= 1:
            return 0.0
        final_idx = len(self.forget_history) - 1
        forget_avg = 0.0
        for ff in range(final_idx):
            forget_avg += self.forget_history[ff][ff] - self.forget_history[final_idx][ff]
        return float(forget_avg / final_idx)
