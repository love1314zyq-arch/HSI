import copy
import logging
import math
import collections
import numpy as np
import torch
from torch.nn import functional as F
from hylearn.lib.modules.batch import dict_set
from hylearn.lib import data, herding, utils

import logging
import os
import pickle
import math
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from collections import OrderedDict

import hylearn.lib.modules.basenet as quantinet
import hylearn.lib.modules.basesample as  fullnet

from hylearn.lib.modules import evaluation, losses, increment_modules
from hylearn.models.base import IncrementalLearner
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


logger = logging.getLogger(__name__)

EPSILON = 1e-8
TRAIN = torch.ones(1)
TEST = torch.zeros(1)
class LSCNet(IncrementalLearner):
    """Pooled Output Distillation Network.

    # Reference:
        * Small Task Incremental Learning
          Douillard et al. 2020
    """

    def __init__(self, args,dpblock):
        self._disable_progressbar = args.get("no_progressbar", False)
        self._device = args["device"][0]

        # Optimization:
        self._batch_size = args["batch_size"]
        self._lr = args["learning_rate"]
        self._weight_decay = args["weight_decay"]
        self._first_epochs = args["first_epochs"]
        self._expand_epochs = args["expand_epochs"]
        self._compress_epochs = args["compress_epochs"]
        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args.get("fixed_memory", True)
        # self._herding_selection = args.get("herding_selection", {"type": "random"})
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")
        self.inplanes = 32
        self._threshold = args["threshold"]

        self._evaluation_type = args.get("eval_type", "icarl")
        self.dpblock = dpblock
        self._network = quantinet.BasicNet(
            args["convnet"],
            self.dpblock[0],
            self.dpblock[1],
            self.dpblock[2],
            self.dpblock[3],
            self.dpblock[4],
            args["channels"],
            device=self._device,
        )
        self._eval_in_first_stage = args.get("eval_in_first_stage", True)
        self._bn_for_att = args.get("bn_for_attention", True)
        self.weight_list = dpblock[5]
        self.orign_list = dpblock[6]
        self.inter_list = dpblock[7]

        self._examplars = {}
        self._means = None
        self._old_model = None
        self.l2 = 0.0001
        self._herding_indexes = []
        self._convnet = args["convnet"]
        self._initial_channel = args["channels"]
        self._weight_generation = args.get("weight_generation")

        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}
        self.best_epoch, self.best_acc = -1, -1.
        self.best_backbone = None
        self._record_classes = []
        self.MODE = torch.empty(1)
        self._saved_gate_old = None
        self._saved_gate_new = None
        self._saved_new_net = None
        self._inherit_new_net = args['inherit_new_net']

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        temp_size = self._memory_size
        rem = []
        rare_size = []
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        else:
            class_state = np.concatenate(self._class_state)
            logger.info("Numbers of samples from each class {}.".format(class_state))
            all_classes = class_state.sum()
            per_classes = class_state/all_classes
            memeory_per_class =[math.floor(j) for j in per_classes*temp_size]
            logger.info("Now {} examplars saved respectively for each class.".format(memeory_per_class))
            return memeory_per_class

    def _before_task(self, train_loader, val_loader,args):
        # self._gen_weights()
        if self._task == 0:
            stu_epoch = self._first_epochs
            lr = self._lr

        else:
            stu_epoch = self._compress_epochs
            lr = self._lr
        self._n_classes += self._task_size
        self._record_classes.append(self._n_classes)
        if  self._task == 0:
            self._network.add_classes(self._task_size)
            self._teacher = fullnet.BasicNet(args["convnet"], args["channels"], device=self._device)
            self._teacher.add_classes(self._task_size)
            self.optimizer1 = torch.optim.Adam(self._network.parameters(), lr=lr, weight_decay=self._weight_decay)
            self.lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, stu_epoch, eta_min=0, last_epoch=-1)
            self.optimizer6 = torch.optim.Adam(self._teacher.parameters(), lr=self._lr, weight_decay=self._weight_decay)
            self.lr_scheduler6 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer6, self._first_epochs, eta_min=0,
                                                                            last_epoch=-1)

        self.optimizer2 = torch.optim.Adam(self.dpblock[0], lr=lr, weight_decay=self._weight_decay)
        self.optimizer3 = torch.optim.Adam(self.dpblock[1], lr=lr, weight_decay=self._weight_decay)
        self.optimizer4 = torch.optim.Adam(self.dpblock[2], lr=lr, weight_decay=self._weight_decay)
        self.optimizer5 = torch.optim.Adam(self.dpblock[3], lr=lr, weight_decay=self._weight_decay)

        self.lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, stu_epoch, eta_min=0, last_epoch=-1)
        self. lr_scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer3, stu_epoch, eta_min=0, last_epoch=-1)
        self.lr_scheduler4 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer4, stu_epoch, eta_min=0, last_epoch=-1)
        self.lr_scheduler5 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer5, stu_epoch, eta_min=0, last_epoch=-1)

        if self._task!=0:
            self.gate_for_old = increment_modules.gate_for_old(self.inplanes,self._bn_for_att).to(self._device)

            self.gate_for_new = increment_modules.gate_for_new(self.inplanes,self._bn_for_att).to(self._device)

            self._net_for_new = fullnet.BasicNet(args["convnet"], args["channels"], device=self._device)
            if self._inherit_new_net:

                self._net_for_new.shallow_net.load_state_dict(self._old_model.shallow_net.state_dict())

            self._net_for_new.add_classes(self._n_classes)
            self._net_for_new.freeze(True)
            self.optimizer7 = torch.optim.Adam(self.gate_for_old.parameters(), lr=self._lr, weight_decay=self._weight_decay)
            self.lr_scheduler7 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer7, self._expand_epochs, eta_min=0,last_epoch=-1)
            self.optimizer8 = torch.optim.Adam(self.gate_for_new.parameters(), lr=self._lr, weight_decay=self._weight_decay)
            self.lr_scheduler8 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer8, self._expand_epochs, eta_min=0,last_epoch=-1)
            self.optimizer9 = torch.optim.Adam(self._net_for_new.parameters(), lr=self._lr, weight_decay=self._weight_decay)
            self.lr_scheduler9 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer9, self._expand_epochs, eta_min=0,last_epoch=-1)

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        if self._task == 0:
            self._distill_step(train_loader, val_loader, 0, self._first_epochs, finetune=False)
        else:
            self._training_step(train_loader, val_loader, 0, self._expand_epochs,self._compress_epochs, finetune=False)

    def _distill_step(
            self, train_loader, val_loader, initial_epoch, nb_epochs, finetune=False):
        # self.MODE = TRAIN
        training_network = self._network
        teacher_network = self._teacher
        threshold = self._threshold
        training_network.train()
        teacher_network.train()
        
        for epoch in range(initial_epoch, nb_epochs):
            self.MODE = TRAIN
            self._metrics = collections.defaultdict(float)
            
            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs1, inputs2, targets = input_dict["inputs1"], input_dict["inputs2"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                self.optimizer3.zero_grad()
                self.optimizer4.zero_grad()
                self.optimizer5.zero_grad()
                self.optimizer6.zero_grad()
                l2_alpha = 0.0
                for qq in range(len(self.dpblock[3])):
                    if qq > 1:
                        l2_alpha += torch.pow(self.dpblock[3][qq], 2)
                for p in range(len(self.dpblock[0])):
                    self.dpblock[0][p].retain_grad()
                for n in range(len( self.dpblock[1])):
                    self.dpblock[1][n].retain_grad()
                    self.dpblock[2][n].retain_grad()
                for q in range(len(self.dpblock[3])):
                    self.dpblock[3][q].retain_grad()
                loss = self._distill_loss(
                    training_network,
                    teacher_network,
                    l2_alpha,
                    inputs1,
                    inputs2,
                    targets,
                    memory_flags,
                    finetune,
                    epoch,
                    threshold
                )
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()
                self.optimizer3.step()
                self.optimizer4.step()
                self.optimizer5.step()
                self.optimizer6.step()


                self._print_metrics(prog_bar, epoch, nb_epochs, i)
            if self.lr_scheduler1:
                self.lr_scheduler1.step()
                self.lr_scheduler2.step()
                self.lr_scheduler3.step()
                self.lr_scheduler4.step()
                self.lr_scheduler5.step()
                self.lr_scheduler6.step()

            if self._eval_in_first_stage and self._task == 0:
                self._network.eval()
                OAC = self._eval_first_task(val_loader)
                logger.info("Val accuracy: {}".format(OAC))
                if OAC >= self.best_acc:
                    self.best_epoch = epoch
                    self.best_acc = OAC
                    self.best_backbone = self._network.copy()
                    self.best_dpblock = copy.deepcopy(self.dpblock)
                self._network.train()

        if self._eval_in_first_stage and self._task == 0:
            logger.info("Best accuracy reached at epoch {} with {}%.".format(self.best_epoch, self.best_acc))

    def _training_step(self, train_loader, val_loader, initial_epoch, ex_epochs,cm_peochs,finetune=False):

        record_p_for_old = []
        record_q_for_old = []
        record_p_for_new = []
        record_q_for_new = []
        old_network = self._old_model
        old_block = self.old_dpblock
        new_network = self._net_for_new
        old_gate = self.gate_for_old
        new_gate = self.gate_for_new
        training_network = self._network
        block = self.dpblock
        ##expansion
        logger.info("Begin expansion procedure of task_{}.".format(self._task))
        new_network.train()
        old_gate.train()
        new_gate.train()
        for epoch in range(initial_epoch, ex_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (ex_epochs - initial_epoch)

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                self.MODE = TRAIN
                inputs1, inputs2, targets = input_dict["inputs1"], input_dict["inputs2"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                self.optimizer7.zero_grad()
                self.optimizer8.zero_grad()
                self.optimizer9.zero_grad()
                # self.optimizer10.zero_grad()
                # self.optimizer11.zero_grad()
                # self.optimizer12.zero_grad()
                # self.optimizer13.zero_grad()
                # self.optimizer14.zero_grad()
                loss = self._expansion_loss(
                    old_network,
                    new_network,
                    old_gate,
                    new_gate,
                    old_block,
                    inputs1,
                    inputs2,
                    targets,
                    memory_flags,
                    finetune,
                    epoch
                )
                loss.backward()
                self.optimizer7.step()
                self.optimizer8.step()
                self.optimizer9.step()

                self._print_metrics(prog_bar, epoch, ex_epochs, i)
            if self.lr_scheduler7:
                self.lr_scheduler7.step()
                self.lr_scheduler8.step()
                self.lr_scheduler9.step()

        #compression
        logger.info("Begin compression procedure of task_{}.".format(self._task))
        self._network.add_classes(self._n_classes)
        self.optimizer1 = torch.optim.Adam(self._network.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        self.lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, self._compress_epochs, eta_min=0, last_epoch=-1)
        self._network.freeze(True)
        training_network.train()
        for epoch in range(initial_epoch, cm_peochs):
            self._metrics = collections.defaultdict(float)
            self.MODE = TRAIN
            self._epoch_percent = epoch / (cm_peochs - initial_epoch)

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs1, inputs2, targets = input_dict["inputs1"], input_dict["inputs2"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                self.optimizer3.zero_grad()
                self.optimizer4.zero_grad()
                self.optimizer5.zero_grad()
                l2_alpha = 0.0
                for qq in range(len(self.dpblock[3])):
                    if qq > 1:
                        l2_alpha += torch.pow(self.dpblock[3][qq], 2)
                for p in range(len(self.dpblock[0])):
                    self.dpblock[0][p].retain_grad()
                for n in range(len( self.dpblock[1])):
                    self.dpblock[1][n].retain_grad()
                    self.dpblock[2][n].retain_grad()
                for q in range(len(self.dpblock[3])):
                    self.dpblock[3][q].retain_grad()
                loss = self._compression_loss(
                    old_network,
                    new_network,
                    old_gate,
                    new_gate,
                    old_block,
                    training_network,
                    block,
                    l2_alpha,
                    inputs1,
                    inputs2,
                    targets,
                    memory_flags,
                    finetune,
                    epoch,
                    record_p_for_old,
                    record_q_for_old,
                    record_p_for_new,
                    record_q_for_new
                )
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()
                self.optimizer3.step()
                self.optimizer4.step()
                self.optimizer5.step()

                self._print_metrics(prog_bar, epoch, cm_peochs, i)
            if self.lr_scheduler1:
                self.lr_scheduler1.step()
                self.lr_scheduler2.step()
                self.lr_scheduler3.step()
                self.lr_scheduler4.step()
                self.lr_scheduler5.step()


    def _expansion_loss(
            self,
            old_network,
            new_network,
            old_gate,
            new_gate,
            old_block,
            inputs1,
            inputs2,
            targets,
            memory_flags,
            finetune,
            epoch,
    ):
        loss = 0
        inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)
        old_network.eval()
        with torch.no_grad():
            self.MODE = TEST
            old_outputs1 = old_network(inputs1,self.MODE,old_block[5],old_block[6],old_block[7])
            old_outputs2 = old_network(inputs2,self.MODE,old_block[5],old_block[6],old_block[7])
            old_features1 = old_outputs1["raw_features"]
            old_features2 = old_outputs2["raw_features"]
            old_im_features1 = old_outputs1["intermdeiate"]
            old_im_features2 = old_outputs2["intermdeiate"]
            old_logits1 = old_outputs1["logits"]
            old_logits2 = old_outputs2["logits"]
        old_values1 = old_gate(old_im_features1.detach())*old_features1.detach()
        old_values2 = old_gate(old_im_features2.detach())*old_features2.detach()
        outputs1 = new_network(inputs1,new_gate,old_values1)
        outputs2 = new_network(inputs2,new_gate,old_values2)
        features1, logits1, proj1, pred1 = outputs1["raw_features"], outputs1["logits"],outputs1["projection"],outputs1["prediction"]
        features2, logits2, proj2, pred2 = outputs2["raw_features"], outputs2["logits"],outputs2["projection"],outputs2["prediction"]
        # if not finetune:
        ncaloss1 = losses.nca(
            logits1,
            targets,
        )
        ncaloss2 = losses.nca(
            logits2,
            targets,
        )

        ncaloss = (ncaloss1+ncaloss2)/2
        loss+=ncaloss
        self._metrics["ncaloss"] += ncaloss.item()
        #try scale
        encoder_loss = losses.compute_contra_loss(proj1.detach(),proj2.detach(), pred1, pred2,targets,epoch)
        self._metrics["enc"] += encoder_loss.item()
        loss += encoder_loss

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss

    def _compression_loss(
            self,
            old_network,
            new_network,
            old_gate,
            new_gate,
            old_block,
            training_network,
            block,
            l2_alpha,
            inputs1,
            inputs2,
            targets,
            memory_flags,
            finetune,
            epoch,
            record_p_for_old,
            record_q_for_old,
            record_p_for_new,
            record_q_for_new
    ):
        loss = 0
        inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)
        old_network.eval()
        new_network.eval()
        old_gate.eval()
        new_gate.eval()
        with torch.no_grad():
            self.MODE = TEST
            old_outputs1 = old_network(inputs1,self.MODE,old_block[5],old_block[6],old_block[7])
            old_outputs2 = old_network(inputs2,self.MODE,old_block[5],old_block[6],old_block[7])
            old_features1 = old_outputs1["raw_features"]
            old_features2 = old_outputs2["raw_features"]
            old_im_features1 = old_outputs1["intermdeiate"]
            old_im_features2 = old_outputs2["intermdeiate"]
            old_values1 = old_gate(old_im_features1.detach())*old_features1.detach()
            old_values2 = old_gate(old_im_features2.detach())*old_features2.detach()
            outputs1 = new_network(inputs1,new_gate,old_values1)
            outputs2 = new_network(inputs2,new_gate,old_values2)
            features1, logits1, proj1, pred1,gate1 = outputs1["raw_features"], outputs1["logits"],outputs1["projection"],outputs1["prediction"],outputs1["new_gate"]
            features2, logits2, proj2, pred2,gate2 = outputs2["raw_features"], outputs2["logits"],outputs2["projection"],outputs2["prediction"],outputs2["new_gate"]

        self.MODE = TRAIN

        Stu_outputs1 = training_network(inputs1,self.MODE,block[5],block[6],block[7])
        Stu_outputs2 = training_network(inputs2,self.MODE,block[5],block[6],block[7])
        Stu_features1, Stu_logits1, Stu_proj1, Stu_pred1 = Stu_outputs1["raw_features"], Stu_outputs1["logits"],Stu_outputs1["projection"],Stu_outputs1["prediction"]
        Stu_features2, Stu_logits2, Stu_proj2, Stu_pred2 = Stu_outputs2["raw_features"], Stu_outputs2["logits"],Stu_outputs2["projection"],Stu_outputs2["prediction"]

        # if not finetune:
        ncaloss1 = losses.nca(
            Stu_logits1,
            targets,
        )
        ncaloss2 = losses.nca(
            Stu_logits2,
            targets,
        )
        ncaloss = (ncaloss1+ncaloss2)/2
        loss+=ncaloss
        self._metrics["ncaloss"] += ncaloss.item()
        #try scale
        encoder_loss =losses.compute_contra_loss(Stu_proj1.detach(),Stu_proj2.detach(), Stu_pred1, Stu_pred2,targets,epoch)
        self._metrics["enc"] += encoder_loss.item()
        loss += encoder_loss

        ecd_angle_loss = losses.compute_contra_loss(pred1.detach(), pred2.detach(), Stu_pred1,
                                                                Stu_pred2, targets,epoch)

        self._metrics["ecag"] += ecd_angle_loss.item()
        loss += ecd_angle_loss.item()
        pod_flat_loss = 0.5*(losses.embeddings_similarity(features2.detach(),Stu_features1) + losses.embeddings_similarity(features1.detach(), Stu_features2))
        self._metrics["pod"] += pod_flat_loss.item()
        loss += pod_flat_loss

        l2_loss = self.l2*l2_alpha
        loss += l2_loss.item()
        self._metrics["l2_loss"] += l2_loss.item()   
        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss


    def _distill_loss(
            self,
            training_network,
            teacher_network,
            l2_alpha,
            inputs1,
            inputs2,
            targets,
            memory_flags,
            finetune,
            epoch,
            threshold
    ):
        inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)

        outputs1 = training_network(inputs1,self.MODE,self.dpblock[5],self.dpblock[6],self.dpblock[7])
        outputs2 = training_network(inputs2,self.MODE,self.dpblock[5],self.dpblock[6],self.dpblock[7])

        sp_out1 = teacher_network(inputs1)
        sp_out2 = teacher_network(inputs2)
        features1, logits1, proj1, pred1 = outputs1["raw_features"], outputs1["logits"],outputs1["projection"],outputs1["prediction"]
        features2, logits2, proj2, pred2 = outputs2["raw_features"], outputs2["logits"],outputs2["projection"],outputs2["prediction"]
        sp_fea1, sp_log1, sp_proj1, sp_pred1 = sp_out1["raw_features"], sp_out1["logits"], sp_out1["projection"],sp_out1["prediction"]
        sp_fea2, sp_log2, sp_proj2, sp_pred2 = sp_out2["raw_features"], sp_out2["logits"], sp_out2["projection"],sp_out2["prediction"]
        ncaloss1 = losses.nca(
            logits1,
            targets,
        )
        ncaloss2 = losses.nca(
            logits2,
            targets,
        )
        tcaloss1 = losses.nca(
            sp_log1,
            targets,
        )
        tcaloss2 = losses.nca(
            sp_log2,
            targets,
        )
        tem = 5
        Tr = losses.compute_relation(pred1.detach(), targets)
        value_mask,tec_loss = losses.compute_adverse_loss(sp_proj1.detach(), sp_proj2.detach(), sp_pred1, sp_pred2,pred1,pred2, targets,Tr,epoch,threshold)
        scale = 1 if epoch <= threshold else epoch / threshold
        encoder_loss = scale*losses.compute_contra_loss(proj1.detach(),proj2.detach(), pred1, pred2,targets,epoch)
        distill_loss = scale*losses.compute_contra_loss(sp_pred2.detach(),sp_pred1.detach(), pred1, pred2,targets,epoch)
        distill_loss2 = scale*(losses.MSE(sp_fea1.detach()/tem, features1/tem)+losses.MSE(sp_fea2.detach()/tem, features2/tem))/2
        loss =(ncaloss1+ncaloss2)/2+encoder_loss+tec_loss+(tcaloss1+tcaloss2)/2+self.l2*l2_alpha+(distill_loss+distill_loss2)
        self._metrics["distloss"] += loss.item()
        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _after_task(self, inc_dataset):
        if self._task == 0:
            # self._network = self.best_backbone
            self._old_model = self.best_backbone.freeze()

            # self.dpblock = self.best_dpblock
            self.old_dpblock = self.best_dpblock
            self._initial_block(self.dpblock)
            self.dpblock = copy.deepcopy(self.old_dpblock)
            self._network = quantinet.BasicNet(
            self._convnet,
            self.dpblock[0],
            self.dpblock[1],
            self.dpblock[2],
            self.dpblock[3],
            self.dpblock[4],
            self._initial_channel,
            device=self._device,
            )
            self._network.shallow_net.load_state_dict(self._old_model.shallow_net.state_dict())

        else:

            self._old_model = None
            self.old_dpblock = copy.deepcopy(self.dpblock)
            self._old_model = quantinet.BasicNet(
            self._convnet,
            self.old_dpblock[0],
            self.old_dpblock[1],
            self.old_dpblock[2],
            self.old_dpblock[3],
            self.old_dpblock[4],
            self._initial_channel,
            device=self._device,
            )
            self._old_model.add_classes(self._n_classes)
            self._old_model.load_state_dict(self._network.state_dict())

            self._old_model.freeze()
            self._initial_block(self.dpblock)
            self.dpblock = copy.deepcopy(self.old_dpblock)
            self._network = quantinet.BasicNet(
            self._convnet,
            self.dpblock[0],
            self.dpblock[1],
            self.dpblock[2],
            self.dpblock[3],
            self.dpblock[4],
            self._initial_channel,
            device=self._device,
            )

            self._network.shallow_net.load_state_dict(self._old_model.shallow_net.state_dict())

    def _eval_task(self, test_loader):
        self.MODE = TEST
        self._old_model.eval()
        with torch.no_grad():
            all_out = []
            all_label = []
            T_out = []
            # sp_label = []
            for step, input_dict in enumerate(test_loader):
                inputs = input_dict["inputs"].to(self._device)
                targets = input_dict["targets"]
                out = self._old_model(inputs, self.MODE,self.old_dpblock[5],self.old_dpblock[6],self.old_dpblock[7])["logits"].detach()
                preds = F.softmax(out, dim=-1)
                _, index = preds.max(dim=1)
                all_out.append(index)
                all_label.append(targets)
            all_out = torch.cat(all_out, dim=0)
            all_label = torch.cat(all_label, dim=0)
            return all_out,all_label

    def _eval_first_task(self, test_loader):
        self.MODE = TEST
        with torch.no_grad():
            all_out = []
            all_label = []
            T_out = []
            for step, input_dict in enumerate(test_loader):
                inputs = input_dict["inputs"].to(self._device)
                targets = input_dict["targets"]
                out = self._network(inputs, self.MODE,self.dpblock[5],self.dpblock[6],self.dpblock[7])["logits"].detach()
                preds = F.softmax(out, dim=-1)
                _, index = preds.max(dim=1)
                all_out.append(index)
                all_label.append(targets)
            all_out = torch.cat(all_out, dim=0)
            all_label = torch.cat(all_label, dim=0)
            OAC, KAPPA, _, _, _, AAC,_ = evaluation.cluster_accuracy(all_label.cpu().numpy(), all_out.cpu().numpy())
            return OAC

                
    def build_examplars(
        self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []


        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            if self._task == 0:
                network = self.best_backbone
                block = self.best_dpblock
            else:
                network  = self._network
                block = self.dpblock

            features, targets = self.extract_features(network,block, loader)

            if isinstance(memory_per_class,list):
                current_percentage = memory_per_class[class_idx]
            else:
                current_percentage = memory_per_class
            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, current_percentage)
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[class_idx][:current_percentage]
                herding_indexes[class_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            # Re-computing the examplar mean (which may have changed due to the training):


            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])


        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes


    def _initial_block(self,block):
        for linear in range(len(block[0])):
            block[0][linear] = (torch.eye(block[0][linear].shape[0],block[0][linear].shape[1]) + 1e-7 * torch.ones(block[0][linear].shape[0],block[0][linear].shape[1])).to('cuda')
            block[0][linear].requires_grad = True
            block[0][linear].retain_grad()
        for gamma in  range(len(block[1])):
            block[1][gamma] =  torch.ones(block[1][gamma].shape).to('cuda')
            block[1][gamma].requires_grad = True
            block[1][gamma].retain_grad()
        for beta in range(len(block[2])):
            block[2][beta] = 1e-5 * torch.ones(block[2][beta].shape).to('cuda')
            block[2][beta].requires_grad = True
            block[2][beta].retain_grad()
        for cita in range(len(block[3])):
            block[3][cita] = torch.Tensor([30]).to('cuda')
            block[3][cita].requires_grad = True
            block[3][cita].retain_grad()
        for bn in range(len(block[4])):
            block[4][bn] = dict_set(
                torch.zeros(block[4][bn]['running_mean'].shape),
                torch.zeros(block[4][bn]['running_mean'].shape))

    def extract_features(self,model,block, loader):
        targets, features = [], []
        model.eval()
        self.MODE = TEST
        for input_dict in loader:
            inputs, _targets = input_dict["inputs"], input_dict["targets"]

            _targets = _targets.numpy()
            _features = model.extract(inputs.to(model.device),self.MODE,block[5],block[6],block[7]).detach().cpu().numpy()

            features.append(_features)
            targets.append(_targets)

        model.train()

        return np.concatenate(features), np.concatenate(targets)

    def get_memory(self):
        return self._data_memory, self._targets_memory

    def _after_task_intensive(self, inc_dataset):
        self._data_memory, self._targets_memory, self._herding_indexes = self.build_examplars(
            inc_dataset, self._herding_indexes
        )
    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )
    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred/T, dim=1)
        soft = torch.softmax(soft/T, dim=1)
        # soft = soft
        soft = soft / soft.sum(1)[:, None]
        return -1*torch.mul(soft, pred).sum()/pred.shape[0]


