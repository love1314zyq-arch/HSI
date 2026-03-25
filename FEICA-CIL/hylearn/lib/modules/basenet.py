import copy
import logging

import torch
from torch import nn
from hylearn.lib.modules.shallow_classifier import CosineClassifier
from hylearn.lib import factory
class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
        linear, gamma, beta,
        cita,
        bn,
        channel ,
        init="kaiming",
        device=None,
    ):
        super(BasicNet, self).__init__()


        self.shallow_net =  factory.get_convnet(convnet_type,linear, gamma, beta,cita,bn,channel)

        self.classifier = CosineClassifier(
            self.shallow_net.out_dim,linear[4],gamma[4],beta[4],cita[4],bn[4], device=device,proxy_per_class=10
        )

        self.device = device
        self.to(self.device)

    def forward(
        self, x,MODE,weight_list,orign_list,inter_list
    ):
        outputs = self.shallow_net(x, MODE,weight_list,orign_list,inter_list)
        selected_features = outputs["raw_features"]

        clf_outputs = self.classifier(selected_features,MODE,orign_list,inter_list)
        outputs.update(clf_outputs)

        return outputs
    def extract(self, x, MODE,weight_list,orign_list,inter_list):
        outputs = self.shallow_net(x, MODE,weight_list,orign_list,inter_list)
        return outputs["raw_features"]
    def copy(self):
        return copy.deepcopy(self)
    def add_classes(self, n_classes,old_weights=None):
        self.classifier.add_classes(n_classes,old_weights=old_weights)
    def add_imprinted_classes(self, class_indexes, inc_dataset,MODE,weight_list,orign_list,inter_list, **kwargs):
        if hasattr(self.classifier, "add_imprinted_classes"):
            self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self,MODE,weight_list,orign_list,inter_list, **kwargs)
    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "convnet":
            model = self.convnet
        elif model == "classifier":
            model = self.classifier
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for param in model.parameters():
            param.requires_grad = trainable
        if hasattr(self, "gradcam_hook") and self.gradcam_hook and model == "convnet":
            for param in self.convnet.last_conv.parameters():
                param.requires_grad = True

        if not trainable:
            model.eval()
        else:
            model.train()

        return self
    @property
    def features_dim(self):
        return self.shallow_net.out_dim