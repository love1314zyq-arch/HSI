import copy
import logging

import torch
from torch import nn
from hylearn.lib.modules.sample_classifier import CosineClassifier
from hylearn.lib import factory
class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
            channel,
        init="kaiming",
        device=None,
    ):
        super(BasicNet, self).__init__()


        self.shallow_net =  factory.get_teacher(convnet_type,channel)

        self.classifier = CosineClassifier(
            self.shallow_net.out_dim, device=device,proxy_per_class=10
        )

        self.device = device
        self.to(self.device)

    def forward(
        self, x,gate=None,old_value=None
    ):
        outputs = self.shallow_net(x,gate=gate,old_value= old_value)
        selected_features = outputs["raw_features"]

        clf_outputs = self.classifier(selected_features)
        outputs.update(clf_outputs)

        return outputs
    def extract(self, x):
        outputs = self.shallow_net(x)
        return outputs["raw_features"]
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
    def add_classes(self, n_classes,old_weights=None):
        self.classifier.add_classes(n_classes,old_weights=old_weights)