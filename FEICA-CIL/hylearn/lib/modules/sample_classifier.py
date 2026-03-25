import copy
import logging

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

import hylearn.lib.modules.distance as distance_lib
logger = logging.getLogger(__name__)
class CosineClassifier(nn.Module):
    classifier_type = "cosine"

    def __init__(
        self,
        features_dim,
        device,
        *,
        proxy_per_class=10,
        distance="cosine",
        merging="softmax",
        scaling=1,
        gamma=1.,
    ):
        super().__init__()

        self.n_classes = 0
        self._weights = nn.ParameterList([])
        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device
        self.distance = distance
        self.merging = merging
        self.gamma = gamma
        self.scaling = scaling


    def forward(self, features):
        weights = self.weights

        features = self.scaling * F.normalize(features, p=2, dim=-1)
        weights = self.scaling * F.normalize(weights, p=2, dim=-1)

        raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        # raw_similarities = distance_lib.cosine_similarity(features, weights)

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
            # similarities = raw_similarities.sum(-1)
        else:
            similarities = raw_similarities
        return {"logits": similarities, "raw_logits": raw_similarities}

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        bs = similarities.shape[0]

        if self.merging == "mean":
            return similarities.view(bs, n_classes, self.proxy_per_class).sum(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            # rep_att = taylor_softmax(self.gamma * simi_per_class, dim=-1, n=4)
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(bs, n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(bs, n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))

    # ------------------
    # Weights management
    # ------------------

    def align_features(self, features):
        avg_weights_norm = self.weights.data.norm(dim=1).mean()
        avg_features_norm = features.data.norm(dim=1).mean()

        features.data = features.data * (avg_weights_norm / avg_features_norm)
        return features

    def add_custom_weights(self, weights, ponderate=None, **kwargs):
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                weights = weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_new_weights_norm = weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_new_weights_norm
                weights = weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        self._weights.append(nn.Parameter(weights))
        self.to(self.device)

    def align_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        if len(self._weights) == 1:
            return

        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((old_norm / new_norm) * self._weights[-1])

    def align_weights_i_to_j(self, indexes_i, indexes_j):
        with torch.no_grad():
            base_weights = self.weights[indexes_i]

            old_norm = torch.mean(base_weights.norm(dim=1))
            new_norm = torch.mean(self.weights[indexes_j].norm(dim=1))

            self.weights[indexes_j] = nn.Parameter((old_norm / new_norm) * self.weights[indexes_j])

    def align_inv_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((new_norm / old_norm) * self._weights[-1])

    @property
    def weights(self):
        return torch.cat([clf for clf in self._weights])

    def add_classes(self, n_classes,old_weights=None):
        if old_weights is not None:
            # self._weights.extend(nn.Parameter(old_weights))
            if (self.proxy_per_class * n_classes-old_weights.shape[0])>0:
                new_weights = torch.zeros(self.proxy_per_class * n_classes-old_weights.shape[0], self.features_dim).to('cuda')
                nn.init.kaiming_normal_(new_weights, nonlinearity="relu")
                final_weights = nn.Parameter(torch.cat([old_weights,new_weights]))
                self._weights.append(final_weights)
            else:
                self._weights.append(nn.Parameter(old_weights))
        else:
                new_weights = nn.Parameter(torch.zeros(self.proxy_per_class * n_classes, self.features_dim))
                nn.init.kaiming_normal_(new_weights, nonlinearity="relu")

                self._weights.append(new_weights)            

        self.to(self.device)
        self.n_classes += n_classes
        return self

    # def inherit_classes(self, old_weights):
    #     for i in range(len(old_weights)):
    #         # self.weights[i] = old_weights[i]
    #         self._weights[0][i] = old_weights[i]
            
    def add_imprinted_classes(
        self, traindataset,trainlabel, network, multi_class_diff="normal", type=None
    ):
        if self.proxy_per_class > 1:
            logger.info("Multi class diff {}.".format(multi_class_diff))

        weights_norm = self.weights.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()

        new_weights = []
        class_indexes = np.unique(trainlabel)
        for class_index in class_indexes:
            data_index = [x for x in range(len(class_indexes)) if trainlabel[x]==class_index]
            classdataset = torch.index_select(traindataset,0,data_index)
            network.eval()
            output = network.extract(classdataset)
            features = output["raw_features"].detach().cpu().numpy()
            features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
            class_embeddings = torch.mean(features_normalized, dim=0)
            class_embeddings = F.normalize(class_embeddings, dim=0, p=2)

            if multi_class_diff == "normal":
                std = torch.std(features_normalized, dim=0)
                for _ in range(self.proxy_per_class):
                    new_weights.append(torch.normal(class_embeddings, std) * avg_weights_norm)
            elif multi_class_diff == "kmeans":
                clusterizer = KMeans(n_clusters=self.proxy_per_class)
                clusterizer.fit(features_normalized.numpy())

                for center in clusterizer.cluster_centers_:
                    new_weights.append(torch.tensor(center) * avg_weights_norm)
            else:
                raise ValueError(
                    "Unknown multi class differentiation for imprinted weights: {}.".
                    format(multi_class_diff)
                )

        new_weights = torch.stack(new_weights)
        self._weights.append(nn.Parameter(new_weights))
        self.to(self.device)

        return self

