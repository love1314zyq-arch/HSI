import datetime
import logging
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)



def check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.).item())


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")



def add_new_weights(network, weight_generation, current_nb_classes, task_size, inc_dataset,MODE=None,weight_list=None,orign_list=None,inter_list=None):
    if isinstance(weight_generation, str):
        warnings.warn("Use a dict for weight_generation instead of str", DeprecationWarning)
        weight_generation = {"type": weight_generation}

    if weight_generation["type"] == "imprinted":
        logger.info("Generating imprinted weights")
        print(10*"**")
        network.add_imprinted_classes(
            list(range(current_nb_classes, current_nb_classes + task_size)), inc_dataset,MODE,weight_list,orign_list,inter_list,
            **weight_generation
        )
    elif weight_generation["type"] == "embedding":
        logger.info("Generating embedding weights")

        mean_embeddings = []
        for class_index in range(current_nb_classes, current_nb_classes + task_size):
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = extract_features(network, loader)
            features = features / np.linalg.norm(features, axis=-1)[..., None]

            mean = np.mean(features, axis=0)
            if weight_generation.get("proxy_per_class", 1) == 1:
                mean_embeddings.append(mean)
            else:
                std = np.std(features, axis=0, ddof=1)
                mean_embeddings.extend(
                    [
                        np.random.normal(loc=mean, scale=std)
                        for _ in range(weight_generation.get("proxy_per_class", 1))
                    ]
                )

        network.add_custom_weights(np.stack(mean_embeddings))
    elif weight_generation["type"] == "basic":
        network.add_classes(task_size)
    elif weight_generation["type"] == "ghosts":
        features, targets = weight_generation["ghosts"]
        features = features.cpu().numpy()
        targets = targets.cpu().numpy()

        weights = []
        for class_id in range(current_nb_classes, current_nb_classes + task_size):
            indexes = np.where(targets == class_id)[0]

            class_features = features[indexes]
            if len(class_features) == 0:
                raise Exception(f"No ghost class_id={class_id} for weight generation!")
            weights.append(np.mean(class_features, axis=0))

        weights = torch.tensor(np.stack(weights)).float()
        network.add_custom_weights(weights, ponderate=weight_generation.get("ponderate"))
    else:
        raise ValueError("Unknown weight generation type {}.".format(weight_generation["type"]))
        
def extract_features(model,MODE,weight_list,orign_list,inter_list , loader):
    targets, features = [], []
    model.eval()
    for input_dict in loader:
        inputs, _targets = input_dict["inputs"], input_dict["targets"]
        _targets = _targets.numpy()
        _features = model.extract(inputs.to(model.device),MODE,weight_list,orign_list,inter_list).detach().cpu().numpy()

        features.append(_features)
        targets.append(_targets)

    model.train()

    return np.concatenate(features), np.concatenate(targets)

