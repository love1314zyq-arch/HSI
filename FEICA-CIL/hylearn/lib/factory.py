import warnings

import torch
from torch import optim

from hylearn.lib.modules import shallow_backbone, sample_backbone
from hylearn.lib import data
from hylearn import models


def get_convnet(convnet_type, linear, gamma, beta,cita,bn,channel):
    if convnet_type == "lsc_backbone":
        return shallow_backbone.shallow_net(linear, gamma, beta,cita,bn,channel)
    # if convnet_type == "resnet101":
    #     return resnet.resnet101(**kwargs)
    # if convnet_type == "resnet18_mtl":
    #     return resnet_mtl.resnet18(**kwargs)
    # elif convnet_type == "resnet34":
    #     return resnet.resnet34(**kwargs)
    # elif convnet_type == "resnet32":
    #     return resnet.resnet32(**kwargs)
    # elif convnet_type == "rebuffi":
    #     return my_resnet.resnet_rebuffi(**kwargs)
    # elif convnet_type == "rebuffi_brn":
    #     return my_resnet_brn.resnet_rebuffi(**kwargs)
    # elif convnet_type == "myresnet18":
    #     return my_resnet2.resnet18(**kwargs)
    # elif convnet_type == "myresnet34":
    #     return my_resnet2.resnet34(**kwargs)
    # elif convnet_type == "densenet121":
    #     return densenet.densenet121(**kwargs)
    # elif convnet_type == "ucir":
    #     return ucir_resnet.resnet32(**kwargs)
    # elif convnet_type == "rebuffi_mcbn":
    #     return my_resnet_mcbn.resnet_rebuffi(**kwargs)
    # elif convnet_type == "rebuffi_mtl":
    #     return my_resnet_mtl.resnet_rebuffi(**kwargs)
    # elif convnet_type == "vgg19":
    #     return vgg.vgg19_bn(**kwargs)

    raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))

def get_teacher(convnet_type,channel):
    if convnet_type == "lsc_backbone":
        return sample_backbone.shallow_net(channel)
    raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))

def get_model(args,dpblock):
    dict_models = {
        "lsc": models.LSCNet,
        # "lwf": None,
        # "e2e": models.End2End,
        # "fixed": None,
        # "oracle": None,
        # "bic": models.BiC,
        # "ucir": models.UCIR,
        # "podnet": models.PODNet,
        # "lwm": models.LwM,
        # "zil": models.ZIL,
        # "gdumb": models.GDumb
    }

    model = args["model"].lower()

    if model not in dict_models:
        raise NotImplementedError(
            "Unknown model {}, must be among {}.".format(args["model"], list(dict_models.keys()))
        )

    return dict_models[model](args,dpblock)


def get_data(args, class_order=None):
    return data.IncrementalDataset(
        args,
        dataset_name=args["dataset"],
        random_order=args["random_classes"],
        shuffle=True,
        batch_size=args["batch_size"],
        workers=args["workers"],
        validation_split=args["validation"],
        onehot=args["onehot"],
        increment=args["increment"],
        initial_increment=args["initial_increment"],
        sampler=get_sampler(args),
        sampler_config=args.get("sampler_config", {}),
        data_path=args["data_path"],
        class_order=class_order,
        seed=args["seed"],
        dataset_transforms=args.get("dataset_transforms", {}),
        all_test_classes=args.get("all_test_classes", False),
        metadata_path=args.get("metadata_path")

    )


def set_device(args):
    devices = []

    for device_type in args["device"]:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device_type))

        devices.append(device)

    args["device"] = devices


def get_sampler(args):
    if args["sampler"] is None:
        return None

    sampler_type = args["sampler"].lower().strip()

    if sampler_type == "npair":
        return data.NPairSampler
    elif sampler_type == "triplet":
        return data.TripletSampler
    elif sampler_type == "tripletsemihard":
        return data.TripletCKSampler

    raise ValueError("Unknown sampler {}.".format(sampler_type))

