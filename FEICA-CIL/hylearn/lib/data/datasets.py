import collections
import glob
import logging
import math
import os
import warnings
from hylearn.lib.modules import transform,HSI_Loader
import numpy as np
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None
    open_image = False

    def set_custom_transforms(self, transforms):
        if transforms:
            raise NotImplementedError("Not implemented for modified transforms.")
    def train_transforms(self,args):
        return None
    def test_transforms(self,args):
        return None
class SalinasA(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'Salinas_corrected', 'Salinas_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []
class Indian_pines(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            matrix = args["matrix"],rate = args["rate"],classes = args["classes"],
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []

class Pavia_university(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'PaviaU', 'PaviaU_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []
class Houston(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'Houston', 'Houston_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []
class KSC(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'KSC', 'KSC_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []
class Botswana(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'Botswana', 'Botswana_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt,is_train= False, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []

class Longkou(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'WHU_Hi_LongKou', 'WHU_Hi_LongKou_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt,is_train= False, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []

class Hanchuan(DataHandler):
    def __init__(self):
        self.im_, self.gt_ = 'WHU_Hi_HanChuan', 'WHU_Hi_HanChuan_gt'

    def base_dataset(self, data_path, args,train=True, download=False):
        img_path = data_path + self.im_ + '.mat'
        gt_path = data_path + self.gt_ + '.mat'
        train_data, test_data, train_gt, test_gt = HSI_Loader.HSI_DataSplit(img_path, gt_path, args["split_rate"],patch_size=(
                                                                            args["image_size"], args["image_size"]),
                                                                            pca=True, pca_dim=args["channels"],
                                                                            is_labeled=True)
        _train_dataset = HSI_Loader.HSI_Data(train_data, train_gt,is_train= True,is_labeled=True)
        _testdataset = HSI_Loader.HSI_Data(test_data, test_gt,is_train= False, is_labeled=True)
        return _train_dataset, _testdataset

    def train_transforms(self,args):
        return transform.Transforms(size=27, s=0.5)
    def test_transforms(self,args):
        return []