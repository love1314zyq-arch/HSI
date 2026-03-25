import torch
from torch.utils.data import Dataset, DataLoader
from hylearn.lib.Toolbox.Preprocessing import Processor
from sklearn.preprocessing import scale,StandardScaler, minmax_scale, normalize
from sklearn.decomposition import PCA
import numpy as np
import random
# from Toolbox.Preprocessing import split_each_class
import copy
np.set_printoptions(threshold=np.inf)

def mixup(sample1,sample2,lam):
    # lam = np.floor(np.random.uniform(0.1,1.0)*10)/10
    
    new_data = (1-lam)*sample2+lam*sample1
    return new_data
def cutmix(sample1,sample2,alpha):
    lam = np.random.beta(alpha, alpha)
    image_h, image_w = sample1.shape[0],sample1.shape[1]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))
    # sample2 = sample2.transpose(1,0,2)
    sample1[y0:y1, x0:x1,:] = sample2[y0:y1, x0:x1,:]
    return sample1
def data_aug(train,targets,matrix,rate,classes):
    for index,label in enumerate(classes):
        print(label)
        data = train[targets==label,:,:,:]
        print(data.shape)
        for sque,pair in enumerate(matrix[index]):
            # if sque<18:
            aug = mixup
            new_data = aug(data[pair[0],:,:,:],data[pair[1],:,:,:],rate[index][sque])
            train = np.append(train, np.expand_dims(new_data, axis=0),axis = 0)
            targets = np.append(targets,np.expand_dims(label, axis=0),axis = 0)
    return train, targets

def HSI_DataSplit(path_to_data, path_to_gt,split_rate, patch_size=(11, 11),matrix=None,rate=None,classes=None  ,pca=True, pca_dim=8, is_labeled=True):
    p = Processor()
    img, gt = p.prepare_data(path_to_data, path_to_gt)
    n_row, n_column, n_band = img.shape
    if pca:
        img = StandardScaler().fit_transform(img.reshape(n_row * n_column, -1))  # .reshape((n_row, n_column, -1))
        pca = PCA(n_components=pca_dim)
        img = pca.fit_transform(img).reshape((n_row, n_column, pca_dim))
    print(patch_size[0], patch_size[1])
    x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
    print(x_patches.shape)
    
    X_tr_, X_ts, y_tr, y_ts = p.split_each_class(x_patches,y_,split_rate)
    if matrix and rate is not None:
        X_tr_cr,y_tr_cr = data_aug(copy.deepcopy(X_tr_),copy.deepcopy(y_tr),matrix,rate,classes)
        return X_tr_cr, X_ts, y_tr_cr,y_ts
    else:
        return X_tr_, X_ts, y_tr,y_ts
#     return p.split_each_class(x_patches,y_,split_rate)
class HSI_Data(Dataset):

    def __init__(self, x_patches,y_,is_train=False, is_labeled=True ):
        # super(HSI_Data, self).__init__(root, transform=transform, target_transform=target_transform)
        p = Processor()
        # print("1p\n")
        y = p.standardize_label(y_)
        # print("2p\n")
        if not is_labeled:
            self.n_classes = np.unique(y).shape[0] - 1
        else:
            self.n_classes = np.unique(y).shape[0]
        # n_class = np.unique(y).shape[0]
        n_samples, n_row, n_col, n_channel = x_patches.shape
        self.data_size = n_samples
        x_patches = StandardScaler().fit_transform(x_patches.reshape((n_samples, -1))).reshape((n_samples, n_row, n_col, -1))
        # if is_train:
        #     x_patches,y =  data_aug(copy.deepcopy(x_patches),copy.deepcopy(y),matrix,rate,classes)
        x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
        self.x_tensor, self.y_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor), \
                                       torch.from_numpy(y).type(torch.LongTensor)
    def get_data(self):
        return self.x_tensor, self.y_tensor
    
    def __getitem__(self, idx):
        x, y = self.x_tensor[idx], self.y_tensor[idx]
        # if self.transform is not None:
        #     x = self.transform(x)
        return x, y

    def __len__(self):
        return self.data_size

