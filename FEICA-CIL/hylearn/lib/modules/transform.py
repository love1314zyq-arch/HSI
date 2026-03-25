import torch
import torchvision
import numpy as np
from sklearn.decomposition import PCA
from sklearn import random_projection
import random
from PIL import ImageFilter
# class GaussianBlur:
#     def __init__(self, kernel_size, min=0.1, max=2.0):
#         self.min = min
#         self.max = max
#         self.kernel_size = kernel_size
#
#     def __call__(self, img):
#         prob = np.random.random_sample()
#         if prob < 0.5:
#             img = np.array(img)
#             sigma = (self.max - self.min) * np.random.random_sample() + self.min
#             img = c(img, (self.kernel_size, self.kernel_size), sigma)
#             img = torch.from_numpy(img).float()
#         return img
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
class Transforms:
    def __init__(self, size, s=1.0, mean=None, std=None, blur=False):
        self.train_transform = [

            torchvision.transforms.RandomChoice([torchvision.transforms.CenterCrop(21),
                                                 torchvision.transforms.CenterCrop(27),
                                                 torchvision.transforms.CenterCrop(25),
                                                 torchvision.transforms.CenterCrop(23)]),
            torchvision.transforms.Resize(size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([MaskMixed(p=0.3)], p=0.5),
            # torchvision.transforms.RandomChoice([
            #                                      MaskPixels(p=0.5),
            #                                      MaskBands(p=0.5)
            #                                      ]),
            ]
        self.test_transform = []
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform(x)


class GroupPermuteBands(object):
    """
    shuffle bands into n_groups
    """

    def __init__(self, n_group=3):
        self.n_group = n_group

    def __call__(self, img):
        n_channel = img.size(0)
        n_group_band = int(np.ceil(n_channel / self.n_group))
        for i in range(self.n_group):
            start = i * n_group_band
            end = start + n_group_band
            if end >= n_channel:
                indx = np.arange(start, n_channel)
                indx_ = np.arange(start, n_channel)
            else:
                indx = np.arange(start, end)
                indx_ = np.arange(start, end)
            np.random.shuffle(indx)
            img[indx_] = img[indx]
        # indx_selected = indx[:n_shuffle]
        # select_mask = np.zeros((n_channel, 1, 1))
        # select_mask[indx_selected] = 1
        # img_shuffled = img[indx]

        return img


class Empty(object):
    def __init__(self, p=0.5):
        """
        :param p:  every pixel will be masked  with a probability of p
        """
        self.p = 1 - p

    def __call__(self, img):
        # n_band, h, w = img.shape
        # mask = np.random.binomial(1, self.p, size=(h, w))
        # mask = torch.from_numpy(mask).float()
        # mask = mask.expand((n_band, h, w))
        # sup_matrix = (1-mask)*torch.ones(mask.shape)*0.5
        # img = mask * img+sup_matrix
        return img


class MaskPixels(object):
    def __init__(self, p=0.5):
        """
        :param p:  every pixel will be masked  with a probability of p
        """
        self.p = 1 - p

    def __call__(self, img):
        n_band, h, w = img.shape
        mask = np.random.binomial(1, self.p, size=(h, w))
        mask = torch.from_numpy(mask).float()
        mask = mask.expand((n_band, h, w))
        sup_matrix = (1 - mask) * torch.ones(mask.shape) * 0.5
        img = mask * img + sup_matrix
        return img


class MaskOccusion(object):
    def __init__(self, p=0.5):
        """
        :param p:  every pixel will be masked  with a probability of p
        """
        self.p = 1 - p

    def __call__(self, img):
        # print(img.shape)
        #
        # Vmin = 0
        # Vmax = 0.03125
        # Rmin = 0.2
        # Rmax= 1/Rmin
        # Lmin = 0.2
        # Lmax = 1/Lmin
        # Ve = random.uniform(Vmin,Vmax)*V
        # Re = random.uniform(Rmin,Rmax)
        # Le = random.uniform(Lmin,Lmax)
        # x = round((Ve*Le/(Re**2))**(1/3))
        # y = round((Ve*Le*Re)**(1/3))
        # z = round((Ve*Re/(Le**2))**(1/3)*(6/27))
        V = 27 * 27
        Vmin = 0
        Vmax = 0.25
        Rmin = 0.3
        Rmax = 3
        Re = random.uniform(Rmin, Rmax)
        Ve = random.uniform(Vmin, Vmax) * V
        x = round((Ve * Re) ** (1 / 2))
        y = round((Ve / Re) ** (1 / 2))
        z = random.randint(0, 2)
        prob = np.random.binomial(1, self.p, size=(img.shape[0]))
        prob = torch.from_numpy(prob).float()
        index = torch.argwhere(prob == 1).squeeze(1)
        # print(x,y,z)
        if x * y * z != 0:
            x_s = random.randint(0, 27 - x)
            y_s = random.randint(0, 27 - y)
            z_s = random.randint(0, 8 - z)
            # print(z_s)
            mask = torch.ones(img.shape)
            for i in index:
                for j in range(x_s, x_s + x):
                    for p in range(y_s, y_s + y):
                        mask[i, j, p] = 0
            fake_mask = (1 - mask) * 0.5
            img = mask * img + fake_mask
            return img
        else:
            return img


class MaskMixed(object):
    def __init__(self, p=0.5):
        """
        :param p:  every pixel will be masked  with a probability of p
        """
        self.p = 1 - p

    def __call__(self, img):
        # print("enter maskmixed")
        n_band, h, w = img.shape
        mask = np.random.binomial(1, self.p, size=(h, w))
        mask = torch.from_numpy(mask).float()
        mask = mask.expand((n_band, h, w))
        prob = np.random.binomial(1, self.p, size=(img.shape[0], 1, 1))
        prob = torch.from_numpy(prob).float()
        prob = prob.expand(img.shape[0], h, w)
        # prob = torch.from_numpy(prob).float()

        mixed = (1 - prob) * mask + prob

        # sup_matrix = (1 - mixed) * np.random.normal(0,1, size=(mixed.shape)).astype(float)
        sup_matrix = (1 - mixed) * torch.ones(mixed.shape) * np.random.rand()*0.5
        img = img * mixed + sup_matrix
        return img


class MaskBands(object):

    def __init__(self, p=0.5):
        """

        :param p: every band will be masked with a probability of p
        """
        self.p = 1. - p

    def __call__(self, img):
        # indx = np.arange(img.shape[0])
        # indx_selected = np.random.choice(indx, self.n_band, replace=False)
        # img = img[indx_selected]

        prob = np.random.binomial(1, self.p, img.shape[0])
        prob = np.reshape(prob, (img.shape[0], 1, 1))
        prob = torch.from_numpy(prob).float()
        # if torch.sum(1-prob) == 1:
        #     index = torch.argwhere(prob == 1).squeeze(1)
        #     avg = torch.mean(img[index,:,:])
        #     img[index,:,:] = avg
        # elif torch.sum(1-prob) >1:
        #     index = torch.argwhere(prob == 1).squeeze(1)
        #     # print(index.shape)
        #     avg = torch.mean(torch.index_select(img,dim =0,index=index),dim = 0)
        #     # print(avg.shape)
        #     for i in index:
        #         img[i,:,:] = avg

        # img = img[np.where(prob == 1)]
        sup_matrix = (1 - prob) * torch.ones(img.shape) * 0.5
        img = img * prob + sup_matrix
        return img


class RandomProjectionBands(object):

    def __init__(self, n_band=None):
        """
        :param n_band: project to n_band
        """
        self.n_band = n_band

    def __call__(self, img):
        # # n_band * w * h
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        n_band, h, w = img.shape
        if self.n_band is None:
            # self.n_band = np.random.randint(3, n_band//2)
            transformer = random_projection.SparseRandomProjection(n_components='auto')
        else:
            transformer = random_projection.SparseRandomProjection(n_components=self.n_band)
        img_ = img.transpose((1, 2, 0))
        x_2d = img_.reshape((-1, n_band))
        x_2d_ = transformer.fit_transform(x_2d)
        img_new = x_2d_.reshape((h, w, -1)).transpose(2, 0, 1)
        img_new = torch.from_numpy(img_new).float()
        return img_new


class ShufflePixel(object):

    def __init__(self):
        pass

    def __call__(self, img):
        n_band, h, w = img.shape
        img_ = img.view(n_band, -1)
        img_ = img_[torch.randperm(n_band)]
        img = img_.view(n_band, h, w)
        return img

