import torch
import torch.nn as nn
import numpy as np
import os
TRAIN = torch.ones(1)
TEST = torch.zeros(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# def LBNforward(x,gamma,beta,bn_param,MODE):
#     # mode = bn_param['mode']
#     mode = MODE
#     eps = bn_param.get('eps', 1e-5)
#     momentum = bn_param.get('momentum',0.9)
#     # x = x.numpy()
#     N,D = x.shape
#     # print(N,D)
#     # print(gamma.shape)
#     running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype))
#     running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype))
#     running_mean = running_mean.to(device)
#     running_var = running_var.to(device)
#     out ,cache = None,None
#
#     if mode == TRAIN:
#         # print('trainmode')
#         sample_mean = torch.mean(x, 0).to(device)
#         sample_var = torch.var(x, 0).to(device)
#         out_ = (x - sample_mean) / torch.sqrt(sample_var + eps)
#         out_ = out_.to(device)
#         running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#         running_var = momentum * running_var + (1 - momentum) * sample_var
#
#         out = gamma * out_ + beta
#         cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)
#
#     elif mode == TEST:
#         # print('testmode')
#         scale = gamma / torch.sqrt(running_var + eps)
#         out = x * scale + (beta - running_mean * scale)
#
#     else:
#         raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
#
#     # Store the updated running means back into bn_param
#     bn_param['running_mean'] = running_mean
#     bn_param['running_var'] = running_var
#     # out = torch.from_numpy(out)
#     return out,cache
#
# def LBNbackward(dout,cache):
#     dx, dgamma, dbeta = None, None, None
#
#     out_, x, sample_var, sample_mean, eps, gamma, beta = cache
#
#     N = x.shape[0]
#     dout_ = gamma * dout
#     dvar = torch.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5, 0)
#     dx_ = 1 / torch.sqrt(sample_var + eps)
#     dvar_ = 2 * (x - sample_mean) / N
#
#     # intermediate for convenient calculation
#     di = dout_ * dx_ + dvar * dvar_
#     dmean = -1 * torch.sum(di, 0)
#     dmean_ = torch.ones_like(x) / N
#
#     dx = di + dmean * dmean_
#     dgamma = torch.sum(dout * out_, 0)
#     dbeta = torch.sum(dout, 0)
#
#     return dx, dgamma, dbeta
def LBNforward(x,gamma,beta,bn_param,MODE):
    # mode = bn_param['mode']
    mode = MODE
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum',0.9)
    # x = x.numpy()
    N,D = x.shape
    # print(x.shape)
    # print(N,D)
    # print(gamma.shape)
    running_mean = bn_param.get('running_mean', torch.zeros((N,D), dtype=x.dtype))
    running_var = bn_param.get('running_var', torch.zeros((N,D), dtype=x.dtype))
    running_mean = running_mean.to(device)
    running_var = running_var.to(device)
    out ,cache = None,None

    if mode == TRAIN:
        # print('trainmode')
        sample_mean = torch.mean(x, 1,keepdim=True).to(device)
        # print(sample_mean.shape)
        sample_var = torch.var(x, 1,keepdim=True).to(device)
        # print(x.shape)
        out_ = (x - sample_mean) / torch.sqrt(sample_var + eps)
        out_ = out_.to(device)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        # print(out_.shape)

        out = out_*gamma+ beta
        cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)

    elif mode == TEST:
        # print('testmode')
        # print(running_var)
        # sample_mean = torch.mean(x, 1,keepdim=True).to(device)
        # # print(sample_mean.shape)
        # sample_var = torch.var(x, 1,keepdim=True).to(device)
        # running_mean = sample_mean
        # running_var = sample_var
        scale = gamma / torch.sqrt(running_var + eps)
        out = x * scale + (beta - running_mean * scale)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    # out = torch.from_numpy(out)
    return out,cache

def LBNbackward(dout,cache):
    dx, dgamma, dbeta = None, None, None

    out_, x, sample_var, sample_mean, eps, gamma, beta = cache

    D = x.shape[1]
    dout_ = gamma * dout
    dvar = torch.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5, 1,keepdim=True)
    dx_ = 1 / torch.sqrt(sample_var + eps)
    dvar_ = 2 * (x - sample_mean) / D

    # intermediate for convenient calculation
    di = dout_ * dx_ + dvar * dvar_
    dmean = -1 * torch.sum(di, 1,keepdim=True)
    dmean_ = torch.ones_like(x) / D
    channel = dout.shape[0]
    dx = di + dmean * dmean_
    dgamma = torch.sum(dout * out_, 1).view(channel,1)
    dbeta = torch.sum(dout, 1).view(channel,1)
    # print(dx.shape)
    # print(dgamma.shape)
    # print(dbeta.shape)
    return dx, dgamma, dbeta

# class bn_param():
    # def __init__(self, mode, running_mean = None,running_var = None,eps = 1e-5, momentum = 0.9):
    #     self.mode = mode,
    #     self.running_mean = running_mean
    #     self.running_var = running_var
    #     self.eps = eps
    #     self.momentum = momentum

def dict_set(  running_mean = None,running_var = None,eps = 1e-5, momentum = 0.9):
    # dict = {'mode':mode,'running_mean':running_mean,'running_var':running_var,'eps':eps,'momentum':momentum}
    dict = {'running_mean': running_mean.to('cuda'), 'running_var': running_var.to('cuda'), 'eps': eps, 'momentum': momentum}
    new_dict ={}
    for key, value in dict.items():
        if value is not None:
            new_dict[key] = value
    return new_dict

def sigmoidforward(x):
    out = 1 / (1+torch.exp(-x))
    return out

def sigmoidbackward(out,dout):
    dx = dout*(1-out)*out
    return dx


def reluforward(x):
    mask = x.ge(0)
    # mask2 = x.lt(0)
    out = mask*x
    # out2 = mask2*x*0.01
    # out=out1+out2
    return out

def relubackward(out,dout):
    mask = out.ge(0)
    # mask2 = out.lt(0)
    dx = mask*dout
    # dx2=mask2*dout*0.01
    # dx = dx1+dx2
    return dx

def FCLBNforward(x,gamma,beta,bn_param,MODE):
    # mode = bn_param['mode']
    mode = MODE
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum',0.9)
    # x = x.numpy()
    N,D = x.shape
    # print(N,D)
    # print(gamma.shape)
    running_mean = bn_param.get('running_mean', torch.zeros(1, dtype=x.dtype))
    running_var = bn_param.get('running_var', torch.zeros(1, dtype=x.dtype))
    running_mean = running_mean.to(device)
    running_var = running_var.to(device)
    out ,cache = None,None

    if mode == TRAIN:
        # print('trainmode')
        sample_mean = torch.mean(x).to(device)
        # print(sample_mean.shape)
        sample_var = torch.var(x).to(device)
        # print(x.shape)
        out_ = (x - sample_mean) / torch.sqrt(sample_var + eps)
        out_ = out_.to(device)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        # print(out_.shape)

        out = out_*gamma+ beta
        cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)

    elif mode == TEST:
        # print('testmode')
        scale = gamma / torch.sqrt(running_var + eps)
        out = x * scale + (beta - running_mean * scale)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    # out = torch.from_numpy(out)
    return out,cache

def FCLBNbackward(dout,cache):
    dx, dgamma, dbeta = None, None, None

    out_, x, sample_var, sample_mean, eps, gamma, beta = cache
    N = x.shape[0]
    D = x.shape[1]
    dout_ = gamma * dout
    dvar = torch.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5)
    dx_ = 1 / torch.sqrt(sample_var + eps)
    dvar_ = 2 * (x - sample_mean) / (N*D)

    # intermediate for convenient calculation
    di = dout_ * dx_ + dvar * dvar_
    dmean = -1 * torch.sum(di)
    dmean_ = 1 /(N*D)
    channel = dout.shape[0]
    dx = di + dmean * dmean_
    dgamma = torch.Tensor([torch.sum(dout * out_)]).to(device)
    dbeta =  torch.Tensor([torch.sum(dout)]).to(device)
    # print(dx.shape)
    # print(dgamma.shape)
    # print(dbeta.shape)
    return dx, dgamma, dbeta
