import torch
import torch.nn as nn
from hylearn.lib.modules.quanti import Conv2d


class shallow_net(nn.Module):
    def __init__(self, linear_list,gamma_list,beta_list,cita_list,bn_list, initial_channel):
        super(shallow_net, self).__init__()
        self.initial_channel = initial_channel
        self.inplanes = 32
        self.conv1 = Conv2d(linear_list[0],gamma_list[0],beta_list[0],cita_list[0],bn_list[0],  0,self.initial_channel, self.inplanes, kernel_size=1, stride=1, padding=0,nbit_w=8, nbit_a=32, bias=False)
        self.batch1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 =    Conv2d(linear_list[1],gamma_list[1],beta_list[1],cita_list[1],bn_list[1],  1,self.inplanes, self.inplanes, kernel_size=4, stride=1, padding=0,nbit_w=8, nbit_a=8, bias=False)
        self.batch2 = nn.BatchNorm2d(self.inplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2,stride=2)

        self.conv3 = Conv2d(linear_list[2],gamma_list[2],beta_list[2],cita_list[2],bn_list[2], 2,self.inplanes, 2*self.inplanes, kernel_size=3, stride=1, padding=0, nbit_w=8, nbit_a=8,bias=False)
        self.batch3 = nn.BatchNorm2d(2*self.inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = Conv2d(linear_list[3],gamma_list[3],beta_list[3],cita_list[3],bn_list[3], 3,2*self.inplanes, 4*self.inplanes, kernel_size=4, stride=1, padding=0, nbit_w=8, nbit_a=8,bias=False)
        self.batch4 = nn.BatchNorm2d(4*self.inplanes)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2,stride=2)
        self.linear = nn.Sequential(
            nn.BatchNorm1d(128, affine=False),
            nn.Linear(128, 256),  # self.resnet.rep_dim),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 128),  # self.resnet.rep_dim),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.out_dim = 128

    def forward(self,x,MODE,weight_list,orign_list,inter_list):
        x = self.conv1(x,MODE,weight_list,orign_list,inter_list)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x,MODE,weight_list,orign_list,inter_list)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x,MODE,weight_list,orign_list,inter_list)
        x = self.batch3(x)
        x = self.relu3(x)
        x_o = self.pool3(x)
        x_p = self.conv4(x_o,MODE,weight_list,orign_list,inter_list)
        x_p = self.batch4(x_p)
        x_p = self.relu4(x_p)
        xi1 = self.pool4(x_p)
        # xi1 = self.conv(x_i)
        xi2 = xi1.view(xi1.size(0), -1)
        xi3 = self.linear(xi2)
        pi4 = self.proj(xi3)
        return {"raw_features": xi2, "projection": xi3, "prediction": pi4,"intermdeiate":x_o}

    def forward_cluster(self, x):
        h = self.conv(x)
        c = torch.flatten(h, 1)
        return c
