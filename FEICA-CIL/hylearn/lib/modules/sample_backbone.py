import torch
import torch.nn as nn
from torch.autograd import Function
class shallow_net(nn.Module):
    def __init__(self,initial_channel):
        super(shallow_net, self).__init__()
        self.initial_channel = initial_channel
        self.inplanes = 32

        self.conv1 = nn.Conv2d(self.initial_channel, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch2 = nn.BatchNorm2d(self.inplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2,stride=2)

        self.conv3 = nn.Conv2d(self.inplanes, 2*self.inplanes, kernel_size=3, stride=1, padding=0, bias=False)
        self.batch3 = nn.BatchNorm2d(2*self.inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = nn.Conv2d(2*self.inplanes, 4*self.inplanes, kernel_size=4, stride=1, padding=0, bias=False)
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

    def forward(self,x,gate,old_value):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x_o = self.pool3(x)
        x = self.conv4(x_o)
        x = self.batch4(x)
        x = self.relu4(x)
        xi1 = self.pool4(x)
        # xi1 = self.conv(x_i)
        xi2 = torch.flatten(xi1, 1)
        if gate is not None:
            # print("previouse shape",x_o.shape)
            gatev = gate(x_o.detach())
            # print("gate shape",gatev.shape)
            # print("xi2",xi2.shape)
            xi2 = gatev*xi2
        if old_value is not None:
            xi2 = xi2+old_value
        xi3 = self.linear(xi2)
        pi4 = self.proj(xi3)
        if  gate is not None:
            return {"raw_features": xi2, "projection": xi3, "prediction": pi4,"intermediate":x_o, "new_gate":gatev }
        else:
            return {"raw_features": xi2, "projection": xi3, "prediction": pi4, "intermediate": x_o}

    def forward_cluster(self, x):
        h = self.conv(x)
        c = torch.flatten(h, 1)
        return c
