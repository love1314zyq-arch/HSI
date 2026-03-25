import torch
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes,bn=True, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc = nn.Sequential(
        #                         nn.Linear(in_planes, in_planes // 8,  bias=False),
        #                         nn.BatchNorm1d(in_planes// 8),
        #                         nn.ReLU(),
        #                         nn.Linear(in_planes // 8, in_planes,bias=False),
        #                        )
        if bn:
            self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                                nn.BatchNorm2d(in_planes// 8),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 8, in_planes, 1, bias=False),
                               )
        else:
            self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                          # nn.BatchNorm2d(in_planes // 8),
                          nn.ReLU(),
                          nn.Conv2d(in_planes // 8, in_planes, 1, bias=False),
                            )
        self.sigmoid = nn.Sigmoid()
        # self.batch1 = nn.BatchNorm2d(in_planes)
        # self.batch2 = nn.BatchNorm2d(in_planes)
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class GlobalAttention(nn.Module):
    def __init__(self, in_planes,bn=True, ratio=16):
        super(GlobalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // 8,  bias=False),
        #                         nn.BatchNorm1d(in_planes// 8),
        #                         nn.ReLU(),
        #                         nn.Linear(in_planes // 8, in_planes,bias=False),
        #                        )
        if bn:
            self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                                nn.BatchNorm2d(in_planes// 8),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 8, in_planes, 1, bias=False),
                               )
        else:
            self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                          # nn.BatchNorm2d(in_planes // 8),
                          nn.ReLU(),
                          nn.Conv2d(in_planes // 8, in_planes, 1, bias=False),
                            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out
class gate_for_old(nn.Module):
    def __init__(self,inplanes ,bn = True):
        super(gate_for_old, self).__init__()
        self.conv = nn.Conv2d(2*inplanes, 4*inplanes, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch = nn.BatchNorm2d(4 * inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.attention = ChannelAttention(4 * inplanes,bn=bn)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.attention(x)
        x1 = torch.flatten(x,1)
        # print("gate-for_old",x1.shape)
        x2 = self.sigmoid(x1.mean(0))
        return x2
class gate_for_new(nn.Module):
    def __init__(self,inplanes,bn = True ):
        super(gate_for_new, self).__init__()
        self.conv = nn.Conv2d(2*inplanes, 4*inplanes, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch = nn.BatchNorm2d(4 * inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        self.attention = GlobalAttention(4 * inplanes,bn=bn)
        # self.linear = nn.Linear(4 * inplanes,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu1(x)
        x = self.attention(x)
        x1 = torch.flatten(x,1)
        # print("x_1",x1.shape)

        x2 = self.sigmoid(x1.mean())
        # print(x2.shape)
        return x2