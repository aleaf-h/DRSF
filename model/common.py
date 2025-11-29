import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#from model.DKG import DKGModule
#from model.ScConv import ScConv


def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False


def l1_regularize(module):
    reg_loss = 0.
    for key, param in module.reg_params.items():
        if "weight" in key and param.requires_grad:
            reg_loss += torch.sum(torch.abs(param))
    return reg_loss


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1,
                 start_with_relu=True, grow_first=True, with_bn=True):
        super(Block, self).__init__()

        self.with_bn = with_bn

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            if with_bn:
                self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            if with_bn:
                rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            if self.with_bn:
                skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=2, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, depth=728, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)
        #self.efm = EFM(728)
        #self.gga = GGA(728,3)
        #self.caa =ChannelAttention(728)
        #self.scc = ScConv(depth)
        self.icb = ICB(depth,depth)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)
        res = embedding*res_map
        #res = self.scc(res)
        res = self.icb(embedding,res)

        return res #res_map2,res_map * self.h(embedding) + self.dropout(embedding)


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.conv2 = nn.Conv2d(in_features, hidden_features, 3, 1, 1)
        # self.conv3 = nn.Conv2d(hidden_features, in_features, 1)
        self.norm1 = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.fc1 = nn.LayerNorm(in_features)
        self.conv11 = nn.Sequential(
            # nn.Conv2d(in_features, hidden_features, (1, 3), padding=(0, 2), groups=1),
            # nn.Conv2d(hidden_features, in_features, (3, 1), padding=(2, 0), groups=1),
            nn.Conv2d(in_features,in_features,1,1),
            nn.SiLU(True),

        )


    def forward(self, x1,x2):
        x1 = x1.permute(0,2,3,1).contiguous()
        x2 = x2.permute(0, 2, 3, 1).contiguous()
        #x = self.norm1(x)
        #xx =x.permute(0, 3, 1, 2)
        xx1 = self.fc1(x1).permute(0, 3, 1, 2).contiguous()
        xx2 = self.fc1(x2).permute(0, 3, 1, 2).contiguous()
        x1 = self.conv11(xx1)
        #print(x1.shape)
        x2 = self.act(xx2)
        #print(x2.shape)
        out = torch.matmul(x1,x2)

        out = self.drop(out)
        # print(out.shape)
        out = x1 + out

        return out
