# from functools import partial
#
# import torch
# import torch.nn as nn
# from timm.layers import DropPath
# import torch.nn.functional as F
# from model.FFF import FFF
#
#
# class FFM(nn.Module):
#     def __init__(self, in_channels, out_channels, xsize):
#         super(FFM, self).__init__()
#         self.xsize = xsize
#         # 1x1 卷积层
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels,  (3, 3), padding=1)
#         #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
#
#         self.upsample = nn.Upsample(size=(xsize, xsize), mode='bilinear', align_corners=True)
#         self.up = nn.UpsamplingNearest2d(scale_factor=2)
#         # 转置卷积层
#         self.transpose_conv = nn.ConvTranspose2d(
#                 in_channels=out_channels,
#                 out_channels=out_channels,
#                 kernel_size=(4, 4),
#                 stride=2,
#                 padding=1,
#                 output_padding=0,
#                 bias=False)
#         #self.transpose_conv = nn.Conv2d(out_channels, out_channels, 1)
#
#         self.conv = nn.Conv2d(out_channels*2, out_channels,kernel_size=1)
#
#         # self.rgcn = My_GCNModel(nfeat=out_channels, nclass=out_channels, dropout=0.2, nhidlayer=1, nhid=64, tensor_size=(2*xsize,2*xsize),
#         #                         patch_size=(2, 2))
#         self.fff = FFF(out_channels)
#
#         # self.norm11 = nn.BatchNorm2d(out_channels)
#         # self.norm1 = nn.BatchNorm2d(in_channels)
#         self.sss = nn.ReLU(inplace=True)
#
#         self.act = nn.Sequential(
#             nn.BatchNorm2d(out_channels),
#
#         )
#
#         self.drop = DropPath(0.2)
#
#     def forward(self, x, front):
#         #x1 = self.norm1(x)
#         # 1x1 卷积操作
#         out_conv1x1 = self.conv1x1(x)
#
#         # 转置卷积操作
#         x_upsampled = self.up(out_conv1x1)
#
#         out_transpose = self.transpose_conv(x_upsampled)
#
#         out_transpose = F.interpolate(out_transpose, size=front.shape[-2:],
#                                       mode='bilinear', align_corners=True)
#
#         #print("oo1",out_transpose.shape)
#         #
#
#
#
#         # front = self.norm11(front)
#
#
#
#         # front = self.act(front)
#
#
#         # 进行 1-sigmoid 操作
#         out_sigmoid = self.sss(out_transpose)
#
#         #out_sigmoid = torch.nn.functional.softmax(out_transpose, dim=1)[:, 1, :, :].unsqueeze(1)
#         # print("oo2", out_sigmoid.shape)
#         #front1 = F.interpolate(front, size=[2 * self.xsize, 2 * self.xsize], mode='bilinear', align_corners=True)
#         # print("fff",front.shape)
#         #out = self.rgcn(front1, out_sigmoid)
#         #print("out1",out.shape)
#
#
#         # 点乘操作
#         out_dot_product = front * out_sigmoid    #torch.mul(front, out_sigmoid)
#         #out_dot_product = self.act(out_dot_product)
#         #print("out2", out_dot_product.shape)
#
#         out_dot_product = self.drop(out_dot_product)
#         #out = self.fff(out_transpose, out_dot_product)
#         out = out_transpose + out_dot_product
#         #out = self.drop(out)
#
#         # 连接操作
#         #out_concat = torch.cat((out_transpose, out_dot_product), dim=1)
#         # print("t",out_transpose.shape)
#         # print("d",out_dot_product.shape)
#         #out = self.rgcn(out_dot_product,out_transpose)
#         #out = self.conv(out_concat)
#         return out
#
#
# # class FFM(nn.Module):
# #     def __init__(self, in_channels, out_channels, xsize,
# #                  norm_layer=partial(nn.LayerNorm,eps=1e-6),expension_ratio=8/3,
# #                  act_layer=nn.GELU,
# #                  drop_path=0.,
# #                  ):
# #         super(FFM, self).__init__()
# #         self.xsize = xsize
# #         # 1x1 卷积层
# #         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
# #
# #         self.upsample = nn.Upsample(size=(xsize, xsize), mode='bilinear', align_corners=True)
# #         # 转置卷积层
# #         self.transpose_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1)
# #
# #         self.conv = nn.Conv2d(out_channels*2, out_channels,kernel_size=1)
# #
# #         # self.rgcn = My_GCNModel(nfeat=out_channels, nclass=out_channels, dropout=0.2, nhidlayer=1, nhid=64, tensor_size=(2*xsize,2*xsize),
# #         #                         patch_size=(2, 2))
# #         self.fff = FFF(out_channels)
# #
# #         #self.norm = norm_layer(in_channels)
# #         self.norm1 = nn.BatchNorm2d(in_channels)
# #         self.norm2 = nn.BatchNorm2d(out_channels)
# #         hidden = int(expension_ratio * in_channels)
# #         self.fc1 = nn.Linear(in_channels, hidden)
# #         self.fc2 = nn.Linear(out_channels, out_channels)
# #         self.fc3 = nn.Linear(out_channels, out_channels)
# #         self.drop_path = DropPath(drop_path)
# #
# #         self.act = act_layer()
# #
# #     def forward(self, x, front):
# #         start = x
# #         x = self.norm1(x)
# #         x = self.fc1(x)
# #         x = self.conv1x1(x)
# #
# #         front = self.norm2(front)
# #         front = self.fc2(front)
# #         front = self.act(front)
# #
# #         mix = x*front
# #         mix = self.fc3(mix)
# #         mix = self.drop_path(mix)
# #
# #         out = start+mix
# #
# #
# #
# #
# #
# #         # 1x1 卷积操作
# #         out_conv1x1 = self.conv1x1(x)
# #
# #         # 转置卷积操作
# #         x_upsampled = self.upsample(out_conv1x1)
# #         out_transpose = self.transpose_conv(x_upsampled)
# #
# #
# #         #print("oo1",out_transpose.shape)
# #
# #         # 进行 1-sigmoid 操作
# #         out_sigmoid = 1 - torch.sigmoid(out_transpose)
# #         #out_sigmoid = torch.nn.functional.softmax(out_transpose, dim=1)[:, 1, :, :].unsqueeze(1)
# #         # print("oo2", out_sigmoid.shape)
# #         #front1 = F.interpolate(front, size=[2 * self.xsize, 2 * self.xsize], mode='bilinear', align_corners=True)
# #         # print("fff",front.shape)
# #         #out = self.rgcn(front1, out_sigmoid)
# #         #print("out1",out.shape)
# #         # 点乘操作
# #         out_dot_product = torch.mul(front, out_sigmoid)
# #         #print("out2", out_dot_product.shape)
# #
# #         #out = self.fff(out_transpose, out_dot_product)
# #         out = out_transpose + out_dot_product
# #         # 连接操作
# #         #out_concat = torch.cat((out_transpose, out_dot_product), dim=1)
# #         # print("t",out_transpose.shape)
# #         # print("d",out_dot_product.shape)
# #         #out = self.rgcn(out_dot_product,out_transpose)
# #         #out = self.conv(out_concat)
# #         return out
# #
#
import torch
import torch.nn as nn


class FFM(nn.Module):
    def __init__(self, in_channels, out_channels, xsize,droppath=0.):
        super(FFM, self).__init__()
        self.xsize = xsize
        # 1x1 卷积层
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.upsample = nn.Upsample(size=(xsize, xsize), mode='bilinear', align_corners=True)
        # 转置卷积层
        self.transpose_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1)

        self.conv = nn.Conv2d(out_channels*2, out_channels,kernel_size=1)

        # self.rgcn = My_GCNModel(nfeat=out_channels, nclass=out_channels, dropout=0.2, nhidlayer=1, nhid=64, tensor_size=(2*xsize,2*xsize),
        #                         patch_size=(2, 2))
        #self.fff = FFF(out_channels)
        # self.ww = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels,kernel_size=3),
        #     nn.GELU()
        # )
        # self.norm1 = nn.BatchNorm2d(out_channels)
        # self.norm2 = nn.BatchNorm2d(in_channels)
        #
        # self.linear1 = nn.Linear(out_channels*xsize*xsize,out_channels)
        #self.act = nn.GELU()
        # self.drop = DropPath(droppath)


    def forward(self, x, front):
        # strat =front
        # front = self.ww(front)
        # front = F.interpolate(front, size=strat.shape[-2:], mode='bilinear', align_corners=True)
        # 1x1 卷积操作
        out_conv1x1 = self.conv1x1(x)

        # 转置卷积操作
        x_upsampled = self.upsample(out_conv1x1)
        out_transpose = self.transpose_conv(x_upsampled)

        #print("oo1",out_transpose.shape)

        # 进行 1-sigmoid 操作
        out_sigmoid = 1 - torch.sigmoid(out_transpose)
        #out_sigmoid = torch.nn.functional.softmax(out_transpose, dim=1)[:, 1, :, :].unsqueeze(1)
        # print("oo2", out_sigmoid.shape)
        #front1 = F.interpolate(front, size=[2 * self.xsize, 2 * self.xsize], mode='bilinear', align_corners=True)
        # print("fff",front.shape)
        #out = self.rgcn(front1, out_sigmoid)
        #print("out1",out.shape)

        # 点乘操作
        out_dot_product = torch.mul(out_sigmoid, front)
        #out_drop = self.drop(out_dot_product)

        out = out_transpose+ out_dot_product


        #print("111",out1.shape)
        # 连接操作
        #out_concat = torch.cat((out_transpose, out_dot_product), dim=1)
        # print("t",out_transpose.shape)
        # print("d",out_dot_product.shape)
        #out = self.rgcn(out_dot_product,out_transpose)
        #out = self.conv(out_concat)

        return out


