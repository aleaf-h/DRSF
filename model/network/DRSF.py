import math
from functools import partial

import numpy as np
from timm.models import xception

from model.CBAM2 import CBAMBlock2
from model.FAD import FAD_Head
from model.FFF import FFF
from model.FPN import FPN


from model.common import GuidedAttention, DepthWiseConv2d, SeparableConv2d, Block  # , SeparableConv2d, Block

import torch
import torch.nn as nn
import torch.nn.functional as F



encoder_params = {
    "xception": {
        "features": 2048,
        "init_op": partial(xception, pretrained=True)
    }
}

class DRSF(nn.Module):
    """ End-to-End Reconstruction-Classification Learning for Face Forgery Detection """

    def __init__(self, num_classes=1, drop_rate=0.2):
        super(DRSF, self).__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = encoder_params[self.name]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(encoder_params[self.name]["features"], num_classes)


        # self.ffm1 = FFM(728,728,19)
        # self.ffm2 = FFM(728,256,37)
        # self.ffm3 = FFM(256,128,74)
        # self.ffm4 = FFM(128,64,147)
        self.ffm1 = FPN(728, 728, 4)
        self.ffm2 = FPN(728, 256, 3)
        self.ffm3 = FPN(256, 128, 2)
        self.ffm4 = FPN(128, 64, 1)



        self.cma = FFF(728)


        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)


        self.dct = FAD_Head(299)
        self.cbam = CBAMBlock2(64,2,1)

        # self.decoder1 = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     SeparableConv2d(728, 256, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        # self.decoder2 = Block(256, 256, 3, 1)
        # self.decoder3 = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     SeparableConv2d(256, 128, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.decoder4 = Block(128, 128, 3, 1)
        # self.decoder5 = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     SeparableConv2d(128, 64, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )

        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.decoder7 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        #self.conv_32 = nn.Conv2d(64,64,3)
        self.conv2223 = nn.Conv2d(728,3,1,1)
        # self.conv12_3 = nn.Sequential(
        #     DepthWiseConv2d(3,32),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        # )
        self.conv12_3 = nn.Sequential(
            nn.Conv2d(3, 32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )



    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def forward(self, x):
        # clear the loss inputs
        self.loss_inputs = dict(recons=[],recons2=[], contra=[])
        noise_x = self.add_white_noise(x) if self.training else x
        #fre = self.bam(x)

        fre = self.dct(noise_x)
        fre = self.conv12_3(fre)

        #fre =self.srm(fre)
        #print("ff",fre.shape)



        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        # print("cc",out.shape)

        fre = F.interpolate(fre,size=(149,149),mode='bilinear',align_corners=True)
        # print("cc", fre.shape)
        mixx = torch.cat((fre,out),dim=1)
        mixx = self.cbam(mixx)
        #mixx = self.conv_32(mixx)
        mixx = F.interpolate(mixx, size=(147, 147), mode='bilinear', align_corners=True)



        #print("234", out.shape)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out1 = self.encoder.act2(out)
        mixx = mixx + out1

        out2 = self.encoder.block1(out1)
        out3 = self.encoder.block2(out2)
        out4 = self.encoder.block3(out3)
        embedding = self.encoder.block4(out4)

        # ff = self.conv2223(embedding)
        # self.loss_inputs['recons'].append(embedding)
        #mixx = self.connect(embedding11,embedding)



        ffm1 = self.ffm1(embedding,out4)

        ffm2 = self.ffm2(ffm1, out3)

        ffm3 = self.ffm3(ffm2, out2)

        ffm4 = self.ffm4(ffm3, out1)


        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)
        norm_embed, corr = self.norm_n_corr(ffm2)
        self.loss_inputs['contra'].append(corr)
        norm_embed, corr = self.norm_n_corr(ffm3)
        self.loss_inputs['contra'].append(corr)
        norm_embed, corr = self.norm_n_corr(ffm4)
        self.loss_inputs['contra'].append(corr)
        #
        #
        f1 = self.decoder6(ffm4)


        f2 = self.decoder7(mixx)

        # ff = self.conv2223(embedding)
        recons_x = F.interpolate(f1, size=x.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_inputs['recons'].append(recons_x)
        recons_mix = F.interpolate(f2, size=x.shape[-2:], mode='bilinear',align_corners=True) # 重构后的图 [32, 3, 299, 299]
        self.loss_inputs['recons2'].append(recons_mix)


        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)


        embedding = self.encoder.block7(embedding)


        embedding = self.encoder.block8(embedding)
        img_att = self.attention(x, recons_x, embedding)
        img_att2 = self.attention(x, recons_mix, embedding)
        #
        img_mix = self.cma(embedding,img_att,img_att2)
        # ff = self.conv2223(embedding)
        # recons_x = F.interpolate(ff, size=x.shape[-2:], mode='bilinear', align_corners=True)
        # self.loss_inputs['recons'].append(recons_x)

        embedding = self.encoder.block9(img_mix)
        # embedding = self.encoder.block9(embedding)
        # self.loss_inputs['recons'].append(embedding)

        #embedding = self.encoder.block9(img_att+img_att2)
        #print("eee1", embedding.shape)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)
        #print("eee", embedding.shape)
        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)
        #print("e1", embedding.shape) #torch.Size([16, 2048, 10, 10])
        embedding = self.global_pool(embedding).squeeze()
        #print("e2",embedding.shape) #e2 torch.Size([16, 2048])
        out = self.dropout(embedding)
        #print("out",out.shape)
        return self.fc(out)



