import torch
from torch import nn
import torch.nn.functional as F
BN_MOMENTUM = 0.1
class FPN(nn.Module):

    def __init__(self,in_channels, out_channels,i):
        super(FPN,self).__init__()

        self.sigmoid_layer = nn.Sigmoid()
        kernel, padding, output_padding = self._get_deconv_cfg(i)
        # Adding pointwise block
        self.pw_block_1 = self._point_wise_block(in_channels, in_channels)

        self.deconv_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=i,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))

        self.conv234 = self._point_wise_block(out_channels*2, out_channels)


    def forward(self, x,front):
        x = self.pw_block_1(x)  # B x 1024 x 8 x 8
        x = self.deconv_layer(x)  # B x 256 x 16 x 16
        # x = self.relu(x) # B x 256 x 16 x 16

        if x.size(3)!=front.size(3):
            x = F.interpolate(x, size=front.shape[-2:], mode='bilinear', align_corners=True)

        x_weighted = self.sigmoid_layer(x)  # B x 256 x 16 x 16
        x_inverse = torch.sub(1, x_weighted, alpha=1)  # B x 256 x 16 x 16
        #x3 = self.pw_block_c3(front)  # B x 256 x 16 x 16

        x3_ = torch.multiply(front, x_inverse)  # B x 256 x 16 x 16
        # print(x.shape)
        # print(x3_.shape)
        x = torch.cat((x, x3_), dim=1)  # B x 512 x 16 x 16
        x = self.conv234(x)

        return x



    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 1:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    def _point_wise_block(self, inplanes, outplanes):
        self.inplanes = outplanes
        module = point_wise_block(inplanes, outplanes)
        return module

def point_wise_block(inplanes, outplanes):
    return nn.Sequential(
        nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
    )

if __name__ == '__main__':
    rgb = torch.randn(2, 728, 19, 19)
    fre = torch.randn(2, 256, 37, 37)

    cma = FPN(728, 256,3)
    output = cma(rgb,fre)
    print(output.shape)