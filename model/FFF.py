import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        query.size(-1)
    )
    p_attn = F.softmax(scores, dim=-1)
    p_val = torch.matmul(p_attn, value)
    return p_val, p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model):
        super().__init__()

        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x ,y, z):
        b, c, h, w = x.size()

        _query = self.query_embedding(x)
        _key = self.key_embedding(y)
        _value = self.value_embedding(z)


        y, _ = attention(_query, _key, _value)
        self_attention = self.output_linear(y)

        return self_attention


class FFF(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, in_channel=256):
        super().__init__()
        self.attention = MultiHeadedAttention( d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel
        )

    def forward(self, rgb,fre,rec):
        self_attention = self.attention(rgb,fre,rec)
        output = rgb + self_attention
        #output = output + self.feed_forward(output)
        return output

class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

if __name__ == '__main__':
    fre = torch.randn(32, 4, 16, 16)
    embedding = torch.randn(32, 4, 16, 16)
    fff = FFF(4)

    fre_mix = fff(embedding,fre)
    #print("fre_mix", fre_mix.shape)

