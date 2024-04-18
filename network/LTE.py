from torch import nn
from torch.nn import functional as F
import torch

from .utils import ResBlock

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x
    
class Custom_Unet(nn.Module):
    def __init__(self,
                 n_feats=64,
                 ch_mul=[1, 2, 4],
                 attention_mul=[],
                 dropout=0.0,
                 channels=3,
                 num_res_blocks=2,):
        super(Custom_Unet, self).__init__()

        self.n_feats = n_feats
        self.dropout = dropout
        self.channels = channels
        self.kernel_size = 3

        # define head module
        self.head = nn.Conv2d(3, n_feats, kernel_size=3, stride=1, padding=1)

        # define downsample module
        channel_list = []
        current_channel = n_feats
        self.downblocks = nn.ModuleList()
        for i, mult in enumerate(ch_mul):
            out_channels = n_feats * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(in_ch=current_channel,
                            out_ch=out_channels,
                            dropout=self.dropout,
                            attn=(mult in attention_mul)))
                current_channel = out_channels
                channel_list.append(current_channel)
            if i != len(ch_mul) - 1:
                out_channels = n_feats * ch_mul[i + 1]
                self.downblocks.append(DownSample(current_channel, out_channels))
                channel_list.append((current_channel, out_channels))
                current_channel = out_channels

        # define middle module
        self.middleblocks = nn.ModuleList([
            ResBlock(in_ch=current_channel, out_ch=current_channel, dropout=self.dropout, attn=True),
            ResBlock(in_ch=current_channel, out_ch=current_channel, dropout=self.dropout, attn=True),
        ])

        # define upsample module
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mul))):
            out_channels = n_feats * mult
            for _ in range(num_res_blocks):
                self.upblocks.append(
                    ResBlock(in_ch=channel_list.pop(),
                            out_ch=out_channels,
                            dropout=self.dropout,
                            attn=(mult in attention_mul)))
            if i != 0:
                curr_ch, out_ch = channel_list.pop()
                self.upblocks.append(UpSample(out_ch, curr_ch))
                self.upblocks.append(ResBlock(in_ch=curr_ch*2,
                                            out_ch=curr_ch,
                                            dropout=self.dropout,
                                            attn=(mult in attention_mul)))

            current_channel = out_channels
        assert len(channel_list) == 0


    def forward(self, x):

        # Downsample
        x = self.head(x)
        x_list = []

        for block in self.downblocks:
            if isinstance(block, DownSample):
                x_list.append(x)
            x = block(x)

        # Middle
        for block in self.middleblocks:
            x = block(x)

        list_x_output = []

        # Upsample
        up = False
        for block in self.upblocks:
            if up:
                x = torch.concat([x_list.pop(), x], dim=1)
                up = False
            if isinstance(block, UpSample):
                up = True
                list_x_output.append(x)
            x = block(x)
        list_x_output.append(x)

        return list_x_output
    
class LTE(nn.Module):
    def __init__(self, model_cfg=None):
        super(LTE, self).__init__()

        ### use vgg19 weights to initialize
        self.model = Custom_Unet(**model_cfg)

    def forward(self, x):
        x_lv1, x_lv2, x_lv4 = self.model(x)
        return x_lv1, x_lv2, x_lv4