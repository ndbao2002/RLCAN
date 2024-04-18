import torch
from torch import nn

from torch.nn import functional as F

from .utils import MeanShift, ResBlock, RRDB, ResidualGroup
from .SFE import SFE
from .LTE import LTE



class TTSR(nn.Module):
    def __init__(self,
                 lte=None,
                 block_types=['RB_A','RB','RRDB'], # [x1_type, x2_type, x4_type]
                 num_input_blocks=[8,8,1], # [x1_num_blocks, x2_num_blocks, x4_num_blocks]
                 up_type='pixelshuffle', # in: conv, pixelshuffle, pixelshuffle+conv, nearest+conv, bicubic+conv
                 num_sfe_blocks=12,
                 n_feats=192,
                 grow_channels=32,
                 dropout=0.0,
                 res_scale=1.0,
                 rgb_range=1.0):
        super(TTSR, self).__init__()


        assert len(block_types) == len(num_input_blocks) == 3

        self.LTE = LTE(model_cfg=lte)
        self.SFE = SFE(num_res_blocks=num_sfe_blocks, n_feats=n_feats, out_feats=256)

        # Stage x1
        self.x1_input_blocks = self.make_input_block(num_input_blocks=num_input_blocks[0],
                                                block_type=block_types[0],
                                                feats=n_feats,
                                                in_out_feats=512,
                                                dropout=dropout,
                                                res_scale=res_scale,
                                                grow_channels=grow_channels,
                                                scale=1)
        self.ps_12 = self.make_upsample_block(512, up_type=up_type)

        # Stage x2
        self.x2_input_blocks = self.make_input_block(num_input_blocks=num_input_blocks[1],
                                                block_type=block_types[1],
                                                feats=n_feats,
                                                in_out_feats=256,
                                                dropout=dropout,
                                                res_scale=res_scale,
                                                grow_channels=grow_channels,
                                                scale=2**2)
        self.ps_24 = self.make_upsample_block(256, up_type=up_type)

        # Stage x4
        self.x4_input_blocks = self.make_input_block(num_input_blocks=num_input_blocks[2],
                                                block_type=block_types[2],
                                                feats=n_feats,
                                                in_out_feats=128,
                                                dropout=dropout,
                                                res_scale=res_scale,
                                                grow_channels=grow_channels,
                                                scale=2**3)

        self.tail = nn.Conv2d(in_channels=128,
                              out_channels=3,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

    def make_input_block(self, num_input_blocks, block_type, feats, in_out_feats,
                         dropout=0.0, res_scale=1.0, grow_channels=32, scale=1):
        if feats > in_out_feats:
            feats = in_out_feats

        input_blocks = []
        for i in range(num_input_blocks):
            if block_type == 'RB':
                input_blocks.append(ResBlock(feats, feats,
                                            dropout=dropout, attn=False, res_scale=res_scale))
            elif block_type == 'RB_A':
                if self.check_attention_block(num_input_blocks, i):
                    input_blocks.append(ResBlock(feats, feats,
                                                dropout=dropout, attn=True, res_scale=res_scale))
                else:
                    input_blocks.append(ResBlock(feats, feats,
                                                dropout=dropout, attn=False, res_scale=res_scale))
            elif block_type == 'RRDB':
                input_blocks.append(RRDB(feats, grow_channels))
            elif block_type == 'RG':
                input_blocks.append(ResidualGroup(feats, 3, 16, nn.ReLU(True), res_scale, 8, scale=scale))
            else:
                raise NotImplementedError('Input block type is not defined!')

        if feats != in_out_feats:
            input_blocks.insert(0, nn.Conv2d(in_out_feats, feats, kernel_size=3, stride=1, padding=1))
            input_blocks.append(nn.Conv2d(feats, in_out_feats, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*input_blocks)

    def check_attention_block(self, num_input_blocks, block_index):
        if num_input_blocks <= 3:
            if block_index == num_input_blocks // 2:
                return True
        else:
            if block_index == (num_input_blocks // 2) or block_index == (num_input_blocks // 2) - 1:
                return True
        return False

    def make_upsample_block(self, in_channel, up_type='conv',):
        if up_type == 'conv':
            return nn.ConvTranspose2d(in_channel, in_channel//4, kernel_size=2, stride=2)
        elif up_type == 'pixelshuffle':
            return nn.PixelShuffle(2)
        elif up_type == 'pixelshuffle+conv':
            return nn.Sequential([
                nn.PixelShuffle(2),
                nn.Conv2d(in_channel//4, in_channel//4, 3, 1, 1)
            ])
        elif up_type == 'nearest+conv':
            return nn.Sequential([
                nn.Upsample(scale_factor=1/2, mode='nearest'),
                nn.Conv2d(in_channel, in_channel//4, 3, 1, 1)
            ])
        elif up_type == 'bicubic+conv':
            return nn.Sequential([
                nn.Upsample(scale_factor=1/2, mode='bicubic'),
                nn.Conv2d(in_channel, in_channel//4, 3, 1, 1)
            ])
        else:
            raise NotImplementedError('Upsample block type is not defined!')

    def forward(self, x, noise=None):
        # Bicubic then texture extraction (pretrained model)
        x = self.sub_mean(x)
        x_lv4 = F.interpolate(x, scale_factor=4, mode='bicubic')
        t_lv1, t_lv2, t_lv4 = self.LTE(x_lv4)

        # Feature extraxtion
        f_lv1 = self.SFE(x)

        # X1 stage
        f_lv1 = torch.concat([f_lv1, t_lv1], dim=1)
        f_lv1 = self.x1_input_blocks(f_lv1)

        # X1 -> X2
        f_lv2 = self.ps_12(f_lv1)

        # X2 stage
        f_lv2 = torch.concat([f_lv2, t_lv2], dim=1)
        f_lv2 = self.x2_input_blocks(f_lv2)

        # X2 -> X4
        f_lv4 = self.ps_24(f_lv2)

        # X4 stage
        f_lv4 = torch.concat([f_lv4, t_lv4], dim=1)
        f_lv4 = self.x4_input_blocks(f_lv4)

        # Conv to RGB
        final = self.tail(f_lv4)

        final = self.add_mean(final)

        return final
    
if __name__ == '__main__':
    TTSR()