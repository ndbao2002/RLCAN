from .utils import MeanShift, Upsampler, default_conv, Mlp

from torch import nn
import torch

from einops import rearrange

from natten import NeighborhoodAttention2D

## Neighbor Attention (CA) Layer
class NALayer(nn.Module):
    def __init__(self, channel, heads=4, window_size=7):
        super(NALayer, self).__init__()

        self.attn = NeighborhoodAttention2D(dim=channel, 
                                            num_heads=heads, 
                                            kernel_size=window_size, 
                                            dilation=1,
                                            rel_pos_bias=True)
        self.mlp = Mlp(in_features=channel, hidden_features=channel*4)
        
        self.gnorm = nn.GroupNorm(1, channel)
        self.lnorm = nn.LayerNorm(channel)


    def forward(self, x):
        N, C, H, W = x.size()
        
        x1 = x
        x1 = rearrange(x1, ('n c h w -> n h w c'))
        
        x = self.gnorm(x)
        x = rearrange(x, ('n c h w -> n h w c'))
        x = self.attn(x) + x1
        x = rearrange(x, ('n h w c -> n (h w) c'))
        x = x + self.mlp(self.lnorm(x)) 
        x = rearrange(x, ('n (h w) c -> n c h w'), h = H)
        return x

## Residual Channel Attention Block (RCAB)
class RNAB(nn.Module):
    def __init__(
        self, conv, n_feat, heads, window_size=7,
        bias=True, gn=True, act=nn.GELU(), res_scale=1.0):

        super(RNAB, self).__init__()
        self.body = nn.Sequential(
            NALayer(channel=n_feat, heads=heads, window_size=window_size),
            conv(n_feat, n_feat, 3, bias=True),
        )
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x)
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualDenseGroup(nn.Module):
    def __init__(self, conv, n_feat, gc, kernel_size, heads, act, res_scale, window_size=7):
        super(ResidualDenseGroup, self).__init__()
        
        self.rnab1 = RNAB(conv, n_feat, heads, window_size=window_size,
                            bias=True, gn=True, act=act, res_scale=res_scale)
        self.adjust1 = conv(n_feat, gc, kernel_size)    
        
        self.rnab2 = RNAB(conv, n_feat + gc, heads, window_size=window_size,
                            bias=True, gn=True, act=act, res_scale=res_scale)
        self.adjust2 = conv(n_feat + gc, gc, kernel_size)    
        
        self.rnab3 = RNAB(conv, n_feat + gc*2, heads, window_size=window_size,
                            bias=True, gn=True, act=act, res_scale=res_scale)
        self.adjust3 = conv(n_feat + gc*2, gc, kernel_size)    
        
        self.rnab4 = RNAB(conv, n_feat + gc*3, heads, window_size=window_size,
                            bias=True, gn=True, act=act, res_scale=res_scale)
        self.adjust4 = conv(n_feat + gc*3, gc, kernel_size)
        
        self.rnab5 = RNAB(conv, n_feat + gc*4, heads, window_size=window_size,
                            bias=True, gn=True, act=act, res_scale=res_scale)
        self.adjust5 = conv(n_feat + gc*4, n_feat, kernel_size)      
    
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.adjust1(self.rnab1(x)))
        x2 = self.lrelu(self.adjust2(self.rnab2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.adjust3(self.rnab3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.adjust4(self.rnab4(torch.cat((x, x1, x2, x3), 1))))
        x5 =            self.adjust5(self.rnab5(torch.cat((x, x1, x2, x3, x4), 1)))
        return x5 * 0.2 + x

## Residual Channel Attention Network (RCAN)
class DNAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(DNAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_feats = args.n_feats
        window_size = args.window_size
        kernel_size = 3
        heads = args.heads
        scale = args.scale
        gc = args.gc
        res_scale = args.res_scale
        act = nn.GELU()

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualDenseGroup(
                conv, n_feats, gc, kernel_size, heads, act=act, res_scale=res_scale, window_size=window_size) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

if __name__ == "__main__":
    pass