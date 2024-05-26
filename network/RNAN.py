from .utils import MeanShift, Upsampler, default_conv, Mlp

from torch import nn

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
    
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=18):
        super(CALayer, self).__init__()
        
        # global average pooling: feature --> point
        # feature channel downscale and upscale --> channel weight
        self.conv_global = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_global(x)
        return x * y

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, heads, act, reduction, n_resblocks, window_size=7):
        super(ResidualGroup, self).__init__()
        modules_body = [
            NALayer(channel=n_feat, heads=heads, window_size=window_size) \
            for i in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        
        self.conv = nn.Sequential(            
            conv(n_feat, n_feat//4, 1, bias=True),
            act if act else nn.Identity(),
            conv(n_feat//4, n_feat, 1, bias=True),
            CALayer(n_feat, reduction)
        )

    def forward(self, x):
        res = self.body(x)
        res = self.conv(res)
        return res + x

## Residual Channel Attention Network (RCAN)
class RNAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RNAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        window_size = args.window_size
        kernel_size = 3
        heads = args.heads
        scale = args.scale
        reduction = args.reduction
        act = nn.GELU()

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, heads, act=act, reduction=reduction, n_resblocks=n_resblocks, window_size=window_size) \
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