from .utils import MeanShift, Upsampler, default_conv, Mlp

from torch import nn

from einops import rearrange

from natten import NeighborhoodAttention2D
    
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
    
## Neighbor Channel Attention (NCA) Block
class NCABlock(nn.Module):
    def __init__(self, conv, n_feat, heads, window_size, act=nn.GELU(), reduction=18):
        super(NCABlock, self).__init__()
        
        self.conv = nn.Sequential(            
            conv(n_feat, n_feat//4, 1, bias=True),
            act if act else nn.Identity(),
            conv(n_feat//4, n_feat, 1, bias=True),
            CALayer(n_feat, reduction)
        )
        
        self.mlp = Mlp(in_features=n_feat, hidden_features=n_feat*4)
        
        self.attn = NeighborhoodAttention2D(dim=n_feat, 
                                            num_heads=heads, 
                                            kernel_size=window_size, 
                                            dilation=1,
                                            rel_pos_bias=True)
        
        self.norm1 = nn.LayerNorm(n_feat)
        self.norm2 = nn.LayerNorm(n_feat)

    def forward(self, x):
        # x: N, C, H, W
        N, C, H, W = x.size()
        
        x = rearrange(x, 'n c h w -> n (h w) c')
        shortcut = x
        
        x = self.norm1(x)
        
        ca_x = self.conv(rearrange(x, 'n (h w) c -> n c h w', h = H))
        ca_x = rearrange(ca_x, 'n c h w -> n (h w) c')
        
        na_x = self.attn(rearrange(x, 'n (h w) c -> n h w c', h = H))
        na_x = rearrange(na_x, 'n h w c -> n (h w) c')
        
        x = shortcut + na_x + ca_x * 0.02
        x = x + self.mlp(self.norm2(x))
        
        x = rearrange(x, 'n (h w) c -> n c h w', h = H)
        
        return x
    
class NABlock(nn.Module):
    def __init__(self, n_feat, heads, window_size):
        super(NABlock, self).__init__()
        
        self.mlp = Mlp(in_features=n_feat, hidden_features=n_feat*4)
        
        self.attn = NeighborhoodAttention2D(dim=n_feat, 
                                            num_heads=heads, 
                                            kernel_size=window_size, 
                                            dilation=1,
                                            rel_pos_bias=True)
        
        self.norm1 = nn.LayerNorm(n_feat)
        self.norm2 = nn.LayerNorm(n_feat)

    def forward(self, x):
        # x: N, C, H, W
        N, C, H, W = x.size()
        
        x = rearrange(x, 'n c h w -> n (h w) c')
        shortcut = x
        
        x = self.norm1(x)
        
        na_x = self.attn(rearrange(x, 'n (h w) c -> n h w c', h = H))
        na_x = rearrange(na_x, 'n h w c -> n (h w) c')
        
        x = shortcut + na_x
        x = x + self.mlp(self.norm2(x))
        
        x = rearrange(x, 'n (h w) c -> n c h w', h = H)
        
        return x

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, heads, act, reduction, n_resblocks, window_size=15, last_window_size=31):
        super(ResidualGroup, self).__init__()
        modules_body = [
            NCABlock(conv=conv, n_feat=n_feat, heads=heads, window_size=window_size, act=act, reduction=reduction) \
            for i in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        
        self.tail = NABlock(n_feat=n_feat, heads=heads, window_size=last_window_size)
        
        self.conv = conv(n_feat, n_feat, kernel_size)

    def forward(self, x):
        res = x + self.body(x)
        res = res + self.conv(self.tail(res))
        return res

## Residual Neighbor Channel Attention Network (RNCAN)
class RNCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RNCAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        window_size = args.window_size
        last_window_size = args.last_window_size
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
                conv, n_feats, kernel_size, heads, act=act, reduction=reduction, n_resblocks=n_resblocks, window_size=window_size, last_window_size=last_window_size) \
            for _ in range(n_resgroups)]
        
        self.norm = nn.LayerNorm(n_feats)
        self.conv_after_body = conv(n_feats, n_feats, 3)

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            Upsampler(conv, scale, 64, act=False),
            conv(64, args.n_colors, kernel_size)]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        
    def normalize(self, x):
        N, C, H, W = x.size()
        
        x = rearrange(x, 'n c h w -> n (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'n (h w) c -> n c h w', h = H)
        
        return x

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        x = self.conv_after_body(self.normalize(self.body(x))) + x
        
        x = self.tail(x)
        x = self.add_mean(x)

        return x

if __name__ == "__main__":
    pass