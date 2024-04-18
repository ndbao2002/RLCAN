from torch import nn
from torch.nn import functional as F

from .utils import ResBlock

class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, out_feats):
        super(SFE, self).__init__()

        self.num_res_blocks = num_res_blocks

        self.head = nn.Conv2d(3, n_feats, kernel_size=3, stride=1, padding=1)

        RBs_list = []
        for _ in range(self.num_res_blocks):
            RBs_list.append(ResBlock(
                in_ch=n_feats,
                out_ch=n_feats,
                dropout=0.0,
                attn=False,
            ))
        self.RBs = nn.Sequential(*RBs_list)

        if n_feats != out_feats:
            self.shortcut = nn.Conv2d(n_feats, out_feats, kernel_size=3, stride=1, padding=1)
        else:
            self.shortcut = nn.Identity()

        # self.tail = nn.Sequential(
        #     nn.Conv2d(n_feats, out_feats*4, kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=2, padding=1),
        # )
        
        self.tail = nn.Conv2d(n_feats, out_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.head(x))
        x1 = x

        x = self.RBs(x)

        x = self.tail(x)
        x = x + self.shortcut(x1)
        return x
