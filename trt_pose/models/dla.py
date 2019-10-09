import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), 'dla'))
import dla_up


class DlaWrapper(torch.nn.Module):
    def __init__(self, dla_fn, cmap_channels, paf_channels):
        super(DlaWrapper, self).__init__()
        self.backbone = dla_fn(cmap_channels + paf_channels, pretrained_base='imagenet')
        self.cmap_channels = cmap_channels
        self.paf_channels = paf_channels
        
    def forward(self, x):
        x = self.backbone(x)
        cmap, paf = torch.split(x, [self.cmap_channels, self.paf_channels], dim=1)
        return cmap, paf
    
    
def dla34up_pose(cmap_channels, paf_channels):
    return DlaWrapper(dla_up.dla34up, cmap_channels, paf_channels)

def dla60up_pose(cmap_channels, paf_channels):
    return DlaWrapper(dla_up.dla60up, cmap_channels, paf_channels)

def dla102up_pose(cmap_channels, paf_channels):
    return DlaWrapper(dla_up.dla102up, cmap_channels, paf_channels)

def dla169up_pose(cmap_channels, paf_channels):
    return DlaWrapper(dla_up.dla169up, cmap_channels, paf_channels)