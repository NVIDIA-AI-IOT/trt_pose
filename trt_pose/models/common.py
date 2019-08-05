import torch


class UpsampleCBR(torch.nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super(UpsampleCBR, self).__init__(
            torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU()
        )
        

class SelectInput(torch.nn.Module):
    
    def __init__(self, index):
        super(SelectInput, self).__init__()
        self.index = index
    
    def forward(self, inputs):
        return inputs[self.index]
    
    
class PoseHead(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels):
        super(PoseHead, self).__init__()
        self.cmap_conv = torch.nn.Conv2d(input_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(input_channels, paf_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        return self.cmap_conv(x), self.paf_conv(x)