import torch


class UpsampleCBR(torch.nn.Sequential):
    def __init__(self, input_channels, output_channels, count=1, num_flat=0):
        layers = []
        for i in range(count):
            if i == 0:
                inch = input_channels
            else:
                inch = output_channels
                
            layers += [
                torch.nn.ConvTranspose2d(inch, output_channels, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU()
            ]
            for i in range(num_flat):
                layers += [
                    torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.BatchNorm2d(output_channels),
                    torch.nn.ReLU()
                ]
            
        super(UpsampleCBR, self).__init__(*layers)

        
class SelectInput(torch.nn.Module):
    
    def __init__(self, index):
        super(SelectInput, self).__init__()
        self.index = index
    
    def forward(self, inputs):
        return inputs[self.index]
    
    
class CmapPafHead(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHead, self).__init__()
        if num_upsample > 0:
            self.cmap_conv = torch.nn.Sequential(
                UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat),
                torch.nn.Conv2d(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
            )
            self.paf_conv = torch.nn.Sequential(
                UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat),
                torch.nn.Conv2d(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.cmap_conv = torch.nn.Conv2d(input_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
            self.paf_conv = torch.nn.Conv2d(input_channels, paf_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        return self.cmap_conv(x), self.paf_conv(x)
    
    
class CmapPafHeadAttention(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHeadAttention, self).__init__()
        self.cmap_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.paf_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.cmap_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
        self.paf_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
            
        self.cmap_conv = torch.nn.Conv2d(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        xc = self.cmap_up(x)
        ac = torch.sigmoid(self.cmap_att(xc))
        
        xp = self.paf_up(x)
        ap = torch.tanh(self.paf_att(xp))
        
        return self.cmap_conv(xc * ac), self.paf_conv(xp * ap)
    
    
