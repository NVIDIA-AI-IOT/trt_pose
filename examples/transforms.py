import torchvision.transforms
import PIL.Image
import torch
from heatmap import HeatmapGenerator
import torch.nn.functional as F


class PoseLoadEntry(object):
        
    def __init__(self, device, dtype, heatmap_stdev, heatmap_window_size=None):
        self.heatmap_generator = HeatmapGenerator(heatmap_window_size, heatmap_stdev)
        self.heatmap_generator = self.heatmap_generator.to(device).type(dtype)
        self.device = device
        self.dtype = dtype
        
    def __call__(self, entry):
        # 0-3 img, 3-(3 + C)
        C = len(entry['anns'][0]['keypoints']) // 3
        height = entry['img']['height']
        width = entry['img']['width']
                             
        image = torchvision.transforms.ToTensor()(PIL.Image.open(entry['path'])).to(self.device).type(self.dtype)
        heatmap = torch.zeros((C, height, width), device=self.device, dtype=self.dtype)
        heatmap_pos = torch.zeros((C, height, width), device=self.device, dtype=self.dtype)
        heatmap_neg = torch.zeros((C, height, width), device=self.device, dtype=self.dtype)
        
        img = entry['img']
        anns = entry['anns']
        
        self.heatmap_generator.fill_coco_many(heatmap, img, anns)
        
        if len(anns) > 1:
            perm = torch.randperm(len(anns))
            pos_ann = anns[perm[0]]
            neg_ann = anns[perm[1]]
            self.heatmap_generator.fill_coco_single(heatmap_pos, img, pos_ann)
            self.heatmap_generator.fill_coco_single(heatmap_neg, img, neg_ann)
            
        return (image, heatmap, heatmap_pos, heatmap_neg)
    
    
class PoseResize():
    
    def __init__(self, image_size, heatmap_size):
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        
    def __call__(self, inputs):
        image, heatmap, heatmap_pos, heatmap_neg = inputs[0], inputs[1], inputs[2], inputs[3]
        image_rs = F.interpolate(image[None, ...], size=self.image_size)[0]
        heatmap_rs = F.interpolate(heatmap[None, ...], size=self.heatmap_size)[0]
        heatmap_pos_rs = F.interpolate(heatmap_pos[None, ...], size=self.heatmap_size)[0]
        heatmap_neg_rs = F.interpolate(heatmap_neg[None, ...], size=self.heatmap_size)[0]
        return (image_rs, heatmap_rs, heatmap_pos_rs, heatmap_neg_rs)
    

class PoseNormalizeImage():
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean, std)
        
    def __call__(self, inputs):
        image = self.normalize(inputs[0])
        return (image, inputs[1], inputs[2], inputs[3])