import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn
import os
import glob
import PIL.Image
import numpy as np


def _coco_category_topology_tensor(coco_category):
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology


def _coco_category_parts(coco_category):
    return coco_category['keypoints']


class CmapPafDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transforms=None):
        self.directory = directory
        self.images_dir = os.path.join(directory, 'images')
        self.cmaps_dir = os.path.join(directory, 'cmaps')
        self.pafs_dir = os.path.join(directory, 'pafs')
        image_paths = glob.glob(os.path.join(self.images_dir, '*.jpg'))
        self.entries = []
        for ip in image_paths:
            base = os.path.splitext(os.path.basename(ip))[0]
            self.entries.append({
                'image': os.path.join(self.images_dir, base + '.jpg'),
                'cmap': os.path.join(self.cmaps_dir, base + '.npy'),
                'paf': os.path.join(self.pafs_dir, base + '.npy')
            })
        self.transforms = transforms
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        cmap = torch.from_numpy(np.load(entry['cmap']))
        paf = torch.from_numpy(np.load(entry['paf']))
        image = PIL.Image.open(entry['image']).convert('RGB')
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, cmap, paf