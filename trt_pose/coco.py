import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn
import os
import glob
import PIL.Image
import json
import numpy as np
import tqdm


def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category
    """
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology


def coco_category_to_parts(coco_category):
    """Gets list of parts name from a COCO category
    """
    return coco_category['keypoints']


def coco_annotations_to_tensors(coco_annotations, image_shape, parts, topology, max_count=100):
    """Gets tensors corresponding to peak counts, peak coordinates, and peak to peak connections
    """
    annotations = coco_annotations
    C = len(parts)
    K = topology.shape[0]
    M = max_count
    IH = image_shape[0]
    IW = image_shape[1]
    counts = torch.zeros((C)).int()
    peaks = torch.zeros((C, M, 2)).float()
    visibles = torch.zeros((len(annotations), C)).int()
    connections = -torch.ones((K, 2, M)).int()

    for ann_idx, ann in enumerate(annotations):
        
        kps = ann['keypoints']
        
        # add visible peaks
        for c in range(C):
            
            x = kps[c*3]
            y = kps[c*3+1]
            visible = kps[c*3 + 2]

            if visible:
                peaks[c][counts[c]][0] = (float(y) + 0.5) / (IH + 1.0)
                peaks[c][counts[c]][1] = (float(x) + 0.5) / (IW + 1.0)
                counts[c] = counts[c] + 1
                visibles[ann_idx][c] = 1
        
        for k in range(K):
            c_a = topology[k][2]
            c_b = topology[k][3]
            if visibles[ann_idx][c_a] and visibles[ann_idx][c_b]:
                connections[k][0][counts[c_a] - 1] = counts[c_b] - 1
                connections[k][1][counts[c_b] - 1] = counts[c_a] - 1
                
    return counts, peaks, connections


class CocoDataset(torch.utils.data.Dataset):
    
    def __init__(self, images_dir, annotations_file, category_name, use_crowd=False, min_area=0.0, max_area=1.0, max_part_count=100, cachefile=None):
        
        if cachefile is not None and os.path.exists(cachefile):
            print('Cachefile found.  Loading from cache file...')
            cache = torch.load(cachefile)
            self.counts = cache['counts']
            self.peaks = cache['peaks']
            self.connections = cache['connections']
            self.topology = cache['topology']
            self.parts = cache['parts']
            self.filenames = cache['filenames']
            return
            
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        cat = [c for c in data['categories'] if c['name'] == category_name][0]
        cat_id = cat['id']
        
        img_map = {}
        for img in data['images']:
            img_map[img['id']] = img
            
        # get all images and annotations belonging to each image that meet criteria
        samples = {}
        for ann in data['annotations']:
            
            # filter by category
            if ann['category_id'] != cat_id:
                continue
                
            # filter by crowd
            if not use_crowd and ann['iscrowd']:
                continue
                
            img_id = ann['image_id']
            img = img_map[img_id]
            height = img['height']
            width = img['width']
            area = ann['area']
            
            # filter by object area
            normalized_area = float(area) / float(height * width)
            if normalized_area < min_area or normalized_area > max_area:
                continue
            
            # add metadata
            if img_id not in samples:
                sample = {}
                sample['img'] = img
                sample['anns'] = [ann]
                samples[img_id] = sample
            else:
                samples[img_id]['anns'] += [ann]
        
        # generate tensors
        self.topology = coco_category_to_topology(cat)
        self.parts = coco_category_to_parts(cat)
        
        N = len(samples)
        C = len(self.parts)
        K = self.topology.shape[0]
        M = max_part_count
        
        self.counts = torch.zeros((N, C), dtype=torch.int32)
        self.peaks = torch.zeros((N, C, M, 2), dtype=torch.float32)
        self.connections = torch.zeros((N, K, 2, M), dtype=torch.int32)
        self.filenames = []
        for i, sample in tqdm.tqdm(enumerate(samples.values())):
            self.filenames.append(sample['img']['file_name'])
            image_shape = (sample['img']['height'], sample['img']['width'])
            counts_i, peaks_i, connections_i = coco_annotations_to_tensors(sample['anns'], image_shape, self.parts, self.topology)
            self.counts[i] = counts_i
            self.peaks[i] = peaks_i
            self.connections[i] = connections_i
        
        if cachefile is not None:
            print('Saving to cache file...')
            torch.save({
                'counts': self.counts,
                'peaks': self.peaks,
                'connections': self.connections,
                'topology': self.topology,
                'parts': self.parts,
                'filenames': self.filenames
            }, cachefile)
            