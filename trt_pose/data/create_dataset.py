import torch
import argparse
import tqdm
import os
import numpy as np
from trt_pose.plugins import generate_cmap, generate_paf
import subprocess
import PIL.Image
import trt_pose.plugins
import pycocotools.coco
from .dataset import _coco_category_parts, _coco_category_topology_tensor


def _coco_annotations_to_tensors(annotations, image_shape, parts, topology, max_count=100):

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
                peaks[c][counts[c]][0] = (float(y) + 0.5) / IH
                peaks[c][counts[c]][1] = (float(x) + 0.5) / IW
                counts[c] = counts[c] + 1
                visibles[ann_idx][c] = 1
        
        for k in range(K):
            c_a = topology[k][2]
            c_b = topology[k][3]
            if visibles[ann_idx][c_a] and visibles[ann_idx][c_b]:
                connections[k][0][counts[c_a] - 1] = counts[c_b] - 1
                connections[k][1][counts[c_b] - 1] = counts[c_a] - 1
                
    return counts, peaks, connections


def generate_cmap_paf(annotations, image_shape, feature_shape, parts, topology, stdev):
    window = int(5 * stdev)
    counts, peaks, connections = _coco_annotations_to_tensors(annotations, image_shape, parts, topology)
    counts = counts[None, ...]
    peaks = peaks[None, ...]
    connections = connections[None, ...]
    cmap = trt_pose.plugins.generate_cmap(counts, peaks, feature_shape[0], feature_shape[1], stdev, window)
    paf = trt_pose.plugins.generate_paf(connections, topology, counts, peaks, feature_shape[0], feature_shape[1], stdev)
    return cmap.cpu()[0].numpy(), paf.cpu()[0].numpy()


def create_dataset(output_dir, coco_images_dir, coco_annotation_file, category_name, image_shape, feature_shape, stdev):
    
    images_dir = os.path.join(output_dir, 'images')
    cmaps_dir = os.path.join(output_dir, 'cmaps')
    pafs_dir = os.path.join(output_dir, 'pafs')
    
    subprocess.call(['mkdir', '-p', images_dir])
    subprocess.call(['mkdir', '-p', cmaps_dir])
    subprocess.call(['mkdir', '-p', pafs_dir])
    
    coco = pycocotools.coco.COCO(coco_annotation_file)
    cat_id = coco.getCatIds(catNms=category_name)[0]
    image_ids = coco.getImgIds(catIds=cat_id)
    
    parts = _coco_category_parts(coco.cats[cat_id])
    topology = _coco_category_topology_tensor(coco.cats[cat_id])
    
    
    for image_id in tqdm.tqdm(image_ids):
        img = coco.imgs[image_id]
        filename = img['file_name']
        filebase = os.path.splitext(filename)[0]
        
        image_path = os.path.join(images_dir, filebase + '.jpg')
        cmap_path = os.path.join(cmaps_dir, filebase + '.npy')
        paf_path = os.path.join(pafs_dir, filebase + '.npy')
        
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_id)
        annotations = [coco.anns[ann_id] for ann_id in ann_ids]
        
        cmap, paf = generate_cmap_paf(annotations, (img['height'], img['width']), feature_shape, parts, topology, stdev)
        
        np.save(cmap_path, cmap)
        np.save(paf_path, paf)
        
        coco_image_path = os.path.join(coco_images_dir, filename)
        im = PIL.Image.open(coco_image_path).resize((image_shape[1], image_shape[0]))
        im.save(image_path)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Output directory of generated pose dataset')
    parser.add_argument('coco_images_dir', type=str, help='Directory of COCO images')
    parser.add_argument('coco_annotation_file', type=str, help='Path of COCO annotations file')
    parser.add_argument('category_name', type=str, help='Name of COCO category to generate pose dataset for')
    parser.add_argument('image_width', type=int, help='Height of resized images')
    parser.add_argument('image_height', type=int, help='Width of resized images')
    parser.add_argument('feature_width', type=int, help='Height of feature maps (paf/cmap)')
    parser.add_argument('feature_height', type=int, help='Width of feature maps (paf/cmap)')
    parser.add_argument('stdev', type=float, help='Stdev in pixels of CMAP / PAF features')
    args = parser.parse_args()
    
    create_dataset(
        args.output_dir,
        args.coco_images_dir,
        args.coco_annotation_file,
        args.category_name,
        (args.image_height, args.image_width),
        (args.feature_height, args.feature_width),
        args.stdev
    )
    