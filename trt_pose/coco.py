import torch
import torch.utils.data
import torch.nn
import os
import PIL.Image
import json
import tqdm
import trt_pose
import trt_pose.plugins
import glob
import torchvision.transforms.functional as FT
import numpy as np
from trt_pose.parse_objects import ParseObjects
import pycocotools
import pycocotools.coco
import pycocotools.cocoeval
import torchvision


        
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


def coco_annotations_to_tensors(coco_annotations,
                                image_shape,
                                parts,
                                topology,
                                max_count=100):
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

            x = kps[c * 3]
            y = kps[c * 3 + 1]
            visible = kps[c * 3 + 2]

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


def coco_annotations_to_mask_bbox(coco_annotations, image_shape):
    mask = np.ones(image_shape, dtype=np.uint8)
    for ann in coco_annotations:
        if 'num_keypoints' not in ann or ann['num_keypoints'] == 0:
            bbox = ann['bbox']
            x0 = round(bbox[0])
            y0 = round(bbox[1])
            x1 = round(x0 + bbox[2])
            y1 = round(y0 + bbox[3])
            mask[y0:y1, x0:x1] = 0
    return mask
            

def convert_dir_to_bmp(output_dir, input_dir):
    files = glob.glob(os.path.join(input_dir, '*.jpg'))
    for f in files:
        new_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(f))[0] + '.bmp')
        img = PIL.Image.open(f)
        img.save(new_path)

        
def get_quad(angle, translation, scale, aspect_ratio=1.0):
    if aspect_ratio > 1.0:
        # width > height =>
        # increase height region
        quad = np.array([
            [0.0, 0.5 - 0.5 * aspect_ratio],
            [0.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 - 0.5 * aspect_ratio],
            
        ])
    elif aspect_ratio < 1.0:
        # width < height
        quad = np.array([
            [0.5 - 0.5 / aspect_ratio, 0.0],
            [0.5 - 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 0.0],
            
        ])
    else:
        quad = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        
    quad -= 0.5

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    quad = np.dot(quad, R)

    quad -= np.array(translation)
    quad /= scale
    quad += 0.5
    
    return quad


def transform_image(image, size, quad):
    new_quad = np.zeros_like(quad)
    new_quad[:, 0] = quad[:, 0] * image.size[0]
    new_quad[:, 1] = quad[:, 1] * image.size[1]
    
    new_quad = (new_quad[0][0], new_quad[0][1],
            new_quad[1][0], new_quad[1][1],
            new_quad[2][0], new_quad[2][1],
            new_quad[3][0], new_quad[3][1])
    
    return image.transform(size, PIL.Image.QUAD, new_quad)


def transform_points_xy(points, quad):
    p00 = quad[0]
    p01 = quad[1] - p00
    p10 = quad[3] - p00
    p01 /= np.sum(p01**2)
    p10 /= np.sum(p10**2)
    
    A = np.array([
        p10,
        p01,
    ]).transpose()
    
    return np.dot(points - p00, A)


def transform_peaks(counts, peaks, quad):
    newpeaks = peaks.clone().numpy()
    C = counts.shape[0]
    for c in range(C):
        count = int(counts[c])
        newpeaks[c][0:count] = transform_points_xy(newpeaks[c][0:count][:, ::-1], quad)[:, ::-1]
    return torch.from_numpy(newpeaks)


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_dir,
                 annotations_file,
                 category_name,
                 image_shape,
                 target_shape,
                 is_bmp=False,
                 stdev=0.02,
                 use_crowd=False,
                 min_area=0.0,
                 max_area=1.0,
                 max_part_count=100,
                 random_angle=(0.0, 0.0),
                 random_scale=(1.0, 1.0),
                 random_translate=(0.0, 0.0),
                 transforms=None,
                 keep_aspect_ratio=False):

        self.keep_aspect_ratio = keep_aspect_ratio
        self.transforms=transforms
        self.is_bmp = is_bmp
        self.images_dir = images_dir
        self.image_shape = image_shape
        self.target_shape = target_shape
        self.stdev = stdev
        self.random_angle = random_angle
        self.random_scale = random_scale
        self.random_translate = random_translate
        
        tensor_cache_file = annotations_file + '.cache'
        
        if tensor_cache_file is not None and os.path.exists(tensor_cache_file):
            print('Cachefile found.  Loading from cache file...')
            cache = torch.load(tensor_cache_file)
            self.counts = cache['counts']
            self.peaks = cache['peaks']
            self.connections = cache['connections']
            self.topology = cache['topology']
            self.parts = cache['parts']
            self.filenames = cache['filenames']
            self.samples = cache['samples']
            return
            
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        cat = [c for c in data['categories'] if c['name'] == category_name][0]
        cat_id = cat['id']

        img_map = {}
        for img in data['images']:
            img_map[img['id']] = img

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

        print('Generating intermediate tensors...')
        self.counts = torch.zeros((N, C), dtype=torch.int32)
        self.peaks = torch.zeros((N, C, M, 2), dtype=torch.float32)
        self.connections = torch.zeros((N, K, 2, M), dtype=torch.int32)
        self.filenames = []
        self.samples = []
        
        for i, sample in tqdm.tqdm(enumerate(samples.values())):
            filename = sample['img']['file_name']
            self.filenames.append(filename)
            image_shape = (sample['img']['height'], sample['img']['width'])
            counts_i, peaks_i, connections_i = coco_annotations_to_tensors(
                sample['anns'], image_shape, self.parts, self.topology)
            self.counts[i] = counts_i
            self.peaks[i] = peaks_i
            self.connections[i] = connections_i
            self.samples += [sample]

        if tensor_cache_file is not None:
            print('Saving to intermediate tensors to cache file...')
            torch.save({
                'counts': self.counts,
                'peaks': self.peaks,
                'connections': self.connections,
                'topology': self.topology,
                'parts': self.parts,
                'filenames': self.filenames,
                'samples': self.samples
            }, tensor_cache_file)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if self.is_bmp:
            filename = os.path.splitext(self.filenames[idx])[0] + '.bmp'
        else:
            filename = os.path.splitext(self.filenames[idx])[0] + '.jpg'

        image = PIL.Image.open(os.path.join(self.images_dir, filename))
        
        im = self.samples[idx]['img']
        
        mask = coco_annotations_to_mask_bbox(self.samples[idx]['anns'], (im['height'], im['width']))
        mask = PIL.Image.fromarray(mask)
        
        counts = self.counts[idx]
        peaks = self.peaks[idx]
        
        # affine transformation
        shiftx = float(torch.rand(1)) * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
        shifty = float(torch.rand(1)) * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
        scale = float(torch.rand(1)) * (self.random_scale[1] - self.random_scale[0]) + self.random_scale[0]
        angle = float(torch.rand(1)) * (self.random_angle[1] - self.random_angle[0]) + self.random_angle[0]
        
        if self.keep_aspect_ratio:
            ar = float(image.width) / float(image.height)
            quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=ar)
        else:
            quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=1.0)
        
        image = transform_image(image, (self.image_shape[1], self.image_shape[0]), quad)
        mask = transform_image(mask, (self.target_shape[1], self.target_shape[0]), quad)
        peaks = transform_peaks(counts, peaks, quad)
        
        counts = counts[None, ...]
        peaks = peaks[None, ...]

        stdev = float(self.stdev * self.target_shape[0])

        cmap = trt_pose.plugins.generate_cmap(counts, peaks,
            self.target_shape[0], self.target_shape[1], stdev, int(stdev * 5))

        paf = trt_pose.plugins.generate_paf(
            self.connections[idx][None, ...], self.topology,
            counts, peaks,
            self.target_shape[0], self.target_shape[1], stdev)

        image = image.convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, cmap[0], paf[0], torch.from_numpy(np.array(mask))[None, ...]

    def get_part_type_counts(self):
        return torch.sum(self.counts, dim=0)
    
    def get_paf_type_counts(self):
        c = torch.sum(self.connections[:, :, 0, :] >= 0, dim=-1) # sum over parts
        c = torch.sum(c, dim=0) # sum over batch
        return c
    
    
class CocoHumanPoseEval(object):
    
    def __init__(self, images_dir, annotation_file, image_shape, keep_aspect_ratio=False):
        
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.image_shape = tuple(image_shape)
        self.keep_aspect_ratio = keep_aspect_ratio
        
        self.cocoGt = pycocotools.coco.COCO('annotations/person_keypoints_val2017.json')
        self.catIds = self.cocoGt.getCatIds('person')
        self.imgIds = self.cocoGt.getImgIds(catIds=self.catIds)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def evaluate(self, model, topology):
        self.parse_objects = ParseObjects(topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        
        results = []

        for n, imgId in enumerate(self.imgIds[1:]):

            # read image
            img = self.cocoGt.imgs[imgId]
            img_path = os.path.join(self.images_dir, img['file_name'])

            image = PIL.Image.open(img_path).convert('RGB')#.resize(IMAGE_SHAPE)
            
            if self.keep_aspect_ratio:
                ar = float(image.width) / float(image.height)
            else:
                ar = 1.0
                
            quad = get_quad(0.0, (0, 0), 1.0, aspect_ratio=ar)
            image = transform_image(image, self.image_shape, quad)

            data = self.transform(image).cuda()[None, ...]

            cmap, paf = model(data)
            cmap, paf = cmap.cpu(), paf.cpu()

        #     object_counts, objects, peaks, int_peaks = postprocess(cmap, paf, cmap_threshold=0.05, link_threshold=0.01, window=5)
        #     object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

            object_counts, objects, peaks = self.parse_objects(cmap, paf)
            object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

            for i in range(object_counts):
                object = objects[i]
                score = 0.0
                kps = [0]*(17*3)
                x_mean = 0
                y_mean = 0
                cnt = 0
                for j in range(17):
                    k = object[j]
                    if k >= 0:
                        peak = peaks[j][k]
                        if ar > 1.0: # w > h w/h
                            x = peak[1]
                            y = (peak[0] - 0.5) * ar + 0.5
                        else:
                            x = (peak[1] - 0.5) / ar + 0.5
                            y = peak[0]

                        x = round(float(img['width'] * x))
                        y = round(float(img['height'] * y))

                        score += 1.0
                        kps[j * 3 + 0] = x
                        kps[j * 3 + 1] = y
                        kps[j * 3 + 2] = 2
                        x_mean += x
                        y_mean += y
                        cnt += 1

                ann = {
                    'image_id': imgId,
                    'category_id': 1,
                    'keypoints': kps,
                    'score': score / 17.0
                }
                results.append(ann)
            if n % 100 == 0:
                print('%d / %d' % (n, len(self.imgIds)))


        if len(results) == 0:
            return
        
        with open('trt_pose_results.json', 'w') as f:
            json.dump(results, f)
            
        cocoDt = self.cocoGt.loadRes('trt_pose_results.json')
        cocoEval = pycocotools.cocoeval.COCOeval(self.cocoGt, cocoDt, 'keypoints')
        cocoEval.params.imgIds = self.imgIds
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()