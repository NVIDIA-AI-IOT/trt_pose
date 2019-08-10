import json
import torch


class Task(object):

    def __init__(self, parts=[], links=[], json_file=None, coco_category=None):
        self.parts = parts
        self.links = links

        if json_file is not None:
            self.from_json_file(json_file)
        elif coco_category is not None:
            self.from_coco_category(coco_category)

    def from_json_file(self, file):
        with open(file, 'r') as f:
            t = json.load(f)
            self.parts = t['parts']
            self.links = t['links']

    def from_coco_category(self, coco_category):
        self.parts = coco_category['keypoints']
        self.links = [[l[0] - 1, l[1] - 1] for l in coco_category['skeleton']]

    @property
    def topology_tensor(self):
        K = len(self.links)
        topology_tensor = torch.zeros((K, 4)).int()

        for k in range(K):
            topology_tensor[k][0] = k * 2  # paf i dim
            topology_tensor[k][1] = k * 2 + 1  # paf j dim
            topology_tensor[k][2] = self.links[k][0]  # source part
            topology_tensor[k][3] = self.links[k][1]  # sink part

        return topology_tensor
