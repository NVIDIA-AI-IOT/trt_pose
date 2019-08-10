import torch


class PoseTransform(object):

    def __call__(self, ann):
        raise NotImplementedError


class RandomAffine(PoseTransform):
    """Applys a random affine transformation to pose annotation
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, ann):
        pass


class GenerateIntermediateTensors(PoseTransform):

    def __init__(self, task, max_peak_count=100):
        self.task = task
        self.max_peak_count = max_peak_count
        self.topology = task.topology_tensor

    def __call__(self, ann):
        objects = ann['objects']

        IH = ann['height']
        IW = ann['width']

        C = len(self.task.parts)
        K = len(self.task.links)
        M = self.max_peak_count

        counts = torch.zeros((C)).int()
        peaks = torch.zeros((C, M, 2)).float()
        visibles = torch.zeros((len(objects), C)).int()
        connections = -torch.ones((K, 2, M)).int()

        for i, obj in enumerate(objects):
            for c in range(C):

                x = obj[c * 3]
                y = obj[c * 3 + 1]
                visible = obj[c * 3 + 2]

                if visible:
                    peaks[c][counts[c]][0] = (float(y) + 0.5) / (IH + 1.0)
                    peaks[c][counts[c]][1] = (float(x) + 0.5) / (IW + 1.0)
                    counts[c] = counts[c] + 1
                    visibles[i][c] = 1

            for k in range(K):

                c_a = self.topology[k][2]
                c_b = self.topology[k][3]

                if visibles[i][c_a] and visibles[i][c_b]:
                    connections[k][0][counts[c_a] - 1] = counts[c_b] - 1
                    connections[k][1][counts[c_b] - 1] = counts[c_a] - 1

        return counts, peaks, connections



class GenerateCmap(PoseTransform):
    """Generates a part confidence map for pose annotation
    """

    def __init__(self, task, shape, std, max_peak_count=100):
        self.task = task
        self.shape = shape
        self.std = std
        self.max_peak_count = max_peak_count

    def __call__(self, ann):
        objects = ann['objects']
        C = len(self.task.parts)
        K = len(self.task.links)
        M = self.max_peak_count

        peak_counts = torch.zeros((C)).int()
        peaks = torch.zeros((C, M, 2)).float()
        visibles = torch.zeros((len(objects), C)).int()
        connections = - torch.ones((K, 2, M)).int()
        pass


class GeneratePaf(PoseTransform):
    """Generates a part affinity field for pose annotation
    """

    def __init__(self, task, std):
        pass

    def __call__(self, ann):
        pass


class ImageToTensor(PoseTransform):
    """Converts PIL image to torch Tensor
    """

    def __call__(self, ann):
        pass


class NormalizeImage(PoseTransform):
    """Normalizes image (as tensor)

    Must be called after ImageToTensor
    """

    def __init__(self, mean, std):
        pass

    def __call__(self, ann):
        pass
