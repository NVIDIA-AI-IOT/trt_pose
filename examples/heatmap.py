import torch


def gaussian_window(window_size, stdev):
    w = window_size // 2
    x = torch.linspace(-w, w, window_size).view(1, window_size)
    y = x.view(window_size, 1)
    gaussian = torch.exp(-(x**2 + y**2) / stdev**2)
    return gaussian


def set_window(mat, idx, window, mode='max'):
    HW = window.shape[0]
    hw = HW // 2
    WW = window.shape[1]
    ww = WW // 2
    H = mat.shape[0]
    W = mat.shape[1]
    
    i = idx[0]
    j = idx[1]
    
    i_m = i - hw
    i_p = i + hw + 1
    j_m = j - ww
    j_p = j + ww + 1
    i_min = max(0, i_m)
    i_max = min(H, i_p)
    j_min = max(0, j_m)
    j_max = min(W, j_p)
    i_min_w = i_min - i_m
    i_max_w = HW + i_max - i_p
    j_min_w = j_min - j_m
    j_max_w = WW + j_max - j_p
    
    if i_max <= i_min:
        return mat
    if j_max <= j_min:
        return mat
    
    if mode == 'max':
        mat[i_min:i_max, j_min:j_max] = torch.max(mat[i_min:i_max, j_min:j_max], window[i_min_w:i_max_w, j_min_w:j_max_w])
    else:
        mat[i_min:i_max, j_min:j_max] = window[i_min_w:i_max_w, j_min_w:j_max_w]
    
    return mat


class HeatmapGenerator(torch.nn.Module):
    def __init__(self, window_size, stdev):
        super(HeatmapGenerator, self).__init__()
        self.window_size = window_size
        self.stdev = stdev
        self.window = torch.nn.Parameter(gaussian_window(self.window_size, self.stdev), requires_grad=False)
        
    def fill_2d(self, heatmap, idx, **kwargs):
        set_window(heatmap, idx, self.window, **kwargs)
        return heatmap
    
    def fill_coco_single(self, heatmap, img, ann, min_visibility=1, **kwargs):
        
        height = img['height']
        width = img['width']
        keypoints = ann['keypoints']
        num_keypoints = len(keypoints) // 3

        for i in range(num_keypoints):

            x = keypoints[i*3]
            y = keypoints[i*3 + 1]
            vis = keypoints[i*3 + 2]

            if vis >= min_visibility:
                self.fill_2d(heatmap[i], (y, x), **kwargs)

        return heatmap
    
    def fill_coco_many(self, heatmap, img, anns, min_visibility=1, **kwargs):
        for ann in anns:
            self.fill_coco_single(heatmap, img, ann, min_visibility, **kwargs)
        return heatmap