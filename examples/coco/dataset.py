from topology import PARTS, TOPOLOGY, COCO_PARTS


def convert_annotations(input_shape, annotations, topology):
    C = len(PARTS)
    K = topology.shape[0]
    M = 100
    IH = input_shape[0]
    IW = input_shape[1]
    counts = torch.zeros((C)).int()
    peaks = torch.zeros((C, M, 2)).float()
    visibles = torch.zeros((len(annotations), C)).int()
    connections = -torch.ones((K, 2, M)).int()
    
    for ann_idx, ann in enumerate(annotations):
        
        kps = ann['keypoints']
        
        # add visible peaks
        for c in range(C):
            
            if PARTS[c] == 'NECK':
                c_coco_l = COCO_PARTS.index('LSHOULDER')
                x_l = kps[c_coco_l*3]
                y_l = kps[c_coco_l*3+1]
                vis_l = kps[c_coco_l*3+2]
                c_coco_r = COCO_PARTS.index('RSHOULDER')
                x_r = kps[c_coco_r*3]
                y_r = kps[c_coco_r*3+1]
                vis_r = kps[c_coco_r*3+2]
                visible = vis_l and vis_r
                x = (x_l + x_r) / 2.0
                y = (y_l + y_r) / 2.0
            else:
                c_coco = COCO_PARTS.index(PARTS[c])
                x = kps[c_coco*3]
                y = kps[c_coco*3+1]
                visible = kps[c_coco*3 + 2]

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