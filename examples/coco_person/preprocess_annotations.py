import argparse
import json
import tqdm


def remove_link(cat, kp_a_name, kp_b_name):
    skeleton = cat['skeleton']
    keypoints = cat['keypoints']
    new_skeleton = []
    
    for link in skeleton:
        if (keypoints[link[0] - 1] == kp_a_name) and (keypoints[link[1] - 1] == kp_b_name):
            continue # skip
        if (keypoints[link[0] - 1] == kp_b_name) and (keypoints[link[1] - 1] == kp_a_name):
            continue # skip
        new_skeleton.append(link)
    cat['skeleton'] = new_skeleton

    
def add_link(cat, kp_a_name, kp_b_name):
    keypoints = cat['keypoints']
    cat['skeleton'].append([keypoints.index(kp_a_name) + 1, keypoints.index(kp_b_name) + 1])


def append_neck_keypoint(ann, cat):
    keypoints = cat['keypoints']
    kps = ann['keypoints']
    l_idx = 3 * keypoints.index('left_shoulder')
    r_idx = 3 * keypoints.index('right_shoulder')
    x_neck = round((kps[l_idx] + kps[r_idx]) / 2.0)
    y_neck = round((kps[l_idx + 1] + kps[r_idx + 1]) / 2.0)
    
    v_l = kps[l_idx + 2]
    v_r = kps[r_idx + 2]
    
    if v_l == 0 or v_r == 0:
        v_neck = 0
    elif v_l == 1 or v_r == 1:
        v_neck = 1
    else:
        v_neck = 2
    
    kps += [x_neck, y_neck, v_neck]
    
def get_cat(data, cat_name):
    return [c for c in data['categories'] if c['name'] == cat_name][0]

def get_anns(data, cat_id):
    return [a for a in data['annotations'] if a['category_id'] == cat_id]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_annotation_file', type=str, help='Path to COCO annotation file')
    parser.add_argument('output_annotation_file', type=str, help='Path to COCO annotation file')
    args = parser.parse_args()
    
    print('Loading...')
    with open(args.input_annotation_file, 'r') as f:
        data = json.load(f)
    
    print('Preprocessing...')
    cat = get_cat(data, 'person')
    cat_id = cat['id']
    anns = get_anns(data, cat_id)

    for a in anns:
        append_neck_keypoint(a, cat)

    cat['keypoints'].append('neck')
    remove_link(cat, 'left_shoulder', 'right_shoulder')
    remove_link(cat, 'left_shoulder', 'left_hip')
    remove_link(cat, 'right_shoulder', 'right_hip')
    add_link(cat, 'neck', 'nose')
    add_link(cat, 'neck', 'left_shoulder')
    add_link(cat, 'neck', 'right_shoulder')
    add_link(cat, 'neck', 'left_hip')
    add_link(cat, 'neck', 'right_hip')
    
    print('Saving...')
    with open(args.output_annotation_file, 'w') as f:
        json.dump(data, f)