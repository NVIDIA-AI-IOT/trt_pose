import json
import trt_pose.coco

import torch
import torch2trt
from torch2trt import TRTModule


import cv2
import torchvision.transforms as transforms
import PIL.Image

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

WIDTH = 224
HEIGHT = 224
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'



with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)


num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print("Reading TensorRT model.......")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

print("TensorRT model loaded")

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
    
    
    
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


def execute(change):
    data = preprocess(change)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    draw_objects(change, counts, objects, peaks)
    
    
    
def main():

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame,(224,224))

        execute(frame)
        frame = cv2.resize(frame,(640,640))
        cv2.imshow("test", frame)
        if not ret:
            break
        key = cv2.waitKey(1)
        
        # To close the window press "Q"
        if key & 0xFF == ord('q') or key ==27:
            break


    cam.release()

    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()