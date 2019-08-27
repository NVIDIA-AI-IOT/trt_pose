# TensorRT Pose Estimation

This project features multi-instance pose estimation accelerated by NVIDIA TensorRT.  It is ideal for use in applications where low latency is necessary.  It includes

- Training scripts to train on any keypoint task data in MSCOCO format

- A collection of models that may be easily optimized with TensorRT using [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

This project can be used easily for the task of human pose estimation, or extended for something new.

If you run into any issues please [let us know](../../issues).

## Tasks

### Human Pose

This task involves detecting human body pose using models trained on the MSCOCO dataset.   To get started follow the human pose [README](tasks/human_pose). 

## Setup

```bash
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```
