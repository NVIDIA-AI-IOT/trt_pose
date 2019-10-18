# TensorRT Pose Estimation

This project features multi-instance pose estimation accelerated by NVIDIA TensorRT.  It is ideal for applications where low latency is necessary.  It includes

- Training scripts to train on any keypoint task data in MSCOCO format

- A collection of models that may be easily optimized with TensorRT using [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

This project can be used easily for the task of human pose estimation, or extended for something new.

If you run into any issues please [let us know](../../issues).


## Tasks

### Human Pose

<img src="https://user-images.githubusercontent.com/4212806/67125332-71a64580-f1a9-11e9-8ee1-e759a38de215.gif" height=256/>

This task involves detecting human body pose using models trained on the MSCOCO dataset.  To get started, select one of the models below and follow the [live_demo.ipynb](tasks/human_pose/live_demo.ipynb) notebook to visualize the results on a live camera feed.  Place the downloaded weights in the [tasks/human_pose](tasks/human_pose) directory, and modify the notebook accordingly.

> More than 20FPS on Jetson Nano.  Concrete benchmarks coming soon :)

| Model | Jetson Nano | Jetson Xavier | Weights |
|-------|-------------|---------------|---------|
| resnet18_baseline_att_224x224_A |  | 251 | [download (81MB)](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) |
| densenet121_baseline_att_256x256_B |  | 101 | [download (84MB)](https://drive.google.com/open?id=13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU) |
<!--
| resnet50_baseline_att_256x256_A |  |  | [download (182MB)](https://drive.google.com/open?id=1eLgzGsh1yjuLG66r9BFmoOzp3nTdVHS2) |
| resnet50_baseline_att_384x384_A |  |  | [download (182MB)](https://drive.google.com/open?id=1ck66N0Lqxqcg-7OImh_5YNwvnrb9yHym) |
| densenet121_baseline_att_224x224_B |  |  | [download (84MB)](https://drive.google.com/open?id=1ZP6Wh9CpFQxiRJYO9ECyIVU-soy4aUoW) |
| densenet121_baseline_att_320x320_A |  |  | [download (84MB)](https://drive.google.com/open?id=1SX-LWAfYNdcNKb42b31UmZwsjXmB3a9l) |
-->


## Setup

Assuming you have PyTorch, TensorRT and torchvision installed on your system

```bash
sudo pip3 install tqdm cython pycocotools
sudo apt-get install python3-matplotlib
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```

## See also

- [torch2trt](http://github.com/NVIDIA-AI-IOT/torch2trt) - An easy to use PyTorch to TensorRT converter

- [JetBot](http://github.com/NVIDIA-AI-IOT/jetbot) - An educational AI robot based on NVIDIA Jetson Nano
- [JetRacer](http://github.com/NVIDIA-AI-IOT/jetracer) - An educational AI racecar using NVIDIA Jetson Nano
- [JetCam](http://github.com/NVIDIA-AI-IOT/jetcam) - An easy to use Python camera interface for NVIDIA Jetson
