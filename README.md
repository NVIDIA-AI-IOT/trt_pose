# trt_pose

<img src="https://user-images.githubusercontent.com/4212806/67125332-71a64580-f1a9-11e9-8ee1-e759a38de215.gif" height=256/>

trt_pose is aimed at enabling real-time pose estimation on NVIDIA Jetson.  You may find it useful for other NVIDIA platforms as well.  Currently the project includes

- Pre-trained models for human pose estimation capable of running in real time on Jetson Nano.  This makes it easy to detect features like ``left_eye``, ``left_elbow``, ``right_ankle``, etc.

- Training scripts to train on any keypoint task data in [MSCOCO](https://cocodataset.org/#home) format.  This means you can experiment with training trt_pose for keypoint detection tasks other than human pose.

To get started, follow the instructions below.  If you run into any issues please [let us know](../../issues).

## Getting Started

To get started with trt_pose, follow these steps.

### Step 1 - Install Dependencies

1. Install PyTorch and Torchvision.  To do this on NVIDIA Jetson, we recommend following [this guide](https://forums.developer.nvidia.com/t/72048)

2. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

    ```python
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sudo python3 setup.py install --plugins
    ```

3. Install other miscellaneous packages

    ```python
    sudo pip3 install tqdm cython pycocotools
    sudo apt-get install python3-matplotlib
    ```
    
### Step 2 - Install trt_pose

```python
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```

### Step 3 - Run the example notebook

We provide a couple of human pose estimation models pre-trained on the MSCOCO dataset.  The throughput in FPS is shown for each platform

| Model | Jetson Nano | Jetson Xavier | Weights |
|-------|-------------|---------------|---------|
| resnet18_baseline_att_224x224_A | 22 | 251 | [download (81MB)](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) |
| densenet121_baseline_att_256x256_B | 12 | 101 | [download (84MB)](https://drive.google.com/open?id=13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU) |

To run the live Jupyter Notebook demo on real-time camera input, follow these steps
 
1. Download the model weights using the link in the above table.  

2. Place the downloaded weights in the [tasks/human_pose](tasks/human_pose) directory

3. Open and follow the [live_demo.ipynb](tasks/human_pose/live_demo.ipynb) notebook

    > You may need to modify the notebook, depending on which model you use

## See also

- [torch2trt](http://github.com/NVIDIA-AI-IOT/torch2trt) - An easy to use PyTorch to TensorRT converter

- [JetBot](http://github.com/NVIDIA-AI-IOT/jetbot) - An educational AI robot based on NVIDIA Jetson Nano
- [JetRacer](http://github.com/NVIDIA-AI-IOT/jetracer) - An educational AI racecar using NVIDIA Jetson Nano
- [JetCam](http://github.com/NVIDIA-AI-IOT/jetcam) - An easy to use Python camera interface for NVIDIA Jetson

## References

Cao, Zhe, et al. "Realtime multi-person 2d pose estimation using part affinity fields." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple baselines for human pose estimation and tracking." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
