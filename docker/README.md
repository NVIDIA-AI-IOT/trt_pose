# trt_pose Docker

This directory contains scripts to build a Docker container for easily try running `trt_pose`.  

## Quick Start

Run the follwoing commands.

```
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
./docker/set_nvidia_runtime.sh
./docker/build.sh
./docker/run.sh --jupyter
```

Then, open a browser on your PC and access `jetson.ip.add.ress:8888`.

## Attention

Run those scripts from the root directory of `trt_pose`, not inside the `docker` directory.
