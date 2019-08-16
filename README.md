# TensorRT Pose Estimation

This project enables multi-instance pose estimation accelerated by NVIDIA TensorRT.  It is ideal for use in applications where low latency is necessary.  It includes

- Training scripts to train on any keypoint task data in MSCOCO format

- Evaluation scripts to assess the accuracy of a trained model 
- A collection of models that may be easily optimized with TensorRT using [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

This project can be used for pre-trained task of human pose estimation, or extended for something new.

If you run into any issues please [let us know](../../issues).

## Human pose

To try out these pre-trained models, please follow the human pose [notebooks](notebooks/human_pose).

| Name | Accuracy | Jetson Nano | Jetson Xavier | Pre-trained Weights |
|-------|------------|-------------|---------------|---------------------|
| [resnet50_baseline_att_256x256_A](experiments/resnet50_baseline_att_256x256_A) |  |  |  |  |
| [densenet121_baseline_att_256x256_A](experiments/densenet121_baseline_att_256x256_A) |  |  |  |  |
| [densenet169_baseline_att_256x256_A](experiments/densenet169_baseline_att_256x256_A) |  |  |  |  |

For more information on how to train or evaluate a human pose model, please read the human_pose [documentation](docs/human_pose.md).
