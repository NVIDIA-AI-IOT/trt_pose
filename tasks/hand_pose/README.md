# Hand Pose Estimation And Classification

This project is an extention of TRT Pose for Hand Pose Detection. The project includes 

- Pretrained models for hand pose estimation capable of running in real time on Jetson Xavier NX.

- Scripts for applications of Hand Pose Estimation

  -  Hand gesture recoginition (hand pose classification) 
  
  -  Cursor control 
  
  -  Mini-Paint type of application 
  
- Pretrained model for gesture recoginition 

## Getting Started 

### Step 1 - Install trt_pose and it's dependencies 

Make sure to follow all the instructions from trt_pose and install all it's depenedencies. 
Follow the following instruction from https://github.com/NVIDIA-AI-IOT/trt_pose. 

### Step 2 - Install dependecies for hand pose 

### Step 3 - Run hand pose and it's applications 

    A) Hand Pose demo 

    B) Hand gesture recoginition (hand pose classification) 
    
        To make your own hand gesture classification from the hand pose estimation, follow the following steps 
        
        - Create your own dataset using the gesture_data_collection.ipynb or gesture_data_collection_with_pose.ipynb. This will allow you to create the type of gestures you want to classify. (eg. tumbs up, fist,etc). This notebook will automatically create a dataset with images and labels that is ready to be trained for gesture classification.
        - Train using the train_gesture_classification.ipynb notebook file. It uses an SVM from scikit-learn. you can experiment your other types of models to train. You can also infer in this training script. 
        

    C) Cursor control application

    D) Mini-Paint

The model was trained using the training script in trt_pose and the hand pose data collected in Nvidia.

Model details: resnet18
