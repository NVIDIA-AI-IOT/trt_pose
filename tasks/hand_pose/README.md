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
      
    pip install traitlets
     

### Step 3 - Run hand pose and it's applications 

A) Hand Pose demo 
      
   - Open and follow live_hand_pose.ipynb notebook. 

B) Hand gesture recoginition (hand pose classification) 
   - Install dependecies
      - scikit-learn 
         - pip install -U scikit-learn 
         - or install it from the source 
   The current gesture classification model supports six classes (fist, pan, stop, fine, peace, no hand). 
   More gestures can be added by a simple process of creating your own dataset and training it on an svm model. 
   An SVM model weight is provided for inference.
        
   To make your own hand gesture classification from the hand pose estimation, follow the following steps 
        
   - Create your own dataset using the gesture_data_collection.ipynb or gesture_data_collection_with_pose.ipynb. 
     This will allow you to create the type of gestures you want to classify. (eg. tumbs up, fist,etc). 
     This notebook will automatically create a dataset with images and labels that is ready to be trained for gesture classification.
        
   - Train using the train_gesture_classification.ipynb notebook file. It uses an SVM from scikit-learn. 
     Other types of models can also be experimented. 
        
 C) Cursor control application
 
    - Install dependecies 
       - pyautogui 
          - python3 -m pip install pyautogui
          - On jetson install it from the source 
          
    - Open and follow the cursor_control_live_demo.ipynb notebook. 
    - This will allow you to control your mouse cursor on your desktop. It uses the hand gesture classification. 
      When your hand geture is pan, you can control the cursor. when it is stop, it's left click. 

D) Mini-Paint

The model was trained using the training script in trt_pose and the hand pose data collected in Nvidia.

Model details: resnet18
