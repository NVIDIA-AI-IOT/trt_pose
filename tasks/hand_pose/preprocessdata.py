import math
import pickle
import cv2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class preprocessdata:
    
    def __init__(self, topology, num_parts):
        self.joints = []
        self.dist_bn_joints = []
        self.topology = topology
        self.num_parts = num_parts
        self.text = "no hand"
        self.num_frames = 7
        self.prev_queue = [7]*self.num_frames
        
    def svm_accuracy(self, test_predicted, labels_test):
        predicted = []
        for i in range(len(labels_test)):
            if labels_test[i]==test_predicted[i]:
                predicted.append(0)
            else:
                predicted.append(1)
        accuracy = 1 - sum(predicted)/len(labels_test)
        return accuracy 
    def trainsvm(self, clf, train_data, test_data, labels_train, labels_test):
        clf.fit(train_data,labels_train)
        predicted_test = clf.predict(test_data)
        return clf, predicted_test   
    #def loadsvmweights():
    
    def joints_inference(self, image, counts, objects, peaks):     
        joints_t = []
        height = image.shape[0]
        width = image.shape[1]
        K = self.topology.shape[0]
        count = int(counts[0])
        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                picked_peaks = peaks[0][j][k]
                joints_t.append([round(float(picked_peaks[1]) * width), round(float(picked_peaks[0]) * height)])
        joints_pt = joints_t[:self.num_parts]  
        rest_of_joints_t = joints_t[self.num_parts:]
        """
        #when it does not predict a particular joint in the same association it will try to find it in a different association 
        for i in range(len(rest_of_joints_t)):
            l = i%self.num_parts
            if joints_pt[l] == [0,0]:
                joints_pt[l] = rest_of_joints_t[i]
        #if nothing is predicted 
        """
        if count == 0:
            joints_pt = [[0,0]]*self.num_parts
        return joints_pt
    def find_distance(self, joints):
        joints_features = []
        for i in joints:
            for j in joints:
                dist_between_i_j = math.sqrt((i[0]-j[0])**2+(i[1]-j[1])**2)
                joints_features.append(dist_between_i_j)
        return joints_features
    def print_label(self, image, gesture_joints):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        color = (255, 0, 0) 
        org = (50, 50)
        thickness = 2
        fontScale = 0.5
        if self.prev_queue == [1]*7:
            self.text = 'fist'
        elif self.prev_queue == [2]*7:
            self.text = 'pan'
        elif self.prev_queue == [3]*7:
            self.text = 'stop'
        elif self.prev_queue == [4]*7:
            self.text = 'peace'
        elif self.prev_queue == [5]*7:
            self.text = 'fine'
        elif self.prev_queue == [6]*7:
            self.text = 'no hand'
        elif self.prev_queue == [7]*7:
            self.text = 'no hand'
        image = cv2.putText(image, self.text, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
        return image

   
       
    
        
        