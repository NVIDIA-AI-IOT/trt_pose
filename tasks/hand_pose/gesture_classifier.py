from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class gesture_classifier:
    
    def __init__(self):
        pass
        
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