#!/usr/bin/env python

import numpy as np
from keras.layers.core import Lambda
from SpatialPyramidPooling import SpatialPyramidPooling
from keras.models import Model, load_model
import keras.backend as K
def run_12ECG_classifier(data,header_data,classes,model):
    data=data/1000
    data=data.T
    data=np.expand_dims(data,axis=1)
    data=np.expand_dims(data,axis=0) 
    num_classes = len(classes)
    
    
    score = model.predict(data).reshape(num_classes,1)
#    print(score)
    label = score
    label[label>=0.5]=1
    label[label<0.5]=0


    return label, score
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
def load_12ECG_model():
    # load the model from disk 
    filename='ecg_2.h5'
    loaded_model = load_model(filename,custom_objects={'f1':f1,'SpatialPyramidPooling':SpatialPyramidPooling,'K':K})

    return loaded_model
