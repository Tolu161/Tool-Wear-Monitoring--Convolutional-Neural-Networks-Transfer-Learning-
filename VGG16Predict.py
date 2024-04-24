#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:58:11 2024

@author: toluojo
"""

#VGG-16 Implementation  - mofidying layers  - PRETRAINED PREDICTION - modifying for regression not binary classification , predict continuous values 

# Importing the needed packages 
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16

import os 
import matplotlib.pyplot as plt 
from tensorflow.keras import layers 
from sklearn import metrics 
from sklearn.metrics import classification_report 



#print(keras.utils.get_file('', ''))


# import train labels and test labels from 
from C1C9LabelPredict import train_labels_C1


#import train data 

# 25/01/24 sort this directory out instead of importing from scripts just link to the directories. 

#train_dir = '/Users/toluojo/Documents/University of Nottingham /YEAR 5 /MMME 4086 - Indv Project /mill/MTF_images_resized'
train_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTF_images_resized_SMCAC'

print("done")
#load images from train directory 

traindata = []
for filename in os.listdir(train_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(train_dir,filename)
        img = plt.imread(img_path)
        traindata.append(img)
        
#convert list to numpy array 

traindata = np.array(traindata)

# print commands to see issue in code 
print("done 1")
        

# Path to the manually downloaded weights file
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

print("done 2 ")


# Path to the manually downloaded weights file
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

#weights_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

#LOAD BASE MODEL 
base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights ='imagenet')
print(base_model.summary())


#added this line to avoid errors with input sizes etc .
#ensure the input shape matches the expected shape of the base model 

input_layer = tf.keras.Input(shape = (224,224,3))

# connnect input layer to base model 
x = base_model(input_layer)

'''
#defining how many layers to freeze during training, layers in base model are switched from 
#trainable to nontrainable depending on size of finetuning parameter 

if fine_tune > 0: 
    for layer in base_model.layers[:-fine_tune]:
        layer.trainable = False
'''        

for layer in base_model.layers:
    layer.trainable = False


# Flatten the output layer to 1 dimension
x = layers.Flatten()(x)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)


# Add a final sigmoid layer with 1 node for classification output , removed the sigmoid activation layer in order to get a linear output for regression line 
# to modify the last layer of the keras regression instead of classification  - will change the activation function
output_layer = layers.Dense(1)(x)

model = tf.keras.models.Model(inputs =input_layer, outputs = output_layer)

#use mean squared error for the loss function , mae - mean absolute error 

model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss = 'mean_squared_error',metrics = ['rms'])
model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss = 'mean_absolute_error',metrics = ['mae'])


vgghist = model.fit(traindata, train_labels_C1, steps_per_epoch = 100, epochs = 1000)


predicted_class = model.predict(traindata)   

# Print the first few predicted class labels
print("Predicted Class Labels:")
print(predicted_class[:20])  # Print the first 5 predicted class labels
print(train_labels_C1[:20])





 #visualise plotting data 
 
from mpl_toolkits.axes_grid1 import ImageGrid


# Plot Visualisation
plt.plot(vgghist.history["mae"])
plt.plot(vgghist.history['loss'])

plt.title("Model Mean Absolute Error and loss")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Epoch")
plt.legend(["MAE","loss"])
plt.show()









































