#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:05:28 2024

@author: toluojo
"""

#VGG-16 Implementation  - CLASSIFICATION APPLICATION FOR PRETRAINED MODEL -CLASSIFICATION -
#this is not transfer learning 

# Importing the needed packages 
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers 
from tensorflow.keras.utils import to_categorical
import numpy as np
import os 
import matplotlib.pyplot as plt 
from sklearn import metrics 
from sklearn.metrics import classification_report 

#print(keras.utils.get_file('', ''))


# import train labels and test labels from 

#from C1C9Label import train_labels


'''
from C1C9_Testtrain import traindata 

'''


# 25/01/24 sort this directory out instead of importing from scripts just link to the directories. 

#train_dir = '/Users/toluojo/Documents/University of Nottingham /YEAR 5 /MMME 4086 - Indv Project /mill/MTF_images_resized'

#test_dir = '/Users/toluojo/Documents/University of Nottingham /YEAR 5 /MMME 4086 - Indv Project /mill/MTF_images_C9'

'''
train_labels_dir = '/Users/toluojo/Documents/University of Nottingham /YEAR 5 /MMME 4086 - Indv Project /mill/C1_TrainLabels.npy'
'''

#train_labels_dir = '/Users/toluojo/Documents/University of Nottingham /YEAR 5 /MMME 4086 - Indv Project /mill/C1_TrainLabels.npy'



'''OTHMAN laptop


train_dir = 'C:/Users/oerro/Downloads/mill/mill/MTF_images_resized'

train_labels_dir = 'C:/Users/oerro/Downloads/mill/mill/C1_TrainLabels.npy'

test_dir = 'C:/Users/oerro/Downloads/mill/mill/MTF_images_C9'

test_labels_dir = 'C:/Users/oerro/Downloads/mill/mill/C9_TrainLabels.npy' 

'''


#train_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTF_images_resized_SMCAC'

#250 SAMPLES
train_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array.npy'
train_labels_dir ='C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabels.npy'


#125 samples
train_dir = ''
train_labels_dir = ''


print("done")
#load images from train directory 


'''
traindata = []
for filename in os.listdir(train_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(train_dir,filename)
        img = plt.imread(img_path)
        traindata.append(img)
'''  



traindata = np.load(train_dir)     
#convert list to numpy array 

traindata = np.array(traindata)

# print commands to see issue in code 
print("done 1")
        

#load the train labels from the directory 
train_labels = np.load(train_labels_dir)


#convert labels to a categorical format - apply one hot encoding , num classes -4 because labels range from 1 to 3 , normally would be num_class = 3 but since class is 1,2,3 it is numclass=4 
#to_categorical expects the label to start from 0 
train_labels_categorical = to_categorical(train_labels,num_classes=3 )


# Path to the manually downloaded weights file
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

print("done 2 ")
#weights_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

#LOAD BASE MODEL 
base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights ='imagenet')
print(base_model.summary())


#added this line to avoid errors with input sizes etc .
#ensure the input shape matches the expected shape of the base model 

input_layer = tf.keras.Input(shape = (224,224,3))

# connnect input layer to base model 
x = base_model(input_layer)

print("done 3")
for layer in base_model.layers:
    layer.trainable = False


# Flatten the output layer to 1 dimension
x = layers.Flatten()(x)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final softmax layer with 3  nodes for classification output - of three
output_layer = layers.Dense(3, activation='softmax')(x)

model = tf.keras.models.Model(inputs =input_layer, outputs = output_layer)

print("done 4")

# perhaps using a loss of categorical  crosentropy / binary classification is causing low accuracy results 
model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.0003), loss = 'categorical_crossentropy',metrics = ['acc'])

print("done 5 ")


#perform mutliple iterations 
#num_iterations = 2
#for i in range(num_iterations):
    #print(f"Iteration {i+1}/{num_iterations}")

#train model 
vgghist = model.fit(traindata, train_labels_categorical, steps_per_epoch=100, epochs=1000)
confusion_matrices = []

# to loop throguh each epoch and generate confusion matrices : 

    # generate the predicted 
predicted_class = model.predict(traindata)

# assigning each predicted value with the class with the hgihest probability , extracting the index of the maximum probability along each row 
predicted_labels = np.argmax(predicted_class, axis =1)
    
# Print the first few predicted class labels
print("Predicted Class Labels:")
print(predicted_class[:20])  # Print the first 5 predicted class labels

# generate the confusion matrix for each epoch 
confusion_matrix = metrics.confusion_matrix(train_labels, predicted_labels)
confusion_matrices.append(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels= [0,1,2])

cm_display.plot(cmap= "Blues")
plt.title("Confusion Matrix")
plt.show()
    
# only produce classification report for last iteration

report = classification_report(train_labels,predicted_labels , labels=[0,1,2],target_names = ["Class 0 ", "Class 1", "Class 2"]) 
print(report)
        
    
#vgghist = model.fit(traindata, train_labels_categorical, steps_per_epoch = 100, epochs = 10)


from mpl_toolkits.axes_grid1 import ImageGrid


# Plot Visualisation
plt.plot(vgghist.history["acc"])
plt.plot(vgghist.history['loss'])

plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","loss"])
plt.show()


fig = plt.figure(figsize=(10, 6))      
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 1),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.3,
                 )

#plt.imshow(GAF_sample[0], cmap='rainbow')
grid = grid[0]


# Plot Visualisation (Accuracy-Loss Graph)
fig, ax = plt.subplots(figsize=(10, 6))

#Plotting the accuracy and the loss 
ax.plot(vgghist.history["acc"], label="Accuracy")
ax.plot(vgghist.history['loss'], label="Loss")

ax.set_title("Model Accuracy and Loss")
ax.set_ylabel("Metrics")
ax.set_xlabel("Epoch")
ax.legend()
plt.show()



#evaluate model with case 9 

# Load test data

test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9.npy'
test_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabels.npy'
test_data = np.load(test_dir)
test_labels = np.load(test_labels_dir)
test_labels_categorical = to_categorical(test_labels,num_classes=3 )

# Load model weights
#model.load_weights(weights_path)

# Make predictions
predicted_class = model.predict(test_data)
predicted_labels = np.argmax(predicted_class, axis=1)

# Compute confusion matrix
confusion_matrix = metrics.confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1, 2])
cm_display.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Generate classification report
report = classification_report(test_labels, predicted_labels, labels=[0, 1, 2], target_names=["Class 0", "Class 1", "Class 2"])
print(report)



print(predicted_labels)


# Evaluate model on test data
loss, accuracy = model.evaluate(test_data, test_labels_categorical)

# Print accuracy and loss
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

























'''
im = grid.imshow(vgghist.history["acc"], cmap='rainbow', origin='lower')
grid.set_title('acc ', fontdict={'fontsize': 12})
grid.cax.colorbar(im)

g1 = grid.imshow(vgghist.history["loss"], cmap='rainbow', origin='lower')
grid.set_title('loss ', fontdict={'fontsize': 12})
grid.cax.colorbar(g1)


grid.cax.toggle_label(True)
'''


























'''

# Plot Accuracy
plt.plot(vgghist.history["acc"])
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Plot Loss
plt.plot(vgghist.history["loss"])
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

'''











































