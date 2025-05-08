# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:36:57 2024

@author: egyto1
"""

#LeTrain Prediction 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#switching c2 to train 
train_dir= 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C2_250.npy'
test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C12_250.npy'
train_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C2_TrainLabelsPredict_250.npy'
test_labels_dir =  'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C12_TrainLabelsPredict_250.npy'
testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C1_250.npy'
valC12_dir =  'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9__250.npy'
testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabelsPredict_250.npy'
valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabelsPredict_250.npy'



traindata = np.load(train_dir)
testdata = np.load(test_dir)


# Reshape and preprocess the input data for three sensors
traindata = traindata.reshape(traindata.shape[0], 224, 224, 3)
testdata = testdata.reshape(testdata.shape[0], 224, 224, 3)

print(traindata.shape)
print(testdata.shape)

traindata = traindata.astype('float32') / 255
testdata = testdata.astype('float32') / 255


train_labels = np.load(train_labels_dir)
test_labels = np.load(test_labels_dir)


#TL Data 
#three sensors 

testC2 = np.load(testC2_dir)
#3 SENSORS
testC2 = testC2.reshape(testC2.shape[0], 224, 224, 3) 
testC2 = testC2.astype('float32') / 255

#C10
#valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C10_250.npy'
valC12 = np.load(valC12_dir)
valC12 = valC12.reshape(valC12.shape[0], 224,224,3) 
valC12 = valC12.astype('float32') / 255



testC2_labels = np.load(testC2_labels_dir) 
valC12_labels = np.load(valc12_labels_dir)

# for training generating train and validation data from one case 
traindata, valdata, train_labels, val_labels = train_test_split(traindata, train_labels, test_size=0.6, random_state=42)

#for transfer learning , generating train and validation 
testC2, valC2, testC2_labels, valC2_labels = train_test_split(testC2, testC2_labels, test_size=0.6, random_state=42)
 


# Define the LeNet CNN architecture
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), padding="valid", activation="tanh", input_shape=(224, 224, 3)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(16, kernel_size=(5, 5), padding="valid", activation="tanh"))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss=keras.metrics.mean_absolute_error, optimizer=opt, metrics=['accuracy','mean_absolute_error'])


# Compile the model
#model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#batch_size=128
# Train the model and validaitng from the same case 
hist_train_C2 = model.fit(traindata, train_labels , epochs=300, verbose=1, validation_data=(valdata, val_labels))

# Evaluate the model
score = model.evaluate(testdata, test_labels)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
 
# Plot model MAE and validation MAE
plt.plot(hist_train_C2.history["mean_absolute_error"])
plt.plot(hist_train_C2.history['val_mean_absolute_error'])
plt.title("Model MAE and Validation MAE")
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.legend(["MAE", "Validation MAE"])
plt.show()

'''
#generate a plot of C1 nad C9 for predicted values of toolwear 
predicted_wear = model.predict(traindata)

plt.plot(predicted_wear, label='Predicted')
plt.plot(train_labels, label='Actual')
plt.title("Actual & Predicted Wear Values vs Sample Index - Train")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()


#generate a plot of C1 nad C9 for predicted values of toolwear 
predicted_wear = model.predict(valdata)

plt.plot(predicted_wear, label='Predicted')
plt.plot(val_labels, label='Actual')
plt.title("Actual & Predicted Wear Values vs Sample Index - Train")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()
'''


predicted_wear = model.predict(testdata)

plt.plot(predicted_wear, label='Predicted')
plt.plot(test_labels, label='Actual')
plt.title(" C9: Actual & Predicted Wear Values vs Sample Index -")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()


# to plot true against predicted

# Generate the predicted wear values
predicted_wear_train_C2 = model.predict(testdata)

# Plot the actual vs predicted wear values
plt.scatter(test_labels, predicted_wear_train_C2)
plt.title("Actual vs Predicted Wear Values")
plt.xlabel("Actual Wear Value")
plt.ylabel("Predicted Wear Value")
plt.show()


#plotting the predicted values against the actual values 



# Save the weights of the trained model 

model.save_weights('initial_model_weights.h5')


# Define the new model for transfer learning
new_model = Sequential()
new_model.add(Conv2D(6, kernel_size=(5, 5), padding="valid", activation="tanh", input_shape=(224, 224, 3)))
new_model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
new_model.add(Conv2D(16, kernel_size=(5, 5), padding="valid", activation="tanh"))
new_model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
new_model.add(Flatten())
new_model.add(Dense(120, activation='tanh'))
new_model.add(Dense(84, activation='tanh'))
new_model.add(Dense(1, activation='linear'))

new_model.summary()

# Load the weights from the initial model
new_model.load_weights('initial_model_weights.h5')

# Freeze layers
for layer in new_model.layers[:-3]:
    layer.trainable = False

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

#opt.build(new_model.trainable_variables)


# Compile the new model
new_model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy','mean_absolute_error'])

# Continue training the new model on new data (case2data)
# Replace traindata and train_labels l with your new dataset
hist = new_model.fit(testC2, testC2_labels , epochs=300, verbose=1, validation_data=(valC2, valC2_labels))

# Evaluate the model
score = model.evaluate(valC12, valC12_labels)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])



# Generate the predicted
predicted_value = model.predict(valC12)
predicted_values = np.argmax(predicted_value, axis=1)


# Plot model MAE and validation MAE
plt.plot(hist.history["mean_absolute_error"])
plt.plot(hist.history['val_mean_absolute_error'])
plt.title("Model MAE and Validation MAE")
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.legend(["MAE", "Validation MAE"])
plt.show()


# TRAIN DATA PLOT FOR TRANSFER LEARNING 
'''
# Plot both actual and predicted wear values on the same graph
# Generate the predicted wear values
predicted_wear = new_model.predict(testC2)

plt.plot(predicted_wear, label='Predicted')
plt.plot(testC2_labels, label='Actual')
plt.title(" Actual & Predicted Wear Values vs Sample Index-Train")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()

# to plot the wear graph 

# Generate the predicted wear values
predicted_wear = new_model.predict(testC2)

# Plot the actual vs predicted wear values
plt.scatter(testC2_labels, predicted_wear)
plt.title("Actual vs Predicted Wear Values Train ")
plt.xlabel("Actual Wear Value")
plt.ylabel("Predicted Wear Value")
plt.show()
'''

#TEST DATA FOR TRANSFER LEARNING

predicted_wear = new_model.predict(valC12)

plt.plot(predicted_wear, label='Predicted')
plt.plot(valC12_labels, label='Actual')
plt.title("Actual & Predicted Wear Values vs Sample Index-Test")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()

# to plot the wear graph 

# Generate the predicted wear values
predicted_wear = new_model.predict(valC12)

# Plot the actual vs predicted wear values
plt.scatter(valC12_labels, predicted_wear)
plt.title("Actual vs Predicted Wear Values")
plt.xlabel("Actual Wear Value")
plt.ylabel("Predicted Wear Value")
plt.show()


#plotting the predicted values against the actual values


print(predicted_wear[0:10])
print (valC12_labels[0:10] )
