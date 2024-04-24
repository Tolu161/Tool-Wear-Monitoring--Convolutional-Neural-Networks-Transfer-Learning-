# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:18:07 2024

@author: egyto1
"""

#lenet prediction - rms 


#LeTrain Prediction 

# Importing the necessary modules 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

# Load the data from directories

#three sensors - 250 samples 
#250 samples - USE 
#train_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C1_250.npy'
#test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9__250.npy'


'''
#switching c2 to train 
train_dir= 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C2_250.npy'
test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C12_250.npy'
train_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C2_TrainLabelsPredict_250.npy'
test_labels_dir =  'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C12_TrainLabelsPredict_250.npy'
testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C1_250.npy'
valC12_dir =  'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9__250.npy'
testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabelsPredict_250.npy'
valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabelsPredict_250.npy'
'''

'''
train_dir =  'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C1_250.npy'
test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9__250.npy'
train_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabelsPredict_250.npy'
test_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabelsPredict_250.npy'
testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C2_250.npy'
valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C12_250.npy'
testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C2_TrainLabelsPredict_250.npy'
valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C12_TrainLabelsPredict_250.npy'
'''


'''

testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C1_250.npy'

valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9__250.npy'

testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabelsPredict_250.npy'

valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabelsPredict_250.npy'

train_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C3_250.npy'

test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C11_250.npy'

train_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C3_TrainLabelsPredict_250.npy'

test_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C11_TrainLabelsPredict_250.npy'

'''



'''
testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C1_250.npy'

valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9__250.npy'

testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabelsPredict_250.npy'

valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabelsPredict_250.npy'

train_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C4_250.npy'

test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C10_250.npy'

train_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C4_TrainLabelsPredict_250.npy'

test_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C10_TrainLabelsPredict_250.npy'
'''



train_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C1_250.npy'

test_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C9__250.npy'

train_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabelsPredict_250.npy'

test_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabelsPredict_250.npy'

'''

testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C4_250.npy'

valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C10_250.npy'

testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C4_TrainLabelsPredict_250.npy'

valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C10_TrainLabelsPredict_250.npy'

'''


#C3
testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C3_250.npy'
#C4
#testC2_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C4_250.npy'

#C11
valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C11_250.npy'

#C10
#valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C10_250.npy'


traindata = np.load(train_dir)
testdata = np.load(test_dir)


# Reshape and preprocess the input data for three sensors
traindata = traindata.reshape(traindata.shape[0], 224, 224, 3)
testdata = testdata.reshape(testdata.shape[0], 224, 224, 3)

print(traindata.shape)
print(testdata.shape)

traindata = traindata.astype('float32') / 255
testdata = testdata.astype('float32') / 255

#250 samples - USE 
#train_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C1_TrainLabelsPredict_250.npy'
#test_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C9_TrainLabelsPredict_250.npy'


train_labels = np.load(train_labels_dir)
test_labels = np.load(test_labels_dir)




#C3 LABEL 
testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C3_TrainLabelsPredict_250.npy'

#C11 LABEL 
valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C11_TrainLabelsPredict_250.npy'


'''
#C4 LABEL 
testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C4_TrainLabelsPredict_250.npy'

#C10 LABEL 
valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C10_TrainLabelsPredict_250.npy'

'''


#TL Data 
#three sensors 

testC2 = np.load(testC2_dir)
#3 SENSORS
testC2 = testC2.reshape(testC2.shape[0], 224, 224, 3) 
testC2 = testC2.astype('float32') / 255


#valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C12_125.npy'
#valC12_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/MTFs_array_C12_250.npy'

valC12 = np.load(valC12_dir)
valC12 = valC12.reshape(valC12.shape[0], 224,224,3) 
valC12 = valC12.astype('float32') / 255


#250 samples 
#testC2_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C2_TrainLabelsPredict_250.npy'
#valc12_labels_dir = 'C:/Users/egyto1/OneDrive - The University of Nottingham/mill/mill/C12_TrainLabelsPredict_250.npy'

testC2_labels = np.load(testC2_labels_dir)
valC12_labels = np.load(valc12_labels_dir)


# for training generating train and validation data from one case 
traindata, valdata, train_labels, val_labels = train_test_split(traindata, train_labels, test_size=0.1, random_state=42)

#for transfer learning , generating train and validation 
testC2, valC2, testC2_labels, valC2_labels = train_test_split(testC2, testC2_labels, test_size=0.9, random_state=42)


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
model.compile(loss=keras.metrics.mean_squared_error, optimizer=opt, metrics=['accuracy','mean_squared_error'])


# Compile the model
#model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#batch_size=128
# Train the model
hist = model.fit(traindata, train_labels, epochs=300, verbose=1, validation_data=(valdata, val_labels))

# Evaluate the model
score = model.evaluate(testdata, test_labels)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])



# Plot model MAE and validation MAE
plt.plot(hist.history["mean_squared_error"])
plt.plot(hist.history['val_mean_squared_error'])
plt.title(" Model RMS and Validation RMS")
plt.ylabel("RMS")
plt.xlabel("Epoch")
plt.legend(["RMS", "Validation RMS"])
plt.show()

#generate a plot of C1 nad C9 for predicted values of toolwear 
predicted_wear = model.predict(traindata)

plt.plot(predicted_wear, label='Predicted')
plt.plot(train_labels, label='Actual')
plt.title(" Actual & Predicted Wear Values vs Sample Index  Training ")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()

predicted_wear = model.predict(testdata)

plt.plot(predicted_wear, label='Predicted')
plt.plot(test_labels, label='Actual')
plt.title("  Actual & Predicted Wear Values vs Sample Index Training ")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()


# to plot true against predicted

# Generate the predicted wear values
predicted_wear = model.predict(testdata)

# Plot the actual vs predicted wear values
plt.scatter(test_labels, predicted_wear)
plt.title("Actual vs Predicted Wear Values")
plt.xlabel("Actual Wear Value")
plt.ylabel("Predicted Wear Value")
plt.show()





# Save the weights of the trained model 

model.save_weights('initial_model_weights.h5')

# Finetuning LeNet 5 

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
new_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy','mean_squared_error'])





'''
# Remove the last layers of the original model
output = model.layers[-4].output

# Add new fully connected layers
output = Flatten()(output)
output = Dense(120, activation='tanh')(output)
output = Dense(84, activation='tanh')(output)
output = Dense(1, activation='linear')(output)

# Define the new model
new_model = Model(inputs=model.input, outputs=output)

# Freeze layers
for layer in new_model.layers[:-3]:
    layer.trainable = False

# Compile the new model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
new_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mean_squared_error'])
'''

# Display the summary of the new model
new_model.summary()



# Continue training the new model on new data (case2data)
# Replace traindata and train_labels l with your new dataset
hist = new_model.fit(testC2, testC2_labels , epochs=300, verbose=1, validation_data=(valC2, valC2_labels))

# Evaluate the model
score = model.evaluate(valC12, valC12_labels)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])



# Generate the predicted
predicted_value = model.predict(testC2)
predicted_values = np.argmax(predicted_value, axis=1)


# Plot model MAE and validation MAE
plt.plot(hist.history["mean_squared_error"])
plt.plot(hist.history['val_mean_squared_error'])
plt.title("Model RMS and Validation RMS")
plt.ylabel("RMS")
plt.xlabel("Epoch")
plt.legend(["RMS", "Validation RMS"])
plt.show()

'''

# to plot the wear graph 

# Generate the predicted wear values
predicted_wear = new_model.predict(testC2)

# Plot the actual vs predicted wear values
plt.scatter(testC2_labels, predicted_wear)
plt.title("Actual vs Predicted Wear Values")
plt.xlabel("Actual Wear Value")
plt.ylabel("Predicted Wear Value")
plt.show()
'''
'''
# Plot both actual and predicted wear values on the same graph
# Generate the predicted wear values
predicted_wear = new_model.predict(testC2)

plt.plot(predicted_wear, label='Predicted')
plt.plot(testC2_labels, label='Actual')
plt.title("Actual & Predicted Wear Values vs Sample Index Transfer learning")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
plt.show()

predicted_wear = new_model.predict(valC12)

plt.plot(predicted_wear, label='Predicted')
plt.plot(valC12_labels, label='Actual')
plt.title("Actual & Predicted Wear Values vs Sample Index Transfer Learning")
plt.xlabel("Sample Index")
plt.ylabel("Wear Value")
plt.legend()
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
plt.title("Actual vs Predicted Wear Values - Test")
plt.xlabel("Actual Wear Value")
plt.ylabel("Predicted Wear Value")
plt.show()







print(predicted_wear[0:10])
print (valC12_labels[0:10] )



