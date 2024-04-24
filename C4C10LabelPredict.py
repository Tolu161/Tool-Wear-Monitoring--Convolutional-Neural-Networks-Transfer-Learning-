# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:18:36 2024

@author: egyto1
"""


# C4 C10 Labels  Predict 

#import necessary modules 
import numpy as np 


#Extracting labels 

#CASE 4 and 10  Labels for prediction 


      
# a for loop to extract the vb wear for every 250 rows in case 1 data   
  
from C4DataMTF import Case_4
Case_4_VB = Case_4.VB

train_labels = []

#reshape train labels to obtain 
Case_4_VB = Case_4_VB.values.reshape(-1)
Case_4VB = Case_4_VB.reshape(-1,1)

Case_4VB = np.asfarray(Case_4VB)
    

#define the step size for samples
train_size = 250

# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_4VB), train_size):
        label = Case_4VB[i]
        train_labels.append(label)

# Convert the list to a NumPy array if needed
train_labels_C4 = np.array(train_labels)

# Print the extracted labels
#print(len(train_labels))  
np.save('C4_TrainLabelsPredict_250',train_labels_C4)





#CASE 10 Labels for prediction 


      
# a for loop to extract the vb wear for every 250 rows in case 1 data   
  
from C10DataMTF import Case_10
Case_10_VB = Case_10.VB

train_labels = []

#reshape train labels to obtain 
Case_10_VB = Case_10_VB.values.reshape(-1)
Case_10VB = Case_10_VB.reshape(-1,1)

Case_10VB = np.asfarray(Case_10VB)
    

#define the step size for samples
train_size = 250

# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_10VB), train_size):
        label = Case_10VB[i]
        train_labels.append(label)

# Convert the list to a NumPy array if needed
train_labels_C10 = np.array(train_labels)

np.save('C10_TrainLabelsPredict_250',train_labels_C10)








