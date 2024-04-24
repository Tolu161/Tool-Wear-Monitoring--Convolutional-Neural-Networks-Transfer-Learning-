# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:18:35 2024

@author: egyto1
"""

# C3 C11 Labels  Predict 

#import necessary modules 
import numpy as np 


# a for loop to extract the vb wear for every 250 rows in case 1 data   
  

from C3DataMTF import Case_3
Case_3_VB = Case_3.VB

train_labels = []

#reshape train labels to obtain 
Case_3_VB = Case_3_VB.values.reshape(-1)
Case_3VB = Case_3_VB.reshape(-1,1)

Case_3VB = np.asfarray(Case_3VB)
    

#define the step size for samples
train_size = 250

# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_3VB), train_size):
        label = Case_3VB[i]
        train_labels.append(label)

# Convert the list to a NumPy array if needed
train_labels_C3 = np.array(train_labels)

# Print the extracted labels
#print(len(train_labels))  
np.save('C3_TrainLabelsPredict_250',train_labels_C3)





#CASE 11 Labels for prediction 


      
# a for loop to extract the vb wear for every 250 rows in case 1 data   
  
from C11DataMTF import Case_11
Case_11_VB = Case_11.VB

train_labels = []

#reshape train labels to obtain 
Case_11_VB = Case_11_VB.values.reshape(-1)
Case_11VB = Case_11_VB.reshape(-1,1)

Case_11VB = np.asfarray(Case_11VB)
    

#define the step size for samples
train_size = 250

# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_11VB), train_size):
        label = Case_11VB[i]
        train_labels.append(label)

# Convert the list to a NumPy array if needed
train_labels_C11 = np.array(train_labels)

np.save('C11_TrainLabelsPredict_250',train_labels_C11)






