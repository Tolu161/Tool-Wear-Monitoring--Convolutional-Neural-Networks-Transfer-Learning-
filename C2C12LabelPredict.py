# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:31:28 2024

@author: egyto1
"""

#C2C12 PREDICT 


#import necessary modules 
import numpy as np 


#Extracting labels 

#CASE 2 Labels for prediction 


      
# a for loop to extract the vb wear for every 250 rows in case 1 data   
  
from C2DataMTF import Case_2
Case_2_VB = Case_2.VB

train_labels = []

#reshape train labels to obtain 
Case_2_VB = Case_2_VB.values.reshape(-1)
Case_2VB = Case_2_VB.reshape(-1,1)

Case_2VB = np.asfarray(Case_2VB)
    

#define the step size for samples
train_size = 500

# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_2VB), train_size):
        label = Case_2VB[i]
        train_labels.append(label)

# Convert the list to a NumPy array if needed
train_labels_C2 = np.array(train_labels)

# Print the extracted labels
#print(len(train_labels))  
np.save('C2_TrainLabelsPredict_500',train_labels_C2)





#CASE 12 Labels for prediction 


      
# a for loop to extract the vb wear for every 250 rows in case 1 data   
  
from C12DataMTF import Case_12
Case_12_VB = Case_12.VB

train_labels = []

#reshape train labels to obtain 
Case_12_VB = Case_12_VB.values.reshape(-1)
Case_12VB = Case_12_VB.reshape(-1,1)

Case_12VB = np.asfarray(Case_12VB)
    

#define the step size for samples
train_size = 500

# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_12VB), train_size):
        label = Case_12VB[i]
        train_labels.append(label)

# Convert the list to a NumPy array if needed
train_labels_C12 = np.array(train_labels)

np.save('C12_TrainLabelsPredict_500',train_labels_C12)










