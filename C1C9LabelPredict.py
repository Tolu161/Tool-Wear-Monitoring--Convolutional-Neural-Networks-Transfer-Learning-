#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:24:23 2024

@author: toluojo
"""

#PREDICTION LABELS CASE 1 AND CASE 9 

#import necessary modules 
import numpy as np 
import pandas as pd 


#Extracting labels 

#CASE 1 Labels for prediction 

# EXTRACTING TRAINING DATA 
      
# a for loop to extract the vb wear for every 250 rows in case 1 data   
  
Case_1 = pd.read_csv('Mill_Case1_gma.csv')

Case_1_VB = Case_1.VB

train_labels = []

#reshape train labels to obtain 
Case_1_VB = Case_1_VB.values.reshape(-1)
Case_1VB = Case_1_VB.reshape(-1,1)

Case_1VB = np.asfarray(Case_1VB)
    

#define the step size for samples
train_size = 500

# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_1VB), train_size):
        label = Case_1VB[i]
        train_labels.append(label)

# Convert the list to a NumPy array if needed
train_labels_C1 = np.array(train_labels)

# Print the extracted labels
#print(len(train_labels))  



np.save('C1_TrainLabelsPredict_500',train_labels_C1)








# EXTRACTING TEST LABEL DATA 

from CASE9 import Case_9 
      
# a for loop to extract the vb wear for every 250 rows in case 1 data     

Case_9_VB = Case_9.VB

test_labelsC9 = []

#reshape train labels to obtain 
Case_9_VB = Case_9_VB.values.reshape(-1)
Case_9VB = Case_9_VB.reshape(-1,1)

Case_9VB = np.asfarray(Case_9VB)
    

#define the step size for samples
train_size = 500

'''
# Use a for loop to iterate through the values and extract labels
for i in range(0, len(Case_9VB), train_size):
    label = Case_9VB[i]
    test_labels.append(label)

# Convert the list to a NumPy array if needed
test_labels_C9 = np.array(test_labels)

# Print the extracted labels
#print(test_labels) 
'''
# Repeat the loop 6 times
num_repeats = 1

for _ in range(num_repeats):
    
    # Use a for loop to iterate through the values and extract labels
    for i in range(0, len(Case_9VB), train_size):
        label = Case_9VB[i]
        test_labelsC9.append(label)

# Convert the list to a NumPy array if needed
test_labels_C9 = np.array(test_labelsC9)

np.save('C9_TrainLabelsPredict_500',test_labels_C9)












