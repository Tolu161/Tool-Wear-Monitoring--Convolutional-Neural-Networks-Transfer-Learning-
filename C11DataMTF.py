# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:53:46 2024

@author: egyto1
"""

# CASE 11 

import scipy.io as spio

import pandas as pd

import numpy as np 
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image 
import os 

#extracting case 2 data 

#dataset is loaded into varible 
mill_dataset = spio.loadmat('mill.mat')

#access the 3d array from loaded data
data = mill_dataset['mill']



# Create an array datalist 
dataList = []


# CASE 11 Prepare 

dataList.append(data[0][71:94])    # - case 11

#create an empty list to store the tempdataframe dictionary 

temp_dataframe =[]
caseSamples = []

#the number of columns 
# to loop through each case loop through integer i in variable  case-i-data and append data to temporary dataframe 

    
    # for loop to loop through index CaseData[j] from 0 which is run 1 to maximum run for each case, determined by the length of each case 
for j in range(len(dataList)): 
        print(j)
        # looping through each case of the dataset case 1 to case 16
        caseSamples = dataList[j]
        print(len(caseSamples))
        #loop through the rows in the casesamples 
        for row in range(len(caseSamples)): 
            
            #print(row)
            # loop through the samples the values that have 9000 for smcAC to AESpindle  , extracting the sample data 
            for sample in range(9000): 
                
                #print(sample)
                # now append the array into the temporary dataframe. 
                 
                temp_dataframe.append([caseSamples[row][0][0], caseSamples[row][1][0], caseSamples[row][2][0], caseSamples[row][3][0], caseSamples[row][4][0], caseSamples[row][5][0], caseSamples[row][6][0], caseSamples[row][7][sample], caseSamples[row][8][sample], caseSamples[row][9][sample], caseSamples[row][10][sample], caseSamples[row][11][sample], caseSamples[row][12][sample]])
                

# create an empty dataframe with the columns  : 
print(len(temp_dataframe))
#use this one 
mill_df = pd.DataFrame(temp_dataframe, columns = ['case', 'run', 'VB', 'time', 'DOC', 'feed', 'material', 'smcAC','smcDC','vib_table','vib_spindle', 'AE_Table', 'AE_Spindle' ] )
print(len(mill_df))


#REMOVING NULL VALUES 

print(len(mill_df.VB))

for i in range(1, len(mill_df.VB)):
    
    value = mill_df.VB[i]
    
 
    
    previous_value = mill_df.VB[i - 1]

    if pd.isnull(value):
       
        
        if not pd.isnull(previous_value):
                
            mill_df.loc[i,'VB'] = previous_value
            

# exporting the dataframe to a csv : - no longer needed 
Case_11= mill_df

#case 9 with three sensors

# to generate threechannels need to concatenate three sets of arrays and then reshape it after 


# Select the desired columns
selected_columns = ['smcAC', 'smcDC', 'vib_table']
selected_data = Case_11[selected_columns]

# Define sample size
sample_size = 250

# Initialize lists to store resized MTFs
resized_MTFs = []

# Initialize MarkovTransitionField object
MTF = MarkovTransitionField(n_bins=30)

# Resize dimensions
target_shape = (224, 224)

# Iterate over the selected data
for col in selected_data.columns:
    # Reshape the data to 2D
    data_2d = selected_data[col].values.reshape(-1, 1)

    # Split the data into samples
    samples = [data_2d[i:i+sample_size] for i in range(0, len(data_2d), sample_size)]

    # Calculate MTF for each sample and store it
    MTFs_col = [MTF.fit_transform(sample.flatten().reshape(1, -1))[0] for sample in samples]

    # Resize each MTF to the target shape
    resized_MTFs_col = [resize(mtf, target_shape) for mtf in MTFs_col]
    
    # Stack along the first axis (time) to create a single channel
    stacked_MTFs_col = np.stack(resized_MTFs_col, axis=0)
    
    # Add the resized MTFs of this column to the list
    resized_MTFs.append(stacked_MTFs_col)

# Combine the resized MTFs into a single array
combined_MTFs = np.stack(resized_MTFs, axis=-1)  # Stack along the last axis (channels)

# Ensure we have at most 612 MTFs
#combined_MTFs = combined_MTFs[:612]

# Save the array to a file
np.save('MTFs_array_C11_250.npy', combined_MTFs)








