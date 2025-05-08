#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:15:27 2023

@author: toluojo
"""
# CONVERTING THE CASE 1 DATA INTO MTF - FOR TRAINING,  keep the images in colour 


import pandas as pd 
import numpy as np 
from pyts.image import MarkovTransitionField
from PIL import Image 
import os 
import matplotlib.pyplot as plt 
from skimage.transform import resize

'''
#6 sensors
# Load the DataFrame
Case_1 = pd.read_csv('Mill_Case1_gma.csv')


# Select the desired columns
selected_columns = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_Table', 'AE_Spindle']
selected_data = Case_1[selected_columns]

# Define sample size
sample_size = 250

# Initialize lists to store resized MTFs
resized_MTFs = []

# Initialize MarkovTransitionField object
MTF = MarkovTransitionField(n_bins=30)

# Resize dimensions
target_shape = (224, 224)

# Define sensor signal groupings
group1 = ['smcAC', 'smcDC']
group2 = ['vib_table', 'vib_spindle']
group3 = ['AE_Table', 'AE_Spindle']

# Iterate over the selected data
for group in [group1, group2, group3]:
    stacked_signals = []
    for col in group:
        # Reshape the data to 2D
        data_2d = selected_data[col].values.reshape(-1, 1)

        # Split the data into samples
        samples = [data_2d[i:i+sample_size] for i in range(0, len(data_2d), sample_size)]

        # Calculate MTF for each sample and store it
        MTFs_col = [MTF.fit_transform(sample.flatten().reshape(1, -1))[0] for sample in samples]

        # Resize each MTF to the target shape
        resized_MTFs_col = [resize(mtf, target_shape) for mtf in MTFs_col]

        # Stack along the first axis (time) to create a single channel
        stacked_signals.append(np.stack(resized_MTFs_col, axis=0))

    # Stack the signals along the last axis (channels) to create a single channel
    combined_channel = np.stack(stacked_signals, axis=-1)
    resized_MTFs.append(combined_channel)

# Combine the resized MTFs into a single array
combined_MTFs = np.stack(resized_MTFs, axis=-1)  # Stack along the last axis (channels)

# Ensure we have at most 612 MTFs
combined_MTFs = combined_MTFs[:612]

# Save the array to a file
np.save('MTFs_array_condensed_6sensors_C1.npy', combined_MTFs)
'''





# to generate threechannels need to concatenate three sets of arrays and then reshape it after 

# Load the DataFrame
Case_1 = pd.read_csv('Mill_Case1_gma.csv')

# Select the desired columns
selected_columns = ['smcAC', 'smcDC', 'vib_table']
selected_data = Case_1[selected_columns]

# Define sample size
sample_size = 500

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

# Save the array to a file
np.save('MTFs_array_C1_500.npy', combined_MTFs)






































'''
# Load the DataFrame
Case_1 = pd.read_csv('Mill_Case1_gma.csv')

# Select the desired columns
selected_columns = ['smcAC', 'smcDC', 'vib_table']
selected_data = Case_1[selected_columns]

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
    
    # Add the resized MTFs of this column to the list
    resized_MTFs.append(resized_MTFs_col)

# Combine the resized MTFs into a single array
combined_MTFs = np.stack(resized_MTFs, axis=-1)  # Stack along the last axis (channels)

# Ensure we have at most 612 MTFs
combined_MTFs = combined_MTFs[:612]

# Save the array to a file
np.save('MTFs_array.npy', combined_MTFs)

'''





''' same 1836 issue 
# Load the DataFrame
Case_1 = pd.read_csv('Mill_Case1_gma.csv')

# Select the desired columns
selected_columns = ['smcAC', 'smcDC', 'vib_table']
selected_data = Case_1[selected_columns]

# Define sample size
sample_size = 250

# Initialize lists to store MTFs
MTFs = []

# Initialize MarkovTransitionField object
MTF = MarkovTransitionField(n_bins=30)

# Iterate over the selected data
for col in selected_data.columns:
    # Reshape the data to 2D
    data_2d = selected_data[col].values.reshape(-1, 1)

    # Split the data into samples
    samples = [data_2d[i:i+sample_size] for i in range(0, len(data_2d), sample_size)]

    # Calculate MTF for each sample and store it
    MTFs_col = [MTF.fit_transform(sample.flatten().reshape(1, -1)) for sample in samples]
    MTFs_col_array = np.concatenate(MTFs_col)
    MTFs.append(MTFs_col_array)

# Combine the MTFs into a 3D array
MTFs_array = np.stack(MTFs, axis=-1)

# Determine the number of sets of MTFs to keep (612 in your case)
num_sets = min(len(MTFs_array), 612)
MTFs_array = MTFs_array[:num_sets]

# Save the array to a file
np.save('MTFs_array.npy', MTFs_array)
'''




































''' tHIS ONE was producing 1836 arrays instead of 612 , the sensor arrays were just put under each other instead of passed as a channel into 3d array 

# Load the dataset
Case_1 = pd.read_csv('Mill_Case1_gma.csv')

# Combine the required signals into a single dataframe
combined_data = Case_1[['smcAC', 'smcDC', 'vib_table']]

# Define the step size for samples
sample_size = 250

# Initialize MarkovTransitionField
MTF = MarkovTransitionField(n_bins=20)

# Store the Markov Transition Fields
MTF_data = []

# Iterate through the combined data in a single loop
for column in combined_data.columns:
    signal_data = combined_data[column].values
    signal_length = len(signal_data)
    
    # Split the signal into chunks of 250 samples
    for i in range(0, signal_length, sample_size):
        chunk = signal_data[i:i+sample_size] 
        
        # Generate Markov Transition Field
        MTF_chunk = MTF.fit_transform(chunk.reshape(1, -1))
        MTF_data.append(MTF_chunk)

# Combine the Markov Transition Fields of all three signals into a 3-dimensional array
MTF_data = np.array(MTF_data)

# Save the combined array to a file
np.save('combined_MTF_data.npy', MTF_data)

# Load the combined MTF data
MTF_data = np.load('combined_MTF_data.npy')

# Create a directory to store the images
output_directory = 'MTF_images_combined_sensors'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through the MTF data and convert each MTF into an image
for i, MTF_chunk in enumerate(MTF_data):
    # Remove the first dimension (batch size)
    MTF_chunk = MTF_chunk.squeeze()
    
    # Create a heatmap using matplotlib
    plt.imshow(MTF_chunk, cmap='viridis')
    
    # Save the heatmap as an image
    image_path = os.path.join(output_directory, f'MTF_image_{i}.png')
    plt.savefig(image_path, format='png')
    plt.close()

'''









'''useful 03.03.24
#computing a dataset of Markov Transition Field 

# Load the dataset
Case_1 = pd.read_csv('Mill_Case1_gma.csv')

# Combine the required signals into a single dataframe
combined_data = Case_1[['smcAC', 'smcDC', 'vib_table']]

# Define the step size for samples
sample_size = 250

# Initialize MarkovTransitionField
MTF = MarkovTransitionField(n_bins=10)

# Store the Markov Transition Fields
MTF_data = []

# Iterate through the combined data in a single loop
for column in combined_data.columns:
    signal_data = combined_data[column].values
    signal_length = len(signal_data)
    
    # Split the signal into chunks of 250 samples
    for i in range(0, signal_length, sample_size):
        chunk = signal_data[i:i+sample_size]
        
        # Generate Markov Transition Field
        MTF_chunk = MTF.fit_transform(chunk.reshape(1, -1))
        MTF_data.append(MTF_chunk)

# Combine the Markov Transition Fields of all three signals into a 3-dimensional array
MTF_data = np.array(MTF_data)

# Save the combined array to a file
np.save('combined_MTF_data.npy', MTF_data)
'''





'''usefufl 3/03/24

#8/12/23
Case_1 = pd.read_csv('Mill_Case1_gma.csv')
C1smcAC = Case_1.smcAC

# reshape the dataframe to a 2d dataframe 
C1smcAC = C1smcAC.values.reshape(-1)
C1smcAC_2D = C1smcAC.reshape(-1,1)

# convert case 1 dataframe into a matrix/ array 
C1smcAC_2D = np.asfarray(C1smcAC_2D)
print(C1smcAC_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1smcAC_2D[i:i+sample_size] for i in range(0,len(C1smcAC_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)



#storing the file paths
MTF_images_paths_SMCAC = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
# instead of converting the markov transiiton field straight into images , converting values into heat maps then uploading the heatmaps in coour to the files 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)


    # Create a heatmap using matplotlib
    plt.imshow(MTF_sample[0], cmap='viridis')

    # Save the heatmap as an RGB image
    heatmap_path = f'MTF_images/heatmap_SMCAC{i}.png'
    plt.savefig(heatmap_path, format='png')
    plt.close()

    # Convert the heatmap to an RGB image
    img = Image.open(heatmap_path).convert('RGB')

    # Resize the image to 224 by 224
    img_resized = img.resize((224, 224))

    # Define the output folder
    output_folder = 'MTF_images_resized_SMCAC'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the resized image with a unique filename
    image_filename = f'{output_folder}/MTF_image_SMCAC{i}.png'
    img_resized.save(image_filename, 'PNG')

    # Append the file path to the list
    MTF_images_paths_SMCAC.append(image_filename)
    
    # Remove the temporary heatmap file to save disk space 
    os.remove(heatmap_path)
 
    
'''




    

''' USEFUL 02/03/24 

#SPINDLE MOTOR DIRECT CURRENT 
   
# Do the same for spindle motor direct current   - smcDC - but not displaying MTF filed 
C1smcDC = Case_1.smcDC

#reshape the dataframe to a 2d dataframe 
C1smcDC = C1smcDC.values.reshape(-1)
C1smcDC_2D = C1smcDC.reshape(-1,1)


C1smcDC_2D = np.asfarray(C1smcDC_2D)
print(C1smcDC_2D.shape)


#define the step size for samples
sample_size = 250


# splitting the case 1 dataframe into the different samples 
samples = [C1smcDC_2D[i:i+sample_size] for i in range(0,len(C1smcDC_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)

#storing the file paths
MTF_images_paths_SMCDC = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    
    # Create a heatmap using matplotlib
    plt.imshow(MTF_sample[0], cmap='viridis')

    # Save the heatmap as an RGB image
    heatmap_path = f'MTF_images/heatmap_SMCDC{i}.png'
    plt.savefig(heatmap_path, format='png')
    plt.close()

    # Convert the heatmap to an RGB image
    img = Image.open(heatmap_path).convert('RGB')

    # Resize the image to 224 by 224
    img_resized = img.resize((224, 224))

    # Define the output folder
    output_folder = 'MTF_images_resized'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the resized image with a unique filename
    image_filename = f'{output_folder}/MTF_image_SMCDC{i}.png'
    img_resized.save(image_filename, 'PNG')

    # Append the file path to the list
    MTF_images_paths_SMCDC.append(image_filename)


    # append a label for the tool wear to a separate file 

    #need to have defined what values 


    # Remove the temporary heatmap file to save disk space 
    #os.remove(heatmap_path)
''' 
    
 
'''
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250) 
    
    #convert image to colour rgb 
    #img1 = img.convert('RGB')

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_SMDC{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_SMDC.append(image_filename)
'''

  
    
  
    
  
''' USEFUL -02/03/24    
  
#TABLE VIBRATION 
    
#now adding vib_table     
C1vibt = Case_1.vib_table

#reshape the dataframe to a 2d dataframe 
C1vibt  = C1vibt.values.reshape(-1)
C1vibt_2D = C1vibt.reshape(-1,1)


C1vibt_2D = np.asfarray(C1vibt_2D)
print(C1vibt_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1vibt_2D[i:i+sample_size] for i in range(0,len(C1vibt_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)

#storing the file paths
MTF_images_paths_VIB_T = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    
    # Create a heatmap using matplotlib
    plt.imshow(MTF_sample[0], cmap='viridis')

    # Save the heatmap as an RGB image
    heatmap_path = f'MTF_images/heatmap_VIB_T{i}.png'
    plt.savefig(heatmap_path, format='png')
    plt.close()

    # Convert the heatmap to an RGB image
    img = Image.open(heatmap_path).convert('RGB')

    # Resize the image to 224 by 224
    img_resized = img.resize((224, 224))

    # Define the output folder
    output_folder = 'MTF_images_resized'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the resized image with a unique filename
    image_filename = f'{output_folder}/MTF_image_VIB_T{i}.png'
    img_resized.save(image_filename, 'PNG')

    # Append the file path to the list
    MTF_images_paths_VIB_T.append(image_filename)

    # Remove the temporary heatmap file to save disk space 
    #os.remove(heatmap_path)
''' 
    
    
'''
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_VIB_T{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_VIB_T.append(image_filename)
'''










''' USEFUL - 02/03/24     
    
#SPINDLE VIBRATION 
        
C1vibsp = Case_1.vib_spindle

#reshape the dataframe to a 2d dataframe 
C1vibsp = C1vibsp.values.reshape(-1)
C1vibsp_2D = C1vibsp.reshape(-1,1)


C1vibsp_2D = np.asfarray(C1vibsp_2D)
print(C1vibsp_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1vibsp_2D[i:i+sample_size] for i in range(0,len(C1vibsp_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)


    
#storing the file paths

MTF_images_paths_VIB_S = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    
     
    # Create a heatmap using matplotlib
    plt.imshow(MTF_sample[0], cmap='viridis')

    # Save the heatmap as an RGB image
    heatmap_path = f'MTF_images/heatmap_VIB_S{i}.png'
    plt.savefig(heatmap_path, format='png')
    plt.close()

    # Convert the heatmap to an RGB image
    img = Image.open(heatmap_path).convert('RGB')

    # Resize the image to 224 by 224
    img_resized = img.resize((224, 224))

    # Define the output folder
    output_folder = 'MTF_images_resized'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the resized image with a unique filename
    image_filename = f'{output_folder}/MTF_image_VIB_S{i}.png'
    img_resized.save(image_filename, 'PNG')

    # Append the file path to the list
    MTF_images_paths_VIB_S.append(image_filename)

    # Remove the temporary heatmap file to save disk space 
    #os.remove(heatmap_path)
    
'''
   
'''
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_VIB_S{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_VIB_S.append(image_filename)
'''
    
    
    
    
    
    
'''USEFUL - 02.03.24 

#TABLE ACOUSTIC EMISSION 


#now adding vib_table     
C1AE_T = Case_1.AE_Table 

#reshape the dataframe to a 2d dataframe 
C1AE_T  = C1AE_T.values.reshape(-1)
C1AE_T_2D = C1AE_T.reshape(-1,1)


C1AE_T_2D = np.asfarray(C1AE_T_2D)
print(C1AE_T_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1AE_T_2D[i:i+sample_size] for i in range(0,len(C1AE_T_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)


#storing the file paths

MTF_images_paths_AE_T = []


# for loop to append each markov transition field to a list 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    
     
    # Create a heatmap using matplotlib
    plt.imshow(MTF_sample[0], cmap='viridis')

    # Save the heatmap as an RGB image
    heatmap_path = f'MTF_images/heatmap_AE_T{i}.png'
    plt.savefig(heatmap_path, format='png')
    plt.close()

    # Convert the heatmap to an RGB image
    img = Image.open(heatmap_path).convert('RGB')

    # Resize the image to 224 by 224
    img_resized = img.resize((224, 224))

    # Define the output folder
    output_folder = 'MTF_images_resized'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the resized image with a unique filename
    image_filename = f'{output_folder}/MTF_image_AE_T{i}.png'
    img_resized.save(image_filename, 'PNG')

    # Append the file path to the list
    MTF_images_paths_AE_T.append(image_filename)

    # Remove the temporary heatmap file to save disk space 
    #os.remove(heatmap_path)
''' 



    
'''
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_AE_T{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_AE_T.append(image_filename)
'''




''' USEFUL - 02/03/24
#SPINDLE ACOUSTIC EMISSION 
    
#Updating the code to save the images and store their file paths 
    
#Spindle Acoustic Emission 

#Now adding vib_table     
C1AE_S = Case_1.AE_Spindle 

#reshape the dataframe to a 2D dataframe 
C1AE_S  = C1AE_S.values.reshape(-1)
C1AE_S_2D = C1AE_S.reshape(-1,1)


C1AE_S_2D = np.asfarray(C1AE_S_2D)
print(C1AE_S_2D.shape)


#define the step size for samples
sample_size = 250

#splitting the case 1 dataframe into the different samples 
samples = [C1AE_S_2D[i:i+sample_size] for i in range(0,len(C1AE_S_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)

#storing the file paths

MTF_images_paths_AE_S = []

# for loop to append each markov transition field to a list 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    
     
    # Create a heatmap using matplotlib
    plt.imshow(MTF_sample[0], cmap='viridis')

    # Save the heatmap as an RGB image
    heatmap_path = f'MTF_images/heatmap_AE_S{i}.png'
    plt.savefig(heatmap_path, format='png')
    plt.close()

    # Convert the heatmap to an RGB image
    img = Image.open(heatmap_path).convert('RGB')

    # Resize the image to 224 by 224
    img_resized = img.resize((224, 224))

    #Define the output folder
    output_folder = 'MTF_images_resized'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Save the resized image with a unique filename
    image_filename = f'{output_folder}/MTF_image_AE_S{i}.png'
    img_resized.save(image_filename, 'PNG')

    #Append the file path to the list
    MTF_images_paths_AE_S.append(image_filename)

    # Remove the temporary heatmap file to save disk space 
    os.remove(heatmap_path)
    
'''

























    
    
'''
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths.append(image_filename)
'''
 


 
''' 
 # EXTRACTING TRAINING DATA 
       
 # a for loop to extract the vb wear for every 250 rows in case 1 data     

 Case_1_VB = Case_1.VB

 train_labels = []

 #reshape train labels to obtain 
 Case_1_VB = Case_1_VB.values.reshape(-1)
 Case_1VB = Case_1_VB.reshape(-1,1)

 Case_1VB = np.asfarray(Case_1VB)
     

 #define the step size for samples
 train_size = 250 

 # Repeat the loop 6 times
 num_repeats = 6

 for _ in range(num_repeats):
     
     # Use a for loop to iterate through the values and extract labels
     for i in range(0, len(Case_1VB), train_size):
         label = Case_1VB[i]
         train_labels.append(label)

 # Convert the list to a NumPy array if needed
 train_labels = np.array(train_labels)

 # Print the extracted labels
 print(len(train_labels))  
     
    
'''  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


