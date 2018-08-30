# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 06:43:24 2018

@author: SN PANIGRAHI DIC
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time
import pickle

featureExtractionFlag = True

def bin_spatial(img, size=(32, 32)):
    # Create the feature vector
    features = cv2.resize(img, size).ravel() 
    return features

# Compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features


# Extract features from a list of images using bin_spatial(), color_hist() and get_hog_features()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Car car and non-car images from Udacity
cars = glob.glob('./data/vehicles/vehicles/**/*.png')
notcars = glob.glob('./data/non-vehicles/non-vehicles/**/*.png')
#test_images = glob.glob('./test_images/*.jpg')

# Car car and non-car images from Bhubaneswar
# =============================================================================
# cars = glob.glob('./data/BBSR/vehicles/vehicles/**/*.png')
# notcars = glob.glob('./data/BBSR/non-vehicles/non-vehicles/**/*.png')
# =============================================================================
#test_images = glob.glob('./BBSR/test_images/*.jpg')

# Return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

# n_cars = number of car images
# n_notcars = number of non-car images
# image_shape = shape of all the standardized images
# data_type = data type of the images in the repository
data_info = data_look(cars, notcars)

# Pick random image
car_ind = np.random.randint(0, data_info["n_cars"])
notcar_ind = np.random.randint(0, data_info["n_notcars"])
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])


# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.savefig('./output_images/data_exploration.jpg')


# Define HOG parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9     #9
pix_per_cell = 4      # 8
cell_per_block = 4    #2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"   # All
spatial_size = (32, 32) # Spatial binning dimensions (16,16)
hist_bins = 64   # Number of histogram bins 128
spatial_feat = True # Spatial features on or off true
hist_feat = True # Histogram features on or off true
hog_feat = True # HOG features on or off true
y_start_stop = [400, 700] # Min and max in y to search in slide_window()  (400,700)


if featureExtractionFlag == True:
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        car_ind = np.random.randint(0, data_info["n_cars"])
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.savefig('./output_images/all_features_of_carimage.jpg')
        
        obj = {
        "color_space": color_space,
        "scaler": X_scaler,
        "orient" : orient,
        "pix_per_cell": pix_per_cell,
        "cell_per_block": cell_per_block,
        "spatial_size": spatial_size,
        "hist_bins": hist_bins,       
        "y_start_stop": y_start_stop,
        "car_features": car_features,
        "notcar_features": notcar_features
        }
        pickle.dump(obj, open('SpatHistHog_feature_set_KITTI.p', 'wb'))
    else: 
        print('Your function only returns empty feature vectors...')


car_ind = np.random.randint(0, data_info["n_cars"])
notcar_ind = np.random.randint(0, data_info["n_notcars"])
 
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])
 
car_feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
notcar_feature_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YCrCb)
 
car_features_ch0, car_hog_image_ch0 = get_hog_features(car_feature_image[:,:,0],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
car_features_ch1, car_hog_image_ch1 = get_hog_features(car_feature_image[:,:,1],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
car_features_ch2, car_hog_image_ch2 = get_hog_features(car_feature_image[:,:,2],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
 
notcar_features_ch0, notcar_hog_image_ch0 = get_hog_features(notcar_feature_image[:,:,0],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
notcar_features_ch1, notcar_hog_image_ch1 = get_hog_features(notcar_feature_image[:,:,1],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
notcar_features_ch2, notcar_hog_image_ch2 = get_hog_features(notcar_feature_image[:,:,2],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
 
# Plot an example of raw and scaled features
fig = plt.figure(figsize=(12,9))
plt.subplot(241)
plt.imshow(car_image)
plt.title('Car Image')
plt.subplot(242)
plt.imshow(car_hog_image_ch0,cmap='gray')
plt.title('Car CH-0 Hog')
plt.subplot(243)
plt.imshow(car_hog_image_ch1,cmap='gray')
plt.title('Car CH-1 Hog')
plt.subplot(244)
plt.imshow(car_hog_image_ch2,cmap='gray')
plt.title('Car CH-2 Hog')

plt.subplot(245)
plt.imshow(notcar_image, cmap='gray')
plt.title('not-Car Image')
plt.subplot(246)
plt.imshow(notcar_hog_image_ch0,cmap='gray')
plt.title('not-Car CH-0 Hog')
plt.subplot(247)
plt.imshow(notcar_hog_image_ch1,cmap='gray')
plt.title('not-Car CH-1 Hog')
plt.subplot(248)
plt.imshow(notcar_hog_image_ch2,cmap='gray')
plt.title('not-Car CH-2 Hog')

fig.tight_layout()
plt.savefig('./output_images/HOG_features_of_car_image.jpg')

plt.show()