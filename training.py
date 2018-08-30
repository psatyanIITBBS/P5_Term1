# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:10:36 2018

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

estimateParam = False

feature_pickle = pickle.load( open("SpatHistHog_feature_set_KITTI.p", "rb" ) )

car_features = feature_pickle["car_features"]
notcar_features = feature_pickle["notcar_features"]
color_space = feature_pickle["color_space"]
X_scaler = feature_pickle["scaler"]
orient = feature_pickle["orient"]
pix_per_cell = feature_pickle["pix_per_cell"]
cell_per_block = feature_pickle["cell_per_block"]
spatial_size = feature_pickle["spatial_size"]
hist_bins = feature_pickle["hist_bins"]
y_start_stop = feature_pickle["y_start_stop"] # Min and max in y to search in slide_window()

ystart = y_start_stop[0]
ystop = y_start_stop[1]


X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
#X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# =============================================================================
# print('Using:',orient,'orientations',pix_per_cell,
#     'pixels per cell and', cell_per_block,'cells per block')
# print('Feature vector length:', len(X_train[0]))
# # Use a linear SVC 
# svc = LinearSVC()
# # Check the training time for the SVC
# t=time.time()
# svc.fit(X_train, y_train)
# t2 = time.time()
# print(round(t2-t, 2), 'Seconds to train SVC...')
# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
# t=time.time()
# =============================================================================

svc = LinearSVC()

svc.get_params()

if estimateParam == True:
    
    
    
    Cs = np.logspace(-6, -1, 10)
    tol_list=[0.0001,0.00001]
    
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs,tol=tol_list))
    
    
    clf.fit(X_train, y_train)
    
    print(clf.best_score_)
    
    print(clf.best_estimator_)
    
    
print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC(C=0.00016681005372000591,tol=0.0001)
svc = LinearSVC(C=4.641588833612782e-05,tol=0.0001)
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

obj = {
        "svc": svc,
        "scaler": X_scaler,
        "orient" : orient,
        "pix_per_cell": pix_per_cell,
        "cell_per_block": cell_per_block,
        "spatial_size": spatial_size,
        "hist_bins": hist_bins,  
        "color_space": color_space,
        "y_start_stop": y_start_stop
    }
pickle.dump(obj, open('svc_pickle.p', 'wb'))


