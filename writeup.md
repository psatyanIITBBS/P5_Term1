## P5_Term1_CarND
---

**Vehicle Detection Project**

The goals/steps of this project are the following:

* A Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and training of a classifier (Linear SVM classifier)
* Appended binned color features, as well as histograms of color features to the HOG feature vector with proper color space transformation. 
* Normalization of the features and randomized selection of training and testing set.
* Implementation of a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_features_of_car_image.jpg
[image3]: ./examples/all_features_of_carimage.jpg
[image4]: ./output_images/windowSliding.jpg
[image5]: ./output_images/test1_tracked.jpg
[image6]: ./output_images/test2_tracked.jpg
[image7]: ./output_images/test3_tracked.png
[image8]: ./output_images/test4_tracked.png
[image9]: ./output_images/HeatMap.jpg.png
[video1]: ./project_video_marked.mp4

### Here, I have considered the [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points individually and describeed how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.    

This writeup provides the details of the project implementation.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

A separate code ("DataExploration.py") is first written for exploring the dataset. This code reads in all the `vehicle` and `non-vehicle` images (Line#110-111).  Then many random pairs were plotted and investigated (Line#144-160). Here is an example of one such pair of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then the HOG features of the dataset images were first studied by plotting out many such images both from the "car" and the "non-car" types (Line#231-271) from all the three channels. Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(4, 4)` and `cells_per_block=(4, 4)`:

![alt text][image2]

The code has a flag ('featureExtractionFlag = False') for activating the feature extraction process with a particular set of parameters. Once the parameters were fixed by the above experimnents on the HOG parameters, the flag was set to 'True' and the code was run. Once the code is run with this flag set to 'True', it saves all the parameters and features for both the classes as a pickled file "SpatHistHog_feature_set_KITTI.p" for future use in training a classifier.

#### 2. Explain how you settled on your final choice of HOG parameters.

Different color spaces (RGB, HSV, LUV, HLS, YUV, YCrCb) and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) were tried out and images were grabbed randomly from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. 

The final chosen colorspace is the `YCrCb` color space and the final HOG parameters chosen are: `orientations=9`, `pixels_per_cell=(4, 4)` and `cells_per_block=(4, 4)`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The feature space saved in the earlier step as a pickled file ("SpatHistHog_feature_set_KITTI.p") is now unpickled (Line#25) in another code "training.py". A linear SVM (`from sklearn.svm import LinearSVC`) has been chosen to classify the vehicle class and the non vehicle class in this project. The training process is carried out using all available color-space channels (0,1 and 2) along with the HOG features (hog_channel = "ALL"), spatial features (spatial_size = (32, 32)) and color histograms (hist_bins = 64). The feature without and with normalization is shown below:

![alt text][image3]

The total data set has been split up into randomized training and test sets using `from sklearn.model_selection import train_test_split`. The train and test set contain 80% and 20% respectively of the initial data set. In order to improve the classifier before creating the final video a grid search using the tool `from sklearn.model_selection import GridSearchCV`. The parameters tested are C (Penalty parameter C of the error term) and tol (Tolerance for stopping criteria). After using the optimized valuess for these parameters (C=4.6415888336e-05,tol=0.0001), The accuracy of the classifier reached more than 0.99%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This part is implemented in the code `VehicleTracking.py`. The sliding window search has been implemented to search the features of a vehicle within a specified size of window that slides accross the image. Wherever it finds a match after running through the classifier it marks it as a vecle bounding box. An example of one such search is shown below.

![alt text][image4]

However, a constant window size will not be able to capture all the vehicles in a frame. A car may seem bigger if it is nearer in the camera frame and seem smaller as it goes far away in the frame. Consequently, a combination of different scales is implemented to be able to capture all sizes of vehicles.  After many trials i ended up with the scales below:

scales = [1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.5,4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_marked.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (Line#399-403).  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

