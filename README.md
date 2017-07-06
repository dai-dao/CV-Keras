# AIND Term II, Computer Vision Capstone Project 
# Facial Keypoint Detection and Real-time Filtering

## My Results

![Alt Text](results/overlay.gif)
![Alt Text](results/keypoints.gif)

## Project Overview

Welcome to the Computer Vision capstone project in the AI Nanodegree program! In this project, you’ll combine your knowledge of computer vision techniques and deep learning to build and end-to-end facial keypoint recognition system. Facial keypoints include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. Your completed code should be able to take in any image containing faces and identify the location of each face and their facial keypoints, as shown below.

### Environment

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/udacity/AIND-CV-FacialKeypoints.git
cd AIND-CV-FacialKeypoints
```

2. Create (and activate) a new environment with Python 3.5 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name aind-cv python=3.5 numpy
	source activate aind-cv
	```
	- __Windows__: 
	```
	conda create --name aind-cv python=3.5 numpy scipy
	activate aind-cv
	```

3. Install TensorFlow.
```
pip install tensorflow-gpu
```

4. Install a few required pip packages (including OpenCV).
```
pip install -r requirements.txt
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```

## Project Rubric

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Submission Files      |  `CV_project.ipynb`--> all completed python functions requested in the main notebook `CV_project.ipynb` **TODO** items should be completed.		|


#### Step 1:  Add eye detections to the face detection setup
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Add eye detections to the current face detection setup. |  The submission returns proper code detecting and marking eyes in the given test image. |


#### Step 2: De-noise an image for better face detection

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| De-noise an image for better face detection.  |  The submission completes de-noising of the given noisy test image with perfect face detections then performed on the cleaned image. |


#### Step 3: Blur and edge detect an image

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Blur and edge detect a test image.  | The submission returns an edge-detected image that has first been blurred, then edge-detected, using the specified parameters. |


#### Step 4: Automatically hide the identity of a person (blur a face)

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Find and blur the face of an individual in a test image. |  The submission should provide code to automatically detect the face of a person in a test image, then blur their face to mask their identity.  |


#### Step 5:  Specify the network architecture
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Specify a convolutional network architecture for learning correspondence between input faces and facial keypoints. | The submission successfully provides code to build an appropriate convolutional network. |


#### Step 6:  Compile and train the model
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Compile and train your convnet.| The submission successfully compiles and trains their convnet.  |


#### Step 7:  Answer a few questions and visualize the loss
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Answer a few questions about your training and visualize the loss function.| The submission successfully discusses any potential issues with their training, and answers all of the provided questions.  |


#### Step 8:  Complete a facial keypoints detector and complete the CV pipeline
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Combine OpenCV face detection with your trained convnet facial keypoint detector. | The submission successfully combines OpenCV's face detection with their trained convnet keypoint detector. |
