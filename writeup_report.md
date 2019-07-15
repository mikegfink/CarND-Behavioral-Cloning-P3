# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center.jpg "Center Image"
[image2]: ./recovery1.jpg "Recovery Image"
[image3]: ./recovery2.jpg "Recovery Image"
[image4]: ./recovery3.jpg "Recovery Image"
[image5]: ./recovery4.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* autodrive.mp4 containing a video of the car driving itself around the course
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the Nvidia network presented in the lessons. The model consists of three convolutional layers with 5x5filters and 2x2 strides. The first layer has 24 filters, the second 36, and the third 48. These are followed by two unstrided layers with 3x3 filter sizes and depths of 64 (model.py lines 42-46).

The convolutional layers are followed by 4 fully connected layers with dimensions of 100, 50, 20, 10 and 1.

The each convolutional layers is activated with RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 38). 

#### 2. Attempts to reduce overfitting in the model

The model contains one mild dropout layer in order to reduce overfitting (model.py lines 50). I experimented with dropout layers in various positions in the network. In each case, the dropout layers reduced both the training and validation accuracy and increased the difference between the training and validation accuracy. In the end, the best results were without including dropout layers. The best solution for reducing overfitting was increasing the volume of training data, particularly in difficult parts of the track. 

The model was trained and validated on different data sets to ensure that the model was not overfitting by randomly separating 20% of the data for validation (code line 56). I created a training set with the second course, but found that it was not necessary for driving the first course. The variability of the second course increased the validation error rate significantly and I did not have time to generate enough detailed sets for the second course. Despite the validation accuracy, this model was also able to drive the first course and the model can be found in model-1.h5.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 55).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the course in reverse and repeating difficult sections of the course. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make use of existing architectures.

My first step was to use a convolution neural network model similar to the Nvidia example network presented in the lesson. I thought this model might be appropriate because it ought to provide functional image recognition and provides a logical path to reduce the inputs to a single output value. Obviously, a network achitecture provided for the purpose of lane and position detection by Nvidia should function for training our system here.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I attmepted to modify the model to include dropout layers to prevent overfitting. Unfortunately, including dropout layers, particularly with high dropout rates (50%) resulted in much lower accuracy and did not reduce the difference between the training and validation accuracy.

Then I decided that the best approach was to increase the corpus of quality data. This primarily involved recording recovery scenarios by aiming the car in a dangerous direction and recording the steering angle returning the care to the center of the lane. Doing this on difficult parts of the track was most effective, as well as driving the track in a clockwise direction. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 37-55) consisted of a convolution neural network with the following layers and layer sizes:
* convolutional - 24, 5x5 filters - 2x2 strides - relu activation
* convolutional - 36, 5x5 filters - 2x2 strides - relu activation
* convolutional - 48, 5x5 filters - 2x2 strides - relu activation
* convolutional - 64, 3x3 filters - relu activation
* convolutional - 48, 3x3 filters - relu activation
* flattener layer
* fully connected layer - 100 nodes
* dropout layer - 10% drop rate
* fully connected layer - 50 nodes
* fully connected layer - 10 nodes
* fully connected layer - 1 nodes

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid road boundaries and stay on the drivable surface. These images show what a recovery looks like starting from facing the road boundary at a corner, to a more cogent trajectory in the center of the road:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would prevent overfitting to left turns since the track is predominately left turns.

After the collection process, I had around 20,000 data points. I then preprocessed this data by normalizing and cropping the images via a lambda function and cropping Keras layer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that the validation accuracy would increase after this point. I used an adam optimizer so that manually training the learning rate wasn't necessary.
