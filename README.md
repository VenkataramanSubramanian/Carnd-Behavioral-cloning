# Carnd-Behavioral-cloning
This is the third project of term-1 of Udacity Self driving car nano-degree program

Behavioral Cloning Project

The goals / steps of this project are the following:
	1.Use the simulator to collect data of good driving behavior ( Training Data)
	2.Build, a convolution neural network in Keras that predicts steering angles from images
	3.Train and validate the model with a training and validation set
	4.Test that the model successfully drives around track one without leaving the road
	5.Summarize the results with a written report


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 which shows how the car is running in autonomous mode

1. Training Data Preparation

	I run the simulator about 3 lap mostly around centered to ensure the training data are proper and can be used in model.
	All the three cameras data were used for training the model.
	The model is trained on about 20k images with 4k used for validation
	
2. Build, a convolution neural network in Keras that predicts steering angles from images

	I have build a CNN based on Nvidia model. The Architecture of the model are 
	First the images are normalized.
	Followed by a convolution network of filter size 24 and filter shape (5,5) and stride as 2
	Followed by a max pooling layer of (2,2)
	Followed by a convolution network of filter size 36 and filter shape (5,5) and stride as 2
	Followed by a max pooling layer of (2,2)
	Followed by a convolution network of filter size 48 and filter shape (5,5) and stride as 2
	Followed by a max pooling layer of (2,2)
	Followed by a convolution network of filter size 64 and filter shape (3,3) 
	Followed by a max pooling layer of (2,2)
	Followed by a convolution network of filter size 24 and filter shape (3,3) 
	Followed by a max pooling layer of (2,2)
	Followed by a flattened fully connected layer of size 1164
	Followed by a flattened fully connected layer of size 100
	Followed by a flattened fully connected layer of size 50
	Followed by a flattened fully connected layer of size 10
	Followed by a flattened fully connected layer of size 1
	
	All of the above uses RELU activation.
	Since using a Nvidia model we can be sure there wont be over-fitting in this model.
	There is a fairly large amount of data to avoid over-fitting and no need for dropout layers.
	
3.Train and validate the model with a training and validation set

	The above model is trained and validated with the training data set prepared using test_train_split.
	
4.Test that the model successfully drives around track one without leaving the road

	The Trained model is saved and executed to run in autonomous model with drive.py and that is recorded into a video file video.mp4.
	The video shows the model is performing very well and driving the car in his track mostly around the center.
	
5. Summary

	As per the requirements the model is performing. And all the needed files are attached. 
	As per the requirements the model is performing. And all the needed files are attached. 

	
