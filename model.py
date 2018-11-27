import pandas as pd
import os
import cv2
import numpy as np
import re


#Read the data from csv using pandas
data=pd.read_csv('./data/driving_log.csv')
print(data.shape)

#Combine all the three cameras file path and measurement angles
data_images=data['center'].append(data['left']).append(data['right'])
measurement=data['steering'].append(data['steering']+0.2).append(data['steering']-0.2)


#Do a test train split for training and validation
from sklearn.model_selection import train_test_split
samples = list(zip(data_images, measurement))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(np.array(train_samples).shape)
print(np.array(validation_samples).shape)

import sklearn

#Generator Block
def generator(samples, batch_size=32):

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread('./data/'+imagePath.strip())
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#Model Architecture 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Creates nVidia Autonomous Car Group model
#This is nvidia inspired model
model=Sequential()
model.add(Lambda(lambda x:x /255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
model.add(Dense(1164,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


#Model is complied and saved
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')
print('Model Saved')


	
