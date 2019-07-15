#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
import csv
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

car_images = []
steering_angles = []
with open('../driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for row in reader:
      steering_center = float(row[3])

      # create adjusted steering measurements for the side camera images
      correction = 0.2 # this is a parameter to tune
      steering_left = steering_center + correction
      steering_right = steering_center - correction

      # read in images from center, left and right cameras
      img_center = ndimage.imread(row[0])
      img_left = ndimage.imread(row[0])
      img_right = ndimage.imread(row[2])

      # add images and angles to data set
      car_images.extend([img_center, img_left, img_right])
      steering_angles.extend([steering_center, steering_left, steering_right])

X_train = np.array(car_images)
y_train = np.array(steering_angles)

# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Nvidia example model from project instructions
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
# Add a mild dropout layer to the fully connected section to reduce overfitting
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)

model.save('model-1.h5')

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    epochs=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('training_result.png')
exit()
