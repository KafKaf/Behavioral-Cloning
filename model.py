from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import csv
import cv2
import sklearn

batch_size = 32
epochs = 30
angle_correction = 0.2
driving_log = "data/driving_log.csv"
data_directory = "data/"

def pre_process_image(image):
    # Crop image to remove noise(horizon + engie cover) Y1:Y2,X1:X2
    cropped_image  = image[70:135,10:350]
    # Resize same as nvidia
    resized_image = cv2.resize(cropped_image, (200, 66), interpolation=cv2.INTER_AREA)
    # Blur the image a bit, bilateralFilter is too slow
    blurred_image = cv2.GaussianBlur(resized_image, (5,5), 0)
    # Convert to YUV color space like nvidia
    final_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2YUV)
    return final_image

samples = []
with open(driving_log, 'r') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# remove header
del samples[0]

train_samples, validation_samples = train_test_split(samples, test_size=0.05)


def generator(samples, batch_size=32):
	num_samples = len(samples)
	
	while 1: # Loop forever so the generator never terminates
		samples = shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				# read in images from center, left and right cameras
				center_image = pre_process_image(cv2.imread(data_directory + batch_sample[0].strip()))
				left_image = pre_process_image(cv2.imread(data_directory + batch_sample[1].strip()))
				right_image = pre_process_image(cv2.imread(data_directory + batch_sample[2].strip()))

				# create adjusted steering measurements for the side camera images
				center_angle = float(batch_sample[3].strip())	
				left_angle = center_angle + angle_correction
				right_angle = center_angle - angle_correction

				# add images and angles to data set
				images.append(center_image)
				images.append(left_image)
				images.append(right_image)
				angles.append(center_angle)
				angles.append(left_angle)
				angles.append(right_angle)

			images = np.array(images)
			angels = np.array(angles)

			# flip images and angles
			images_flipped = np.fliplr(images)
			angles_flipped = [-x for x in angles]
                        
                        # join all lists
			X_train = np.concatenate((images, images_flipped), axis=0)
			y_train = np.concatenate((angles, angles_flipped), axis=0)
			
                        yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(66, 200, 3),
        output_shape=(66, 200, 3)))

# Convlutional network
model.add(Convolution2D(24,5,5,border_mode="valid",subsample=(2,2),init='glorot_normal',activation='elu'))
model.add(Convolution2D(36,5,5,border_mode="valid",subsample=(2,2),init='glorot_normal',activation='elu'))
model.add(Convolution2D(48,5,5,border_mode="valid",subsample=(2,2),init='glorot_normal',activation='elu'))
model.add(Convolution2D(64,3,3,border_mode="valid",init='glorot_normal',activation='elu'))
model.add(Convolution2D(64,3,3,border_mode="valid",init='glorot_normal',activation='elu'))
model.add(Flatten())
model.add(Dense(100,init='glorot_normal',activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(50,init='glorot_normal',activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(10,init='glorot_normal',activation='elu'))
model.add(Dropout(0.4))
model.add(Dense(1,init='glorot_normal'))
model.compile(loss='mse', optimizer=Adam(lr=1e-4))
history_object = model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples), validation_data=validation_generator, nb_val_samples=6*len(validation_samples), nb_epoch=epochs)
model.save('model.h5')

# Print training and validation losses for future visualization
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
