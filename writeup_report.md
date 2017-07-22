#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Visualization"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* process.sh processing my simulator data prior to training

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

At first, I must admit that some of my project choices have been derived by the nature of the simulator, and are not the right decisions in a different setup.

When I started recording driving data on the simulator with my keyboard, I noticed right away(just from the driving experience) that driving using the keyboard acts like a step function unlike the real world driving where it's a continuous function.

The steering angle goes from high to 0 very fast(especially with low fps), it's impossible to train on this data(and belive me that I tried), even when I smoothed the data.

I could have bought a joystick, but I always like to use stuff that I've got, so my first choice was to train the model on udacity data with minimal extra self recorded data.

You will see in the next paragraphs the implications of my choice.

####1. An appropriate model architecture has been employed

Many models were tested, Nvidia's model worked the best for me with some small changes.

Nvidia Reference: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

My model consists of a convolution neural network with changing filter sizes, sub sampling sizes and depths.

You can see a summary of the model below.

Data is normalized using a Keras lambda layer for zero mean and small variance.

All layers include ELU activation function to introduce nonlinearity instead of RELU.
I didn't want to change the Nvidia model too much, therefore I didn't add batch normalization.
Since elu has negative values unlike relu, it gets the mean closer to zero which speeds up learning, same as batch normalization just without the extra overhead.

I used xavier initializer for all layers with elu activation function, it had the fastest learning.
*Side note: tt can be a bit problematic with relu activation function since relu doesn't have negative values, so we can expect half the values to be set to 0.

####2. Attempts to reduce overfitting in the model

The model contains static dropout(20%) for the fully connected layers in order to reduce overfitting.

The dropout reduces the overfitting, but the model still overfits a bit since the dataset is very small and it's only recorded from one non diversed track.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Another problem with a small dataset, is that in order to make the training data set as large as possible, I made the validation set split of only 5% of the dataset, but it was still enough to see that the model is not overfitting too much.

####3. Model parameter tuning

The model used an adam optimizer with custom learning rate of 0.0001 with 30 ephocs.

You can check the learning loss in the visualization notebook.

####4. Appropriate training data

As mentioned before, I decided to rely on udacity data with some minimal extra data.

I used a combination of normal and recovery turns(left and right) as the extra data.

See next section for more details.

###Model Architecture and Training Strategy

####1. Solution Design Approach

###Model Architecture

The overall strategy for deriving a model architecture was to learn the right features without noise and that it will generalize well.

As you know, I chose convlutional neural network as it's the best architecture for the job.

My first approach was to treat it as a regression problem as it's the easy way to go.
Later I tried to treat it as a classifcation problem since I thought it will smooth the driving behaviour with the correct amount of bins, didn't work too well, so I abandoned this approach.

I started with a simple convolutional network, moving towards more complex models until settling for Nvidia's model with some small changes.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The first attempts had really low error on both the training set and the validation set, it was overfitting to the first track.

To combat the overfitting, I modified the model and added dropout layers which had significant impact on the learning.
It doesn't fully eliminate the overfitting since the dataset is really small and non diversed.

The process included pre processing the data, augmenting the data and training on it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior all I did was to add more turning/recovery data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The most important tuning is the DATA, all others are minor in comparison.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     

lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]                  

Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0

Here is a visualization of the architecture without the dropout.
![alt text][image1]

The image processing,augmentation and training happens with a generator so no need to fit all the data in memory for training, just a small batch, for me the batch size was 32.

####3. Creation of the Training Set & Training Process

As I stated many times before, I decided to rely on udacity data, and my goal was to drive track 1 with minimal data.

It was obvious from the first training attempt that the data is the most important thing for the model to perform good on the simulator.

Training data was chosen to keep the vehicle driving on the road in a safe manner.

Udacity data had a bias towards the center and it didn't turn as needed, so I needed to add more turning data.

I plotted the center camera steering angles (check Behavioral-Cloning-Visualization.ipynb)
As you can from the visualization, the steering angles are not distributed well and are mostly at 0.

Since I don't have a lot of data, I didn't want to do some sort of histgoram equalization by removing data, but rather add more angles data for better distribution.

First, I took the left and right cameras images as well, which triples the dataset and makes it much more distributed with almost no cost, the angle correction is 20 degrees for each side. (check Behavioral-Cloning-Visualization.ipynb)

Then all images are flipped on x axis, which serves 2 purposes:
1) Make the dataset larger, doubles it
2) Track 1 has bias towards left turns, this cancels this bias.

I plotted the angles distribution after all of this(check Behavioral-Cloning-Visualization.ipynb), and as you can see it's much more distributed and was good enough for me.

I tried to make the recording training data phase fast and non burdensome as I could.
As a lazy person, recording recovery data can be a real hassle, and since the driving is like a step function and the steering angles goes back to 0, I decided to write a bash script to differentiate images with angles and create a new driving log with relative paths to concatenate with udacity driving log.

I did it for 6 times and added all the angles data to udacity data.

1) remove absolute directory prefix from driving log, so we can join it later with udacity relative driving log.

./process.sh  -a remove_prefix_from_file -r data/train6/driving_log.csv -p /data/workspace/sdc/term1/CarND-Behavioral-Cloning-P3/data/train6/

2) create a new driving log with only images with angles and move the images to a new directory, so I will copy only them to the server since it takes ages.

./process.sh -a remove_no_angle_records -d data/train6/

3) scp to aws server

4) join driving logs

cat new_driving_log.csv >> driving_log.csv

After the collection process, I preprocessed the data(check Behavioral-Cloning-Visualization.ipynb)
1) Crop image to remove noise of horizon and engine cover.
2) Resize image to 66x200, same as Nvidia inputs.
3) Blur the image - Guassien blur - remove noise as well.
4) Convert the image to YUV color space - first reason was to have the same color space for model training and driving as they had different color spaces(RGB and BGR), YUV was inspired from Nvidia's paper and it actually worked much better than other color spaces including grayscale.

I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used adam optimizer with a small learning rate + had dropout layers, so I ran the training for 30 epoches.

The last thing I did was to do some sort of moving average on drive.py, only on past steering angles.
The reasoning for it was to smooth the driving experience as it's turning too much from left to right and vice versa.
It didn't have the effect that I wanted, but it still helped with smoothing the driving a bit, I didn't even feel a small lag.
