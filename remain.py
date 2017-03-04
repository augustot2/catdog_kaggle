150# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display, Image, HTML
import cv2
os.chdir("/home/augusto/catdog_kaggle")
TRAIN_DIR = '/home/augusto/catdog_kaggle/data/train/'
TEST_DIR = '/home/augusto/catdog_kaggle/data/test/'

# used for scaling/normalization
IMAGE_SIZE = 150; # 150x150.  Also, 224, 96, 64, and 32 are also common
CHANNELS = 3
pixel_depth = 255.0  # Number of levels per pixel.

# for small-sample testing
OUTFILE = '/Users/pal004/Desktop/CatsVsDogsRedux/CatsAndDogs_pal15Jan2017_SmallerTest.npsave.bin'
TRAINING_AND_VALIDATION_SIZE_DOGS = 1000
TRAINING_AND_VALIDATION_SIZE_CATS = 1000
TRAINING_AND_VALIDATION_SIZE_ALL  = 2000
TRAINING_SIZE = 2000  # TRAINING_SIZE + VALID_SIZE must equal TRAINING_AND_VALIDATION_SIZE_ALL
VALID_SIZE = 400
TEST_SIZE_ALL = 500

if (TRAINING_SIZE + VALID_SIZE != TRAINING_AND_VALIDATION_SIZE_ALL):
   print ("Error, check that TRAINING_SIZE+VALID_SIZE is equal to TRAINING_AND_VALIDATION_SIZE_ALL")
   exit ()

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_images = train_dogs[:TRAINING_AND_VALIDATION_SIZE_DOGS] + train_cats[:TRAINING_AND_VALIDATION_SIZE_CATS]
train_labels = np.array ((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + (['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS))
test_images =  test_images[:TEST_SIZE_ALL]
test_labels = np.array (['unknownclass'] * TEST_SIZE_ALL)


# resizes to IMAGE_SIZE/IMAGE_SIZE while keeping aspect ratio the same.  pads on right/bottom as appropriate
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    if (img.shape[0] >= img.shape[1]): # height is greater than width
       resizeto = (IMAGE_SIZE, int (round (IMAGE_SIZE * (float (img.shape[1])  / img.shape[0]))));
    else:
       resizeto = (int (round (IMAGE_SIZE * (float (img.shape[0])  / img.shape[1]))), IMAGE_SIZE);

    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)

    return img3[:,:,::-1]  # turn into rgb format

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = read_image(image_file);
        image_data = np.array (image, dtype=np.float32);
        image_data[:,:,0] = (image_data[:,:,0].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,1] = (image_data[:,:,1].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,2] = (image_data[:,:,2].astype(float) - pixel_depth / 2) / pixel_depth

        data[i] = image_data; # image_data.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    return data

train_normalized = prep_data(train_images)
test_normalized = prep_data(test_images)

print("Train shape: {}".format(train_normalized.shape))
print("Test shape: {}".format(test_normalized.shape))



np.random.seed (133)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset_rand, train_labels_rand = randomize(train_normalized, train_labels)
test_dataset, test_labels = randomize(test_normalized, test_labels)

# split up into training + valid
valid_dataset = train_dataset_rand[:VALID_SIZE,:,:,:]
valid_labels =   train_labels_rand[:VALID_SIZE]
train_dataset = train_dataset_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE,:,:,:]
train_labels  = train_labels_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE]
print ('Training', train_dataset.shape, train_labels.shape)
print ('Validation', valid_dataset.shape, valid_labels.shape)
print ('Test', test_dataset.shape, test_labels.shape)

import tensorflow as tf
image_size = IMAGE_SIZE # TODO: redundant, consolidate
num_labels = 2
num_channels = 3 # rg

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (labels=='cats').astype(np.float32); # set dogs to 0 and cats to 1
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print ('Training set', train_dataset.shape, train_labels.shape)
print ('Validation set', valid_dataset.shape, valid_labels.shape)
print ('Test set', test_dataset.shape, test_labels.shape)
##################################################################
#                                                                #
# Applying CNN in the data using Tensor Flow library             #
#                                                                #
##################################################################
from six.moves import range
import tensorflow as tf
import time
import random
start_time = time.time()

def accuracy(predictions, labels):
 return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
         / predictions.shape[0])

batch_size = 16
patch_size = 3
depth = 12*4
num_hidden = 4


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Input data.
network = input_data(shape=[None,150,150,3],name='input')
network = conv_2d(network,32,3,padding='same', activation='relu',regularizer="L2")
network = max_pool_2d(network,2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")￼￼￼
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.0001,
                    loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=3)


model.fit({'input': train_dataset}, {'target': train_labels}, n_epoch= 100,batch_size=batch_size,
          validation_set=({'input': valid_dataset}, {'target': valid_labels}),
          snapshot_step=20, show_metric=True, run_id='convnet_mnist')
model.save('my_model.tflearn')
