 #!/usr/local/bin/python

#%matplotlib inline


from six.moves import cPickle as pickle
import os
import numpy as np
from PIL import Image
import random
import select_data
import pandas as pd
import re
import matplotlib.pyplot as plt
from IPython import display
import matplotlib
#import pylab as plt

matplotlib.style.use('ggplot')

##########################################################
# Pick random images and it's labels in train folder and #
# separate into train dataset and validation dataset     #
##########################################################

num_images   = 4000
image_size   = 150
num_channels = 3
valid_num    = 1000
num_labels = 2

#use only if you don't have label.pickle and Image.pickle

select_data.random_select("/home/augusto/catdog_kaggle/data150/train",num_images,image_size,num_channels)


with open("/home/augusto/catdog_kaggle/data150/Train.pickle",'rb') as pickle_file:
    dataset = pickle.load(pickle_file)

with open("/home/augusto/catdog_kaggle/data150/Label.pickle",'rb') as pickle_file:
    labels = pickle.load(pickle_file)

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size ,image_size,num_channels)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(dataset[valid_num:num_images], labels[valid_num:num_images])
valid_dataset, valid_labels = reformat(dataset[0:valid_num], labels[0:valid_num])

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)


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
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.00001,
                    loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=True)
model.fit({'input': train_dataset}, {'target': train_labels}, n_epoch= 100,batch_size=batch_size,
          validation_set=({'input': valid_dataset}, {'target': valid_labels}),
          snapshot_step=20, show_metric=True, run_id='convnet_mnist')
