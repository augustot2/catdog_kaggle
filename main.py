#!/usr/local/bin/python


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
matplotlib.style.use('ggplot')

##########################################################
# Pick random images and it's labels in train folder and #
# separate into train dataset and validation dataset     #
##########################################################

num_images   = 12000
image_size   = 56
num_channels = 3
valid_num    = 500
num_labels = 2

#use only if you don't have label.pickle and Image.pickle
#select_data.random_select("/home/augusto/catdog_kaggle/data56/train",num_images,image_size,num_channels)

with open("/home/augusto/catdog_kaggle/data56/Train.pickle",'rb') as pickle_file:
    dataset = pickle.load(pickle_file)

with open("/home/augusto/catdog_kaggle/data56/Label.pickle",'rb') as pickle_file:
    labels = pickle.load(pickle_file)


test_dataset, test_names = select_data.load_test_dataset("/home/augusto/catdog_kaggle/data56/test",image_size,num_channels,num_images = 10)


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size ,image_size,num_channels)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(dataset[valid_num:num_images], labels[valid_num:num_images])
valid_dataset, valid_labels = reformat(dataset[0:valid_num], labels[0:valid_num])
test_dataset, _ = reformat(test_dataset, labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
#  results_label = test_prediction.eval()
# test_id = [re.sub(r'[.jpg]','',id) for id in test_id]

#data_frame_test = pd.DataFrame(collumns = ["id","label"])
  #data_frame_test.to_csv("test.csv",sep=",",encoding='utf-8')
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

batch_size = 3*20
patch_size = 3
depth = 22
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1),name ="layer1_weights")
  layer1_biases = tf.Variable(tf.zeros([depth]),name ="layer1_biases")
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1),name ="layer2_weights")
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]),name ="layer2_biases")
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1),name ="layer3_weights")
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]),name ="layer3_biases")
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1),name ="layer3_weights")
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]),name ="layer4_biases")

  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.max_pool(hidden, [1,2,2,1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    hidden = tf.nn.dropout(hidden, tf.constant(0.5, dtype=tf.float32) )
    shape = hidden.get_shape().as_list()
    #hidden = tf.nn.dropout(hidden, tf.constant(0.75, dtype=tf.float32) )
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

    return tf.matmul(hidden, layer4_weights) + layer4_biases

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  # Optimizer
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

  saver = tf.train.Saver({"my_layer1_weights":layer1_weights ,
                        "my_layer1_biases":layer1_biases ,
                        "my_layer2_weights":layer2_weights ,
                        "my_layer2_biases":layer2_biases ,
                        "my_layer3_weights":layer3_weights ,
                        "my_layer3_biases":layer3_biases ,
                        "my_layer3_weights":layer3_weights ,
                        "my_layer4_biases":layer4_biases })



num_steps = 750
step_break = 25
loss_acc = pd.DataFrame(columns=["loss", "MinBatch_acc", "Valid_acc"],index = np.arange(0,num_steps+1,step_break))
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111) # Create an axes.

#plt.figure()
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % step_break == 0):
      batch_acc = accuracy(predictions, batch_labels)
      valid_acc = accuracy(valid_prediction.eval(), valid_labels)
      data = [l,batch_acc,valid_acc]
      loss_acc.loc[step] = [l, batch_acc, valid_acc]
      print('Minibatch loss at step %d: %.1f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % batch_acc )
      print('Validation accuracy: %.1f%%' %  valid_acc)
      loss_acc.plot(secondary_y= ['prex'],ax=ax)
      plt.pause(.1)
      ax.cla()

  plt.close()
  save_path = saver.save(session, "/home/augusto/catdog_kaggle/model.cpkt")
  print("Save to path: ", save_path)


#loss_acc = loss_acc.cumsum()
#


print("--- %s seconds ---" % (time.time() - start_time))
