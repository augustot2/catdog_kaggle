#!/usr/local/bin/python

from six.moves import cPickle as pickle
import os
import numpy as np
from PIL import Image
import random
import time

def random_pick(train_path, num_images):
    os.chdir(train_path)
    images = os.listdir()
    image_size = len(images)
    random_index = random.sample(range(0,image_size),num_images)

    return [images[i] for i in random_index]

def random_select(train_path, num_images,image_size,num_channel):
    os.chdir(train_path)
    current_path =os.path.join(train_path, os.pardir)

    images = random_pick(train_path, num_images)

    Images_pickle_file = open("../Train.pickle","wb")
    Label_pickle_file  = open("../Label.pickle","wb")

    dataset = np.ndarray(shape=(num_images, image_size, image_size, num_channel),
                             dtype=np.float32)
    label = np.ndarray(shape=(num_images),
                             dtype=np.float32)
    print("choosed :", num_images, " images")
    time.sleep(3)
    j = 0
    for image in images:
        print(image)
        image_matrix = Image.open(image)
        image_matrix = np.asarray(image_matrix)
        dataset[j,:,:,:] = image_matrix

        if 'cat' in image:
            print("cat")
            label[j] = 0
        else:
            print("dog")
            label[j] = 1
        j += 1

    pickle.dump(dataset, Images_pickle_file )
    pickle.dump(label, Label_pickle_file)
    print("Label pickle stored in:", os.path.abspath(current_path),"/Label.pickle")
    print("Image pickle stored in:", os.path.abspath(current_path),"/Train.pickle")

def load_test_dataset(train_path, image_size,num_channel, num_images = [None]):
    os.chdir(train_path)
    current_path =os.path.join(train_path, os.pardir)

    images = os.listdir()

    if num_images == None:
        num_images = len(images)

    dataset = np.ndarray(shape=(num_images, image_size, image_size, num_channel),
                             dtype=np.float32)
    j = 0
    for image in images[0:num_images]:
        print(image)
        image_matrix = Image.open(image)
        image_matrix = np.asarray(image_matrix)
        dataset[j,:,:,:] = image_matrix

    return dataset, images
