import tensorflow as tf
import numpy as np
import select_data
import random
import os
import pandas as pd
import time

image_size = 56
batch_size = 16
patch_size = 3
depth = 12
num_hidden = 120
num_channels = 3
num_labels = 2

submission_csv_path = "/home/augusto/catdog_kaggle/sample_submission.csv"
test_folder_path = "/home/augusto/catdog_kaggle/data56/test"
model_path =  '/home/augusto/catdog_kaggle/model.cpkt'

dataframe = pd.read_csv(submission_csv_path)
dataframe.iloc[:,1] = -1

id = os.listdir(test_folder_path)

step = 500
for start in range(0,12001,step):
    test_dataset, test_names = select_data.load_test_dataset(test_folder_path,image_size,num_channels,start,start+step)

    graph = tf.Graph()

    with graph.as_default():
        tf_test_dataset = tf.constant(test_dataset)
        #weights and biases
        patch_size1 = 2
        layer1_weights = layer1_weights = tf.Variable(tf.truncated_normal(
                [patch_size1, patch_size1, num_channels, depth], stddev=0.1),name ="layer1_weights")
        layer1_biases  = tf.Variable(tf.zeros([depth]),name ="layer1_biases")

        patch_size2 = 10
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1),name ="layer2_weights")
        layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]),name ="layer2_biases")


        layer3_weights = tf.Variable(tf.truncated_normal(
          [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

        layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        # Model.
        def model(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv = tf.nn.avg_pool(hidden, [1,2,2,1], [1, 2, 2, 1], padding='SAME')
            conv = tf.nn.dropout(conv,0.64)
            hidden = tf.nn.relu(conv + layer2_biases)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases


        saver = tf.train.Saver({"my_layer1_weights":layer1_weights ,
                            "my_layer1_biases":layer1_biases ,
                            "my_layer2_weights":layer2_weights ,
                            "my_layer2_biases":layer2_biases ,
                            "my_layer3_weights":layer3_weights ,
                            "my_layer3_biases":layer3_biases ,
                            "my_layer3_weights":layer3_weights ,
                            "my_layer4_biases":layer4_biases })

        test_prediction = tf.nn.softmax(model(tf_test_dataset))

        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            saver.restore(session,model_path)
            #predictions = session.run(test_prediction)
            #session.run(test_prediction.eval())
            j=0
            print("len(test_names): ", len(test_names))
            for pred in test_prediction.eval():
                #print("j :",j," test_names[j][:-4]",test_names[j][:-4])
                index = int(test_names[j][:-4])
                index = index - 1
                #print("test :", int(test_names[j][:-4]))
                dataframe["label"][index]    = np.argmax(pred)
                print(np.argmax(pred))
                j+=1
            dataframe.to_csv(submission_csv_path,sep=",",encoding='utf-8')
            print("step :", start)
            time.sleep(2)
