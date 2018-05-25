import sys, os
sys.path.append(os.pardir)

import numpy as np
import tensorflow as tf

#from keras.datasets import mnist
import common.mnist as mnist
from datetime import datetime


# Set hyper-parameters:
n_epoch, batch_size, lr = 10, 64, 0.001
shuffle, verbose = True, True

# Load data:
x_train, y_train, x_test, y_test = mnist.load(flatten=False)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(-1, 28, 28, 1)/255.
#y_train = keras.utils.to_categorical(y_train, 10)
#x_test = x_test.reshape(-1, 28, 28, 1)/255.
#y_test = keras.utils.to_categorical(y_test, 10)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

# Setup a model:
tf.set_random_seed(0)
layer1 = tf.layers.conv2d(x, 32, [3, 3], activation=tf.nn.relu)
layer1 = tf.layers.max_pooling2d(layer1, [2, 2], [2, 2])
layer1 = tf.layers.dropout(layer1, 0.5, is_training)

layer2 = tf.layers.conv2d(layer1, 64, [3, 3], activation=tf.nn.relu)
layer2 = tf.layers.max_pooling2d(layer2, [2, 2], [2, 2])
layer2 = tf.layers.dropout(layer2, 0.5, is_training)

layer3 = tf.contrib.layers.flatten(layer2)
layer3 = tf.layers.dense(layer3, 256, activation=tf.nn.relu)
layer3 = tf.layers.dropout(layer3, 0.5, is_training)

output = tf.layers.dense(layer3, 10, activation=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
           logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# To prevent CUDA_ERROR_OUT_OF_MEMORY:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Train the model
n_data = x_train.shape[0]
n_batch = n_data // batch_size + (1 if n_data % batch_size else 0)
message = "Epoch[{:3d}] ({:3d}%) > Loss {:.3f} / Acc. {:.3f}"
t_start = datetime.now()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        index = np.arange(x_train.shape[0])
        if shuffle:
            np.random.shuffle(index)

        if verbose:
            print('')

        loss_train, acc_train = 0, 0
        for i in range(n_batch):
            batch = index[i*batch_size:(i+1)*batch_size]
            data = {x:x_train[batch], y:y_train[batch], is_training:True}

            sess.run(optimizer, feed_dict=data)
            loss_batch, acc_batch  = sess.run([loss, acc], feed_dict=data)
            loss_train += loss_batch
            acc_train += acc_batch

            if verbose and (i+1) % 100 == 0:
                print(message.format(epoch+1, int(100*(i+1)*batch_size/n_data),
                                     loss_batch, acc_batch))

        print(message.format(epoch+1, 100, loss_train/n_batch, acc_train/n_batch),
              "(Time {})".format(datetime.now() - t_start))

        # Evaluate the model after training:
        loss_test, acc_test= sess.run([loss, acc],
                            feed_dict={x:x_test, y:y_test, is_training:False})
        print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                        epoch+1, loss_test, acc_test))