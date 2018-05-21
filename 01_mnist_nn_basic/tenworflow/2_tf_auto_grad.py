import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from datetime import datetime


# Set hyper-parameters:
n_epoch, batch_size, lr = 10, 64, 0.01
shuffle, verbose = True, True

# Load data:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)/255.
y_train = keras.utils.to_categorical(y_train, 10)
x_test = x_test.reshape(-1, 784)/255.
y_test = keras.utils.to_categorical(y_test, 10)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
data = {x:x_train, y:y_train}

# Setup a model:
w1 = tf.Variable(tf.random_normal([784, 200], stddev=0.1))
b1 = tf.Variable(tf.zeros([200]))
w2 = tf.Variable(tf.random_normal([200, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))

# Forward propagation:
lin1 = tf.add(tf.matmul(x, w1), b1)
sig1 = tf.nn.sigmoid(lin1)
lin2 = tf.add(tf.matmul(sig1, w2), b2)
output = lin2
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
           logits=output, labels=y))

# Backward propagation:
grad_w1, grad_b1, grad_w2, grad_b2 = tf.gradients(loss, [w1, b1, w2, b2])

# Update model parameters:
update_w1 = w1.assign(w1 - lr*grad_w1)
update_b1 = b1.assign(b1 - lr*grad_b1)
update_w2 = w2.assign(w2 - lr*grad_w2)
update_b2 = b2.assign(b2 - lr*grad_b2)

# Evaluate loss and accuracy:
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
            data = {x:x_train[batch], y:y_train[batch]}

            sess.run([update_w1, update_b1, update_w2, update_b2], feed_dict=data)
            loss_batch, acc_batch  = sess.run([loss, acc], feed_dict=data)
            loss_train += loss_batch
            acc_train += acc_batch

            if verbose and (i+1) % 100 == 0:
                print(message.format(epoch+1, int(100*(i+1)*batch_size/n_data),
                                     loss_batch, acc_batch))

        print(message.format(epoch+1, 100, loss_train/n_batch, acc_train/n_batch),
              "(Time {})".format(datetime.now() - t_start))

    # Evaluate the model after training:
    loss_test, acc_test= sess.run([loss, acc], feed_dict={x:x_test, y:y_test})
    print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    epoch+1, loss_test, acc_test))