import numpy as np
import tensorflow as tf


# Set hyper-parameters:
n_epoch, lr = 1000, 0.01

# Load data:
n_data = 1000
noise = np.random.normal(0, 1, (n_data, 1))
x_train = np.random.uniform(-10, 10, (n_data,1))
y_train = x_train*2 + 10 + noise

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

# Setup a model
np.random.seed(0)
w1 = tf.Variable(tf.random_normal([1,1]))
b1 = tf.Variable(tf.random_normal([1]))

output = tf.add(tf.matmul(x, w1), b1)
loss = tf.reduce_mean(tf.square(output - y))

grad_y = 2*(output - y)/tf.cast(tf.shape(y)[0], tf.float32)
grad_w1 = tf.matmul(tf.transpose(x), grad_y)
grad_b1 = tf.reduce_sum(grad_y, 0)

update_w1 = w1.assign(w1 - lr*grad_w1)
update_b1 = b1.assign(b1 - lr*grad_b1)

# Train the model:

# To prevent CUDA_ERROR_OUT_OF_MEMORY:
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#with tf.Session(config=config) as sess:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        sess.run([update_w1, update_b1], feed_dict={x:x_train, y:y_train})
        loss_ = sess.run(loss, feed_dict={x:x_train, y:y_train})

    # Evaluate the trained model:
    print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss_))
    print(w1.eval(), b1.eval())