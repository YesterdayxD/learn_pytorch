import tensorflow as tf
import numpy as np

x = np.random.rand(5000, 784)
y = np.random.rand(5000, 10)
with tf.name_scope('X__Y'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

with tf.name_scope('weight'):
    w = tf.Variable(tf.random_normal(shape=(784, 10)))
    b = tf.Variable(tf.random_normal(shape=(1, 10)))

with tf.name_scope('model'):
    y_ = tf.matmul(X, w) + b

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_ - y))
    tf.summary.scalar('loss', loss)

optim = tf.train.GradientDescentOptimizer(learning_rate=0.9).minimize(loss=loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    #sess.run()
    write = tf.summary.FileWriter('./logs', sess.graph)
