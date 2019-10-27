import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import model
from monitor import napmonitor


model_dir = './model2/'
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

num_classes = 10
sizeOfNeuronsToMonitor = 1024

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
y, intermediate = model.model2(image, keep_prob)
total = mnist.test.labels.shape[0]
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct = tf.reduce_sum(tf.cast(correct, tf.float32))

saver = tf.train.Saver()
monitor = napmonitor.NAP_Monitor(num_classes, sizeOfNeuronsToMonitor)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    print('restore succeed.')
    feed_dict = {x: mnist.test.images[:, :], y_: mnist.test.labels[:, :], keep_prob: 1.0}
    monitor.addAllNeuronPatternsToClass(intermediate.eval(session=sess, feed_dict=feed_dict),
                                        y.eval(session=sess, feed_dict=feed_dict),
                                        y_.eval(session=sess, feed_dict=feed_dict), -1)
