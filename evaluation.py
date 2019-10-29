import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import model
from monitor import napmonitor

model_dir = './model3/'
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
y, _ = model.model3(image, keep_prob)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(model_dir))
print('restore succeed.')

test_images1 = mnist.test.images[:, :]
test_labels1 = mnist.test.labels[:, :]

# print(y.eval(session=sess, feed_dict={x: test_images1, y_: test_labels1, keep_prob: 1.0}))
print('Accuracy of the network on the 10000 test images: {} %'.format(accuracy.eval(session=sess, feed_dict={x: test_images1, y_: test_labels1, keep_prob: 1.0})))

# test_images2 = mnist.test.images[5000:, :]
# test_labels2 = mnist.test.labels[5000:, :]
#
# #print(y.eval(session=sess, feed_dict={x: test_images2, y_: test_labels2, keep_prob: 1.0}))
# print(accuracy.eval(session=sess, feed_dict={x: test_images2, y_: test_labels2, keep_prob: 1.0}))

sess.close()
