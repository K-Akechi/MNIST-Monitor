import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import model

model_dir = './model3/'
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
y, intermediate = model.model3(image, keep_prob)
predicted = tf.argmax(y, 1)
label = tf.argmax(y_, 1)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

saver = tf.train.Saver()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    print('restore succeed.')

    samples = []
    # pred = []
    ground = []
    equal = 0

    for i in range(55000):
        images = mnist.train.images[i:i+1, :]
        labels = mnist.train.labels[i:i+1, :]
        feed_dict = {x: images, y_: labels, keep_prob: 1.0}
        intermediateValues, predictedNp, labelNp, = sess.run([intermediate, predicted, label], feed_dict=feed_dict)
        if predictedNp == labelNp:
            equal += 1
            samples.extend(intermediateValues)
            ground.extend(labelNp)

    samples = np.array(samples)
    ground = np.array(ground)
    print(samples.shape, ground.shape)
    np.save('training_set_neuron_outputs', samples)
    np.save('training_set_labels', ground)

    samples_test = []
    pred_test = []
    label_test = []
    equal = 0
    none = 0
    for i in range(10000):
        images = mnist.test.images[i:i+1, :]
        labels = mnist.test.labels[i:i+1, :]
        feed_dict = {x: images, y_: labels, keep_prob: 1.0}
        intermediateValues, predictedNp, labelNp, = sess.run([intermediate, predicted, label], feed_dict=feed_dict)
        if predictedNp == labelNp:
            equal += 1
        else:
            none += 1
        samples_test.extend(intermediateValues)
        pred_test.extend(predictedNp)
        label_test.extend(labelNp)
    samples_test = np.array(samples_test)
    pred_test = np.array(pred_test)
    label_test = np.array(label_test)

    print(equal, none)
    print(samples_test.shape, pred_test.shape, label_test.shape)
    np.save('test_set_neuron_outputs', samples_test)
    np.save('test_set_predictions', pred_test)
    np.save('test_set_labels', label_test)
