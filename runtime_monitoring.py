import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import model
from monitor import napmonitor
# import sys
import time


# sys.setrecursionlimit(10000)

model_dir = './model3/'
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

num_classes = 10
sizeOfNeuronsToMonitor = 40

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
y, intermediate = model.model3(image, keep_prob)
predicted = tf.argmax(y, 1)
label = tf.argmax(y_, 1)
total = mnist.test.labels.shape[0]
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct = tf.reduce_sum(tf.cast(correct, tf.float32))

saver = tf.train.Saver()
monitor = napmonitor.NAP_Monitor(num_classes, sizeOfNeuronsToMonitor)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

print(mnist.train.images.shape, mnist.test.images.shape)

with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    print('restore succeed.')
    start_time = time.time()
    for i in range(550):
        # print(mnist.test.images[i, :], mnist.test.labels[i, :])
        images = mnist.train.images[i*100:(i+1)*100, :]
        labels = mnist.train.labels[i*100:(i+1)*100, :]
        feed_dict = {x: images, y_: labels, keep_prob: 1.0}
        # print(intermediate.eval(session=sess, feed_dict=feed_dict), predicted.eval(session=sess, feed_dict=feed_dict),
        #       label.eval(session=sess, feed_dict=feed_dict))

        monitor.addAllNeuronPatternsToClass(intermediate.eval(session=sess, feed_dict=feed_dict),
                                            predicted.eval(session=sess, feed_dict=feed_dict),
                                            label.eval(session=sess, feed_dict=feed_dict), -1)
        print(i*100)
    # images = mnist.test.images[:, :]
    # labels = mnist.test.labels[:, :]
    # feed_dict = {x: images, y_: labels, keep_prob: 1.0}
    #
    # monitor.addAllNeuronPatternsToClass(intermediate.eval(session=sess, feed_dict=feed_dict),
    #                                     predicted.eval(session=sess, feed_dict=feed_dict),
    #                                     label.eval(session=sess, feed_dict=feed_dict), -1)
    duration = time.time() - start_time
    print('finish in {} seconds.'.format(duration))

    # Perform run-time monitoring
    outofActivationPattern = 0
    outofActivationPatternAndResultWrong = 0
    feed_dict = {x: mnist.test.images[:, :], y_: mnist.test.labels[:, :], keep_prob: 1.0}
    predictedNp, intermediateValues, labels, correct_num = sess.run([predicted, intermediate, label, correct], feed_dict=feed_dict)
    print(predictedNp, labels)
    result = (predictedNp == labels)
    for exampleIndex in range(intermediateValues.shape[0]):
        if not monitor.isPatternContained(intermediateValues[exampleIndex, :], predictedNp[exampleIndex]):
            outofActivationPattern += 1
            if result[exampleIndex] == False:
                outofActivationPatternAndResultWrong += 1
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct_num / total))
    print('Out-of-activation pattern on the 10000 test images: {} %'.format(100 * outofActivationPattern / total))
    print('Out-of-activation pattern & misclassified / out-of-activation pattern : {} %'.format(
        100 * outofActivationPatternAndResultWrong / (outofActivationPattern)))
    print('Out-of-activation pattern & misclassified / mis-classified: {} %'.format(
        100 * outofActivationPatternAndResultWrong / (total - correct_num)))
    print(outofActivationPattern, outofActivationPatternAndResultWrong, result, total)
