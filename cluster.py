import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import model
import time

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
    start_time = time.time()
    samples = []
    pred = []
    ground = []
    equal = 0
    none = 0
    for i in range(55000):
        images = mnist.train.images[i:i+1, :]
        labels = mnist.train.labels[i:i+1, :]
        feed_dict = {x: images, y_: labels, keep_prob: 1.0}
        intermediateValues, predictedNp, labelNp, = sess.run([intermediate, predicted, label], feed_dict=feed_dict)
        if predictedNp == labelNp:
            equal += 1
        else:
            none += 1
        samples.extend(intermediateValues)
        pred.extend(predictedNp)
        ground.extend(labelNp)
    duration = time.time() - start_time
    print('{}seconds'.format(duration))
    samples = np.array(samples)
    pred = np.array(pred)
    ground = np.array(ground)
    # print(samples, pred.shape, ground.shape)
    y_pred = KMeans(n_clusters=10, random_state=9).fit_predict(samples)
    print(metrics.calinski_harabaz_score(samples, y_pred))
    stat = np.zeros((10, 10), dtype=int)
    for i in range(samples.shape[0]):
        stat[y_pred[i]][ground[i]] += 1
    print(stat)



