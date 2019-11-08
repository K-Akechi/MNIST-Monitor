import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, mixture
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import pandas as pd
from sklearn.manifold import TSNE
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


    samples_test = []
    pred_test = []
    ground_test = []
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
        ground_test.extend(labelNp)
    print(equal, none)

    samples = np.array(samples)
    pred = np.array(pred)
    ground = np.array(ground)
    samples_test = np.array(samples_test)
    pred_test = np.array(pred_test)
    ground_test = np.array(ground_test)

    # print(samples, pred.shape, ground.shape)
    start_time = time.time()
    # km = KMeans(n_clusters=10, random_state=9, n_jobs=-1)
    # km.fit(samples)
    # gmm = mixture.GaussianMixture(n_components=10, random_state=9)
    # gmm.fit(samples)
    # bandwidth = estimate_bandwidth(samples, quantile=0.2, n_samples=1000)
    # ms = MeanShift(bin_seeding=True).fit(samples)
    agg = AgglomerativeClustering().fit(samples)
    y_pred = agg.predict(samples)
    # y_pred = km.predict(samples)
    # y_pred = gmm.predict(samples)
    # y_pred = dbscan.labels_
    duration = time.time() - start_time
    print('{}seconds'.format(duration))
    # centers = km.cluster_centers_
    print(y_pred)
    print(metrics.calinski_harabaz_score(samples, y_pred))
    stat = np.zeros((10, 10), dtype=int)
    for i in range(samples.shape[0]):
        stat[y_pred[i]][ground[i]] += 1
    print(stat)

    # y_test = km.predict(samples_test)
    # y_test = gmm.predict(samples_test)
    # # y_test = dbscan.predict(samples_test)
    # out_of_cluster = 0
    # out_of_cluster_and_mis = 0
    # mis_classified = 0
    # correct = 0
    # index = np.array([9, 3, 0, 6, 2, 4, 8, 1, 5, 7])
    # for i in range(10000):
    #     if pred_test[i] != ground_test[i]:
    #         mis_classified += 1
    #         if index[y_test[i]] != pred_test[i]:
    #             out_of_cluster_and_mis += 1
    #     else:
    #         correct += 1
    #     if index[y_test[i]] != pred_test[i]:
    #         out_of_cluster += 1
    # print(out_of_cluster, out_of_cluster_and_mis, mis_classified, correct)


