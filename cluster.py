import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, mixture, cluster
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time


def kmeans(samples, n_clusters, samples_to_predict):
    km = cluster.KMeans(n_clusters=n_clusters, n_jobs=-1)
    km.fit(samples)
    return km.predict(samples_to_predict)


def gaussian(samples, n_clusters, samples_to_predict):
    gmm = mixture.GaussianMixture(n_components=n_clusters)
    gmm.fit(samples)
    return gmm.predict(samples_to_predict)


def meanshift(samples, samples_to_predict):
    bandwidth = cluster.estimate_bandwidth(samples, n_jobs=-1)
    ms = cluster.MeanShift(bandwidth=bandwidth, n_jobs=-1)
    ms.fit(samples)
    return ms.predict(samples_to_predict)


if __name__ == '__main__':
    n = 10

    interValues_train = np.load('training_set_neuron_outputs.npy')
    labels_train = np.load('training_set_labels.npy')
    interValues_test = np.load('test_set_neuron_outputs.npy')
    labels_test = np.load('test_set_labels.npy')
    predictions_test = np.load('test_set_predictions.npy')
    print('data retrieve success.')

    stat = np.zeros((n, 10), dtype=int)
    start_time = time.time()
    # kmeans_result = kmeans(interValues_train, n, interValues_train)
    meanshift_result = meanshift(interValues_train, interValues_train)
    duration = time.time() - start_time
    print('clustering finish in {} seconds'.format(duration))
    for i in range(interValues_train.shape[0]):
        stat[meanshift_result[i]][labels_train[i]] += 1
    print(stat)
