import tensorflow as tf


def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable('weights', dtype='float32',shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    # initial = tf.constant(0.1, shape=shape, dtype=float)
    return tf.get_variable('bias', shape=shape, dtype='float32', initializer=tf.constant_initializer(0.05))


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model1(image, keep_prob):
    with tf.variable_scope('conv1'):
        w_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(image, w_conv1, 1) + b_conv1)

    with tf.variable_scope('downsample1'):
        w_downsample1 = weight_variable([5, 5, 32, 32])
        h_downsample1 = tf.nn.relu(conv2d(h_conv1, w_downsample1, 2))

    with tf.variable_scope('conv2'):
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_downsample1, w_conv2, 1) + b_conv2)

    with tf.variable_scope('downsample2'):
        w_downsample2 = weight_variable([5, 5, 64, 64])
        h_downsample2 = tf.nn.relu(conv2d(h_conv2, w_downsample2, 2))
        flat = tf.reshape(h_downsample2, [-1, 7 * 7 * 64])

    with tf.variable_scope('fc1'):
        w_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    intermediate = h_fc1

    with tf.variable_scope('fc2'):
        w_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    return y, intermediate

def model2(image, keep_prob):
    with tf.variable_scope('conv1'):
        w_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(image, w_conv1, 1) + b_conv1)

    with tf.variable_scope('maxpool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv2'):
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 1) + b_conv2)

    with tf.variable_scope('maxpool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    with tf.variable_scope('fc1'):
        w_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    intermediate = h_fc1

    with tf.variable_scope('fc2'):
        w_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    return y, intermediate