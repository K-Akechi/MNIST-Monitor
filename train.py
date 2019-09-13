import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
import time
import model

train_dir = './model2/'
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

def main(argv=None):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    y = model.model2(image, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        restore_dir = tf.train.latest_checkpoint(train_dir)
        if restore_dir:
            saver.restore(sess, restore_dir)
            print('restore succeed')
        else:
            sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(20000 + 1):
            start_time = time.time()
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            duration = time.time() - start_time
            if step % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g, %.3fsec" % (step, train_accuracy, duration))
            if step % 2000 == 0:
                saver.save(sess, train_dir, global_step=step)
        test_batch = mnist.test.next_batch(5000)
        #print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        print("1st part test accuracy %g" % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
        test_batch = mnist.test.next_batch(5000)
        print("2nd part test accuracy %g" % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
        coord.request_stop()
        coord.join(threads)
    return


if __name__ == "__main__":
    tf.app.run()