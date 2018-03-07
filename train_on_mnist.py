from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
from datetime import datetime
import tensorflow as tf
from lenet import Lenet
import config as cfg
import numpy as np


def main():

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train           = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test             = mnist.test.images, mnist.test.labels

    assert(len(X_train) == len(y_train))
    assert(len(X_validation) == len(y_validation))
    assert(len(X_test) == len(y_test))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))

    #The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
    #However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.
    # Pad images with 0s
    X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    print("Updated Image Shape: {}".format(X_train[0].shape))

    #Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train)


    net = Lenet(10)
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=net.one_hot_y, logits=net.output)
        loss_operation = tf.reduce_mean(cross_entropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate = cfg.LEARNING_RATE)
        training_op = optimizer.minimize(loss_operation)

    with tf.name_scope("eval"):
        correct = tf.equal(tf.argmax(net.output, 1), tf.argmax(net.one_hot_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    cross_entropy_summary = tf.summary.scalar('cross_entropy',loss_operation)
    acc_train_summary = tf.summary.scalar('training_accuracy', accuracy)
    acc_val_summary = tf.summary.scalar('validation_accuracy', accuracy)


    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    num_examples = len(X_train)
    num_batches = num_examples // cfg.BATCH_SIZE

    with tf.Session() as sess:
        init.run()
        print("Training...")
        print()
        for epoch in range(cfg.EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            batch_index = 0
            for offset in range(0, num_examples, cfg.BATCH_SIZE):
                end = offset + cfg.BATCH_SIZE
                X_batch, y_batch = X_train[offset:end], y_train[offset:end]

                if batch_index % 10 == 0:
                    cross_entropy_str = cross_entropy_summary.eval(feed_dict={net.X: X_batch, net.y: y_batch})
                    step = epoch * num_batches + batch_index
                    file_writer.add_summary(cross_entropy_str, step)

                sess.run(training_op, feed_dict={net.X: X_batch, net.y: y_batch})
                batch_index += 1

            acc_train_str = acc_train_summary.eval(feed_dict={net.X: X_train, net.y: y_train})
            acc_val_str = acc_val_summary.eval(feed_dict={net.X: X_validation, net.y: y_validation})
            file_writer.add_summary(acc_train_str,epoch)
            file_writer.add_summary(acc_val_str,epoch)
            print("Epoch:", epoch)
            save_path = saver.save(sess, cfg.INTERMEDIATE_PATH)

        save_path = saver.save(sess, cfg.FINAL_PATH)
        print("Model Saved")

    file_writer.close()

    with tf.Session() as sess:
        saver.restore(sess, save_path)

        acc_test = accuracy.eval(feed_dict={net.X: X_test, net.y: y_test})
        print("Test Accuracy = {:.3f}".format(acc_test))



#Standard way to define program starting point
if __name__ == '__main__':
    main()
