from tensorflow.examples.tutorials.mnist import input_data
from lenet import Lenet
import tensorflow as tf
import config as cfg
import numpy as np


def main():
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_test, y_test = mnist.test.images, mnist.test.labels

    assert(len(X_test) == len(y_test))

    # Pad images with 0s
    X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    answer = 'y'

    net = Lenet(10)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, cfg.FINAL_PATH)
        while answer != 'n':
            index = np.random.randint(0, len(X_test))

            Z = net.logits.eval(feed_dict={net.X: X_test[index].reshape(1,32,32,1)})
            y_pred = np.argmax(Z, axis=1)

            print("Predicted class:", y_pred)
            print("Actual class:   ", y_test[index])
            print()
            answer = input("Do you want to predict one more number?(y/n) ")



#Standard way to define program starting point
if __name__ == '__main__':
    main()
