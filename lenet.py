import tensorflow as tf
import config as cfg
from tensorflow.contrib.layers import flatten


class Lenet:
    """Class to generate a CNN Lenet-5"""

    def __init__(self,X, n_output=10, is_trainable=False):
        with tf.name_scope("LeNet"):

            # LAYER 1 - CONVOLUTIONAL. INPUT = 32X32X1 - OUTPUT = 28X28X6
            with tf.name_scope("conv1"):
                self.conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6),mean=cfg.MU, stddev=cfg.SIGMA), name="weights", trainable=is_trainable)
                self.conv1_b = tf.Variable(tf.zeros(6), name="bias", trainable=is_trainable)
                self.conv1 = tf.nn.conv2d(X, self.conv1_W, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_b
                self.conv1 = tf.nn.relu(self.conv1, name="activation")

            # LAYER 1 - POOLING. INPUT = 28X28X6 - OUTPUT = 14X14X6
            self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")

            # LAYER 2 - CONVOLUTIONAL. INPUT = 14X14X6 OUTPUT = 10X10X16
            with tf.name_scope("conv2"):
                self.conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16),mean=cfg.MU, stddev=cfg.SIGMA), name="weights", trainable=is_trainable)
                self.conv2_b = tf.Variable(tf.zeros(16), name="bias", trainable=is_trainable)
                self.conv2 = tf.nn.conv2d(self.pool1, self.conv2_W, strides=[1, 1, 1, 1], padding='VALID') + self.conv2_b
                self.conv2 = tf.nn.relu(self.conv2, name="activation")

            # LAYER 2 - POOLING. INPUT = 10X10X16 - OUTPUT = 5X5X16
            self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool2")

            # FLATTEN. INPUT = 5X5X16 - OUTPUT = 400
            self.fc0 = flatten(self.pool2)

            with tf.name_scope("fc1"):
                self.fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=cfg.MU, stddev=cfg.SIGMA), name="weights", trainable=is_trainable)
                self.fc1_b = tf.Variable(tf.zeros(120), name="bias", trainable=is_trainable)
                self.fc1 = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b
                self.fc1 = tf.nn.relu(self.fc1, name="activation")

            with tf.name_scope("fc2"):
                self.fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=cfg.MU, stddev=cfg.SIGMA), name="weights", trainable=is_trainable)
                self.fc2_b = tf.Variable(tf.zeros(84), name="bias", trainable=is_trainable)
                self.fc2 = tf.matmul(self.fc1, self.fc2_W) + self.fc2_b
                self.fc2 = tf.nn.relu(self.fc2, name="activation")

            with tf.name_scope("fc3_outputs"):
                self.fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_output), mean=cfg.MU, stddev=cfg.SIGMA), name="weights", trainable=is_trainable)
                self.fc3_b = tf.Variable(tf.zeros(n_output), name="bias", trainable=is_trainable)
                self.logits = tf.matmul(self.fc2, self.fc3_W) + self.fc3_b
                if is_trainable:
                    self.output = tf.identity(self.logits, name="activation")
                else:
                    self.output = tf.nn.softmax(self.logits, name="activation")
