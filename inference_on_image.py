from lenet import Lenet
import tensorflow as tf
import config as cfg
import numpy as np
from PIL import Image


def main():

    image = Image.open("./img/prototype3.tiff")
    image = np.array(image)

    # PLACEHOLDERS FOR FEEDING INPUT DATA
    X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name="X")

    net = Lenet(X)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, cfg.FINAL_PATH)
        Z = net.output.eval(feed_dict={X: image.reshape(1, 32, 32, 1)})
        y_pred = np.argmax(Z, axis=1)

        print("tutte le classi:", Z)
        print("Predicted class:", y_pred)


# Standard way to define program starting point
if __name__ == '__main__':
    main()
