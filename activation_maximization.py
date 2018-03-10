import tensorflow as tf
from lenet import Lenet
import config as cfg
from PIL import Image


def main():

    for i in range(10):
        # i = 3
        tf.reset_default_graph()

        X = tf.Variable(tf.random_uniform(shape=(1, 32, 32, 1), minval=0, maxval=0.5), name="X")

        net = Lenet(X, is_trainable=False)

        with tf.name_scope("am"):
            am = (tf.log(net.output[:, i]) - 0.5 * tf.norm(X, ord=2, name="l2-norm"))

        with tf.name_scope("optimize-am"):
            optimizer = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE)
            training_op = optimizer.minimize(-am)  # minimizing(-cost) equals to maximize(cost)

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet"))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, cfg.FINAL_PATH)
            print("Optimizing...")
            print()
            for epoch in range(5000):
                sess.run(training_op)
                print("Epoch:", epoch," Loss:", am.eval())
                print(' Classes Probability:', net.output.eval())

            img = sess.run(tf.squeeze(X))  # -tf.reduce_min(X))/(tf.reduce_max(X)-tf.reduce_min(X))
            img = Image.fromarray(img, "F")
            img.save("./img/prototype{}.tiff".format(i))

            print("Prototype saved as image")


# Standard way to define program starting point
if __name__ == '__main__':
    main()
