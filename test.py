import tensorflow as tf
import numpy as np

tf.reset_default_graph()
x = tf.Variable(np.arange(16), name="x")
y = tf.split(x, 4, axis=0, name="y")

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    xval, yval = sess.run([x, y])


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
k = tf.constant(1.0)
c = tf.add(a, b)
d = tf.subtract(b, k)
e = tf.multiply(c, d)

with tf.Session() as sess:
    # summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    result = sess.run(e, feed_dict={a:2.0, b:0.5})
