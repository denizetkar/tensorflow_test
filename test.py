import tensorflow as tf

# tf.reset_default_graph()
var1 = tf.get_variable(name="var1", initializer=tf.constant(2.))
current_scope = tf.contrib.framework.get_name_scope()
with tf.variable_scope(current_scope, reuse=True):
    var2 = tf.get_variable("var1", [])
assert var1 is var2
with tf.Session() as sess:
    sess.run(tf.variables_initializer([var1, var2]))
    print(sess.run(var1))
sess.close()

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op) # assignment is an operation, it should run to take effect
    print(W.eval())
sess.close()

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='x')
# tensorboard --logdir="./graphs"
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))
sess.close()
writer.close()

import tensorflow as tf
import numpy as np
import math
inputs = tf.placeholder(shape=[None, 5], dtype=tf.float32)
feature_ratio = 0.5
idx = np.sort(np.random.choice(inputs.shape[1].value,
                               math.ceil(inputs.shape[1].value*feature_ratio),
                               replace=False))
reduced_inputs = tf.stack([inputs[:, i] for i in idx], axis=1, name=f'reduced_input{i}')
x = np.array([[1,2,3,4,5],[6,7,8,9,10]])
with tf.Session() as sess:
    print(sess.run(reduced_inputs, feed_dict={inputs: x}))
sess.close()


import tensorflow as tf
import numpy as np
concat_output = tf.placeholder(tf.float32, [None,3,4])
decision_net = tf.placeholder(tf.float32, [None,3])
decided_output = tf.reduce_sum(concat_output * tf.reshape(decision_net, [-1,decision_net.shape[1],1]), axis=1)
x = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]] , [[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
y = np.array([[1,2,3], [3,2,1]])
with tf.Session() as sess:
    print( sess.run(decided_output, feed_dict={concat_output: x, decision_net: y}) )
sess.close()


import tensorflow as tf
import numpy as np
target = tf.placeholder(tf.float32, [None,3])
output = tf.placeholder(tf.float32, [None,3])
loss = tf.nn.l2_loss(target-output)
huber_l = tf.losses.huber_loss(target, output)
x = np.array([[0,0,0], [1,1,1]])
y = np.array([[2,2,2], [3,3,3]])
with tf.Session() as sess:
    print(sess.run(loss, feed_dict={target:x, output:y}))
    print(sess.run(loss, feed_dict={target:x[0:1], output:y[0:1]})+sess.run(loss, feed_dict={target:x[1:2], output:y[1:2]}))
sess.close()
