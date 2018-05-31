import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

f = np.load("data/mnist.npz")
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

# unrolled through 28 time steps
time_steps = 28
# hidden LSTM units
num_units = 100
# rows of 28 pixels
n_input = 28
# learning rate for adam
learning_rate = 0.01
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 10
# size of batch
batch_size = 1000


def int_to_categorical(array, num_of_classes):
    y_categorical = np.zeros((array.shape[0], num_of_classes))
    y_categorical[[i for i in range(array.shape[0])], array] = 1
    return y_categorical


def mnist_next_batch(x, y, batch_size = 128):
    batch_size = max(1, batch_size)
    total_size = x.shape[0]
    assert y.shape[0] == total_size
    x = x.reshape((total_size, -1))
    y = int_to_categorical(y, n_classes)
    index = 0
    while index < total_size:
        upper_index = min(index + batch_size, total_size)
        yield (x[index:upper_index], y[index:upper_index])
        index = upper_index


tf.reset_default_graph()
# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units, n_classes], dtype=tf.float64))
out_bias = tf.Variable(tf.random_normal([n_classes], dtype=tf.float64))

# defining placeholders
# input image placeholder
x = tf.placeholder(tf.float64, [None, time_steps, n_input])
# input label placeholder
y = tf.placeholder(tf.float64, [None, n_classes])

# processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
batch = tf.unstack(x, time_steps, 1)

# defining the network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, batch, dtype=tf.float64)

# converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
epoch = 1
while epoch <= 2:
    iteration = 1
    for batch_x, batch_y in mnist_next_batch(x_train, y_train, batch_size=batch_size):
        batch_x = batch_x.reshape((-1, time_steps, n_input))
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iteration % 1 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("Iteration: {}, Accuracy: {}, Loss: {}".format(iteration, acc, los))

        iteration = iteration + 1

    print("Epoch {} Finished.".format(epoch))
    epoch = epoch + 1

# calculating test accuracy
test_data = x_test.reshape((-1, time_steps, n_input))
test_label = int_to_categorical(y_test, n_classes)
print("Testing Accuracy: {}".format(sess.run(accuracy, feed_dict={x: test_data, y: test_label})))
sess.close()
