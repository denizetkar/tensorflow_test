import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import mnist_utils
if __name__ != '__main__':
    os.chdir(os.path.join('.', 'VAE'))


def make_encoder(images, code_size=2):
    images = tf.layers.flatten(images)
    hidden = tf.layers.dense(images, 512, tf.nn.elu)
    mean = tf.layers.dense(hidden, code_size, tf.nn.elu)
    return mean


def make_decoder(code, data_shape=[28, 28, 1]):
    hidden = tf.layers.dense(code, 512, tf.nn.elu)
    logit = tf.layers.dense(hidden, np.prod(data_shape), tf.nn.elu)
    logit = tf.reshape(logit, [-1] + data_shape)
    return logit


if __name__ == '__main__':
    mnist_folder = os.path.join('..', 'data', 'mnist.npz')
    (train_x, train_y), (_, _) = mnist_utils.read_mnist(mnist_folder, flatten=False)
    train_y = np.argmax(train_y, axis=1)
    train_x_0 = train_x
    # np.random.shuffle(train_x_0)
    train_x_0 = train_x_0[:min(len(train_x_0), 1000)]

    code_size = 100
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])

    encoder = make_encoder(inputs, code_size)
    output = make_decoder(encoder)
    loss = tf.losses.huber_loss(inputs, output)
    optimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    mnist_utils.safe_mkdir('checkpoints')
    retrain = True
    epochs = 1000
    save_intervals = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if not retrain:
            ckpt = tf.train.get_checkpoint_state('checkpoints')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        for epoch in range(epochs):
            total_loss = 0
            n_batch = 0
            try:
                batch_loss, _ = sess.run([loss, optimize], feed_dict={inputs: train_x_0})
                total_loss += batch_loss
                n_batch += 1
            except tf.errors.OutOfRangeError:
                pass
            print(f'Average loss at epoch {epoch+1}/{epochs}: {total_loss/n_batch}')
            if epoch == (epochs - 1) or (save_intervals is not None and epoch % save_intervals == 0):
                saver.save(sess, 'checkpoints/vae_network', write_meta_graph=False)
        my_samples = sess.run(output, feed_dict={inputs: train_x_0})

    img_num = 111
    plt.subplot(1, 2, 1)
    plt.imshow(train_x_0.reshape((-1, 28, 28))[img_num], cmap='gray')
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(my_samples.reshape((-1, 28, 28))[img_num], cmap='gray')
    plt.title('compressed image')
    plt.tight_layout()
    plt.show()
