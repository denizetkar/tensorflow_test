import tensorflow as tf
import numpy as np
from functools import reduce
import os


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def int_to_categorical(array, num_of_class):
    y_categorical = np.zeros((array.shape[0], num_of_class), dtype=np.int32)
    y_categorical[[i for i in range(array.shape[0])], array] = 1
    return y_categorical


def read_mnist(path, flatten=True):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    if flatten:
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
    else:
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = int_to_categorical(y_train, 10)
    y_test = int_to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def get_mnist_dataset(batch_size):
    # Step 1: Read in data
    mnist_folder = os.path.join('..', 'data', 'mnist.npz')
    train, test = read_mnist(mnist_folder, flatten=False)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, test_data


def get_mnist_accuracy(test_dataset, train_output):
    iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    initializer = iterator.make_initializer(test_dataset)
    inputs, targets = iterator.get_next()
    inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x * y, inputs.shape[1:])])
    targets = tf.reshape(targets, [-1, reduce(lambda x, y: x * y, targets.shape[1:])])
    with tf.Session() as sess:
        sess.run(initializer)
        total_correct_preds = 0
        batch_num = 0
        total_size = 0
        try:
            while True:
                batch_target = sess.run(targets)
                batch_output = train_output[batch_num]
                correct_pred = np.equal(np.argmax(batch_target, 1), np.argmax(batch_output, 1))
                batch_accuracy = np.sum(correct_pred.astype(np.float32))
                total_correct_preds += batch_accuracy
                batch_num += 1
                total_size += batch_target.shape[0]
        except tf.errors.OutOfRangeError:
            pass
    return total_correct_preds/total_size
