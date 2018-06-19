import tensorflow as tf
import numpy as np
import os
import dt_network
from functools import reduce

import matplotlib.pyplot as plt
import gc

def int_to_categorical(array, num_of_class):
    y_categorical = np.zeros((array.shape[0], num_of_class), dtype=np.int32)
    y_categorical[[i for i in range(array.shape[0])], array] = 1
    return y_categorical


def read_mnist_v2(path, flatten=True):
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


def get_mnist_dataset_v2(batch_size):
    # Step 1: Read in data
    mnist_folder = os.path.join('..', 'data', 'mnist.npz')
    train, test = read_mnist_v2(mnist_folder, flatten=False)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, test_data


def get_mnist_accuracy(test_dataset, train_output):
    iterator = tf.data.Iterator.from_structure(test_data.output_types, test_data.output_shapes)
    initializer = iterator.make_initializer(test_data)
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


if __name__ == '__main__':
    batch_size = 200
    tf.reset_default_graph()
    with tf.name_scope('data'):
        train_data, test_data = get_mnist_dataset_v2(batch_size)
    dec_tree_net = dt_network.DTNetwork(train_data, num_of_decisions=1,
                                        inference_network=dt_network.CNNSoftmaxClassifier,
                                        decision_network_args={'hidden_layers': []},
                                        loss_func=dt_network.softmax_cross_entropy,
                                        optimizer_func_args={'lr': 0.00005})
    # dec_tree_net = dt_network.DTNetwork(train_data, num_of_decisions=10,
    #                                     inference_network=dt_network.SoftmaxClassifier,
    #                                     inference_network_args={'hidden_layers': []},
    #                                     decision_network_args={'hidden_layers': []},
    #                                     loss_func=dt_network.softmax_cross_entropy,
    #                                     optimizer_func_args={'lr': 0.001})
    dec_tree_net.build()
    dec_tree_net.train(epochs=10, save_intervals=1)
    train_output, train_loss = dec_tree_net.eval(test_data)
    train_acc = get_mnist_accuracy(test_data, train_output)
    print(f'train loss: {train_loss}, train acc: {train_acc}')

    del dec_tree_net, train_data, test_data
    tf.reset_default_graph()
    gc.collect()

    path = os.path.join('..', 'data', 'mnist.npz')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = int_to_categorical(y_train, 10)
    y_test = int_to_categorical(y_test, 10)

    pred = np.concatenate(train_output, axis=0)
    pred_labels = np.argmax(pred, axis=1)
    real_labels = np.argmax(y_test, axis=1)
    label_err = pred_labels - real_labels
    idx = np.where(label_err != 0)[0]
    worst_err = np.abs(pred - y_test)[idx, real_labels[idx].astype(int)]
    worst_idx = np.flip(np.argsort(worst_err), axis=0)

    # A few of the wrong predicted digits
    for i in range(min(9, len(worst_err))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[idx[worst_idx[i]]].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title("class {}, pred {}".format(real_labels[idx[worst_idx[i]]], pred_labels[idx[worst_idx[i]]]))
    plt.tight_layout()
    plt.show()

    # pred_idx = []
    # threshold = 0.8
    # for i in range(pred.shape[0]):
    #     if all(pred[i] <= threshold):
    #         pred_idx.append(i)
    # pred_idx_num = 0
    # print(pred[pred_idx[pred_idx_num]])
    # testset_img_num = pred_idx[pred_idx_num]
    # plt.imshow(x_test[testset_img_num].reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title(f'class {real_labels[testset_img_num]}, pred {pred_labels[testset_img_num]}')
    # plt.show()
