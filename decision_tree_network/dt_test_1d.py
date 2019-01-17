import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
try:
    import dt_network
except Exception:
    from decision_tree_network import dt_network

if __name__ == '__main__':
    data_x = np.linspace(1, 10, 1000, dtype=np.float32).reshape((-1, 1))
    data_y = np.log(data_x)
    test_x = np.linspace(0.01, 30, 3000, dtype=np.float32).reshape((-1, 1))
    test_y = np.log(test_x)

    tf.reset_default_graph()
    dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y)).batch(data_x.shape[0])
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    initializer = iterator.make_initializer(dataset)
    inputs, targets = iterator.get_next()
    dt_net = dt_network.DTNetwork(dataset, num_of_decisions=1, inference_network=dt_network.AdvancedMLP,
                                  inference_network_args={'layers': [
                                      {'units': 256, 'activation': tf.nn.selu, 'use_bias': True,
                                       'kernel_initializer': tf.variance_scaling_initializer(),
                                       'bias_initializer': tf.variance_scaling_initializer()},
                                      {'units': data_y.shape[1], 'activation': tf.nn.selu, 'use_bias': True,
                                       'kernel_initializer': tf.variance_scaling_initializer(),
                                       'bias_initializer': tf.variance_scaling_initializer()}
                                  ]},
                                  optimizer_func_args={'learning_rate': 1})
    dt_net.build()
    dt_net.train(epochs=1000, retrain=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(test_x.shape[0])
    test_pred, _ = dt_net.eval(test_dataset)
    test_pred = np.concatenate(test_pred, axis=0)
    train_pred, _ = dt_net.eval(dataset)
    train_pred = np.concatenate(train_pred, axis=0)

    plt.subplot(2, 2, 1)
    plt.plot(data_x, data_y)
    plt.title('train')
    plt.subplot(2, 2, 2)
    plt.plot(test_x, test_y)
    plt.title('test')
    plt.subplot(2, 2, 3)
    plt.plot(data_x, train_pred)
    plt.title('train prediction')
    plt.subplot(2, 2, 4)
    plt.plot(test_x, test_pred)
    plt.title('test prediction')
    plt.tight_layout()
    plt.show()
