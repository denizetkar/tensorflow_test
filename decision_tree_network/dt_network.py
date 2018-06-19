import tensorflow as tf
# import numpy as np
import os
from functools import reduce


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


HUBER_LOSS = tf.losses.huber_loss


def softmax_cross_entropy(labels, predictions, epsilon=1e-10):
    predictions = tf.clip_by_value(predictions, epsilon, 1-epsilon)
    logits = tf.log(predictions)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    return tf.reduce_mean(entropy)


class AdvancedMLP(object):
    """ Using tf.nn.selu over tf.nn.relu is adviced for accuracy."""
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self._layers = list(layers)

    def build(self, inputs, units=None):
        """ Do not care about 'units' parameter since this
            class was meant to be advanced."""
        net = tf.reshape(inputs, [-1, reduce(lambda x, y: x * y, inputs.shape[1:])])
        for i, layer in enumerate(self._layers):
            net = tf.layers.dense(net, name=f'dense_layer{i}', **layer)
        return net


class MLP(AdvancedMLP):
    def __init__(self, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = []
        self._hidden_layers = list(hidden_layers)
        layers = []
        for hidden_layer in self._hidden_layers:
            layers.append({'units': hidden_layer, 'activation': tf.nn.relu, 'use_bias': True,
                           'kernel_initializer': tf.variance_scaling_initializer(),
                           'bias_initializer': tf.variance_scaling_initializer()})
        super().__init__(layers)

    def build(self, inputs, units=None):
        net = super().build(inputs, units)
        if units is not None:
            net = tf.layers.dense(net, units, activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer=tf.variance_scaling_initializer(),
                                  bias_initializer=tf.variance_scaling_initializer(),
                                  name='last_dense_layer')
        return net


class SoftmaxClassifier(MLP):
    def __init__(self, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = []
        super().__init__(hidden_layers)

    def build(self, inputs, num_of_class):
        net = super().build(inputs)
        net = tf.layers.dense(net, num_of_class, activation=tf.identity, use_bias=True,
                              kernel_initializer=tf.variance_scaling_initializer(),
                              bias_initializer=tf.variance_scaling_initializer(), name='pre_soft_max')
        net = tf.nn.softmax(net, name='soft_max')
        return net


class CNNSoftmaxClassifier(object):
    def __init__(self, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = [{'type': 'conv2d'}, {'type': 'relu'}, {'type': 'max2d'}, {'type': 'fc'}]
        for layer in hidden_layers:
            if not isinstance(layer.get('type', None), str):
                layer['type'] = ''
            if layer.get('args', None) is None:
                layer['args'] = {}
        self._hidden_layers = list(hidden_layers)

    def build(self, inputs, fc_units=None):
        net = inputs
        for i, hidden_layer in enumerate(self._hidden_layers):
            if hidden_layer['type'] == 'conv2d':
                try:
                    net = tf.layers.conv2d(net, name=f'conv2d_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.conv2d(net, 32, 3, padding='same', name=f'conv2d_{i}')
            elif hidden_layer['type'] == 'conv3d':
                try:
                    net = tf.layers.conv3d(net, name=f'conv3d_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.conv3d(net, 32, 3, padding='same', name=f'conv3d_{i}')
            elif hidden_layer['type'] == 'relu':
                try:
                    net = tf.nn.relu(net, name=f'relu_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.nn.relu(net, name=f'relu_{i}')
            elif hidden_layer['type'] == 'max2d':
                try:
                    net = tf.layers.max_pooling2d(net, name=f'max2d_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.max_pooling2d(net, 2, 2, padding='same', name=f'max2d_{i}')
            elif hidden_layer['type'] == 'max3d':
                try:
                    net = tf.layers.max_pooling3d(net, name=f'max3d_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.max_pooling3d(net, 2, 2, padding='same', name=f'max3d_{i}')
            elif hidden_layer['type'] == 'avg2d':
                try:
                    net = tf.layers.average_pooling2d(net, name=f'avg2d_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.average_pooling2d(net, 2, 2, padding='same', name=f'avg2d_{i}')
            elif hidden_layer['type'] == 'avg3d':
                try:
                    net = tf.layers.average_pooling3d(net, name=f'avg3d_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.average_pooling3d(net, 2, 2, padding='same', name=f'avg3d_{i}')
            elif hidden_layer['type'] == 'fc':
                net = tf.reshape(net, [-1, reduce(lambda x, y: x*y, net.shape[1:])])
                try:
                    net = tf.layers.dense(net, name=f'dense_layer_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.dense(net, 512, activation=tf.nn.relu,
                                          use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                                          bias_initializer=tf.variance_scaling_initializer(),
                                          name=f'dense_layer_{i}')
            elif hidden_layer['type'] == 'dropout':
                try:
                    net = tf.layers.dropout(net, name=f'dropout_{i}', **hidden_layer['args'])
                except TypeError:
                    net = tf.layers.dropout(net, rate=0.2, name=f'dropout_{i}')
            else:
                # nothing to do here
                pass
        if fc_units is not None:
            net = tf.layers.dense(net, fc_units, activation=tf.identity, use_bias=True,
                                  kernel_initializer=tf.variance_scaling_initializer(),
                                  bias_initializer=tf.variance_scaling_initializer(), name='pre_soft_max')
            net = tf.nn.softmax(net, name='soft_max')
        return net


class InferenceNetwork(object):
    def __init__(self, inputs, units, name, network=MLP()):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._net = network.build(inputs, units)

    def get_network(self):
        return self._net


class DecisionNetwork(InferenceNetwork):
    def __init__(self, inputs, num_of_decisions, name, network=SoftmaxClassifier()):
        super().__init__(inputs, num_of_decisions, name, network)


class DTNetwork(object):
    """ Use tf.data.BatchDataset for providing a dataset and make sure
        the data shape is formatted as (batch_size, ...).
        If 'num_of_decision' is not >1 then Decision Tree Network is
        equivalent to a single Inference Network."""
    def __init__(self, io_dataset, name='dec_tree_network', num_of_decisions=None,
                 inference_network=MLP, inference_network_args=None,
                 decision_network=SoftmaxClassifier, decision_network_args=None,
                 loss_func=HUBER_LOSS, loss_func_args=None,
                 optimizer_func=tf.train.AdamOptimizer, optimizer_func_args=None):
        if inference_network_args is None:
            inference_network_args = {}
        if decision_network_args is None:
            decision_network_args = {}
        if loss_func_args is None:
            loss_func_args = {}
        if optimizer_func_args is None:
            optimizer_func_args = {}
        self.iterator = tf.data.Iterator.from_structure(io_dataset.output_types, io_dataset.output_shapes)
        self.initializer = self.iterator.make_initializer(io_dataset)
        self.inputs, self.targets = self.iterator.get_next()
        self.name = name
        if num_of_decisions is None or num_of_decisions < 1:
            self.num_of_decisions = 1
        else:
            self.num_of_decisions = int(num_of_decisions)
        self.inference_network = inference_network
        self.inference_network_args = inference_network_args
        self.decision_network = decision_network
        self.decision_network_args = decision_network_args
        self.loss_func = loss_func
        self.loss_func_args = loss_func_args
        self.optimizer_func = optimizer_func
        self.optimizer_func_args = optimizer_func_args
        self.target_dim = self.targets.shape[1].value
        self.g_step = None
        self.decision = None
        self.output = None
        self.loss = None
        self.merged = None
        self.training_step = None

    def build(self):
        inference_net_list = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.g_step = tf.Variable(0, trainable=False, dtype=tf.int32)
            for i in range(self.num_of_decisions):
                try:
                    i_net = InferenceNetwork(self.inputs, self.target_dim, f'inference{i}',
                                             network=self.inference_network(**self.inference_network_args)
                                             ).get_network()
                except TypeError:
                    i_net = InferenceNetwork(self.inputs, self.target_dim, f'inference{i}',
                                             network=self.inference_network()).get_network()
                inference_net_list.append(i_net)

            try:
                self.decision = DecisionNetwork(self.inputs, self.num_of_decisions, 'decision',
                                                network=self.decision_network(**self.decision_network_args)
                                                ).get_network()
            except TypeError:
                self.decision = DecisionNetwork(self.inputs, self.num_of_decisions, 'decision',
                                                network=self.decision_network()).get_network()
            concat_output = tf.stack(inference_net_list, axis=1)
            if self.num_of_decisions > 1:
                self.output = tf.reduce_sum(concat_output * tf.reshape(self.decision, [-1, self.decision.shape[1]] + [
                    1] * len(concat_output.shape[2:])), axis=1, name='decided_output')
            else:
                self.output = tf.reduce_sum(concat_output, axis=1, name='decided_output')

            try:
                self.loss = self.loss_func(self.targets, self.output, **self.loss_func_args)
            except TypeError:
                self.loss = self.loss_func(self.targets, self.output)
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()
            try:
                optimizer = self.optimizer_func(**self.optimizer_func_args)
            except TypeError:
                optimizer = self.optimizer_func()

            self.training_step = optimizer.minimize(self.loss)

    def eval(self, test_io_dataset):
        test_initializer = self.iterator.make_initializer(test_io_dataset)
        total_output = []
        total_loss = 0
        n_batch = 0
        with tf.Session() as sess:
            sess.run([test_initializer, tf.global_variables_initializer()])
            saver = tf.train.Saver(tf.trainable_variables())
            ckpt = tf.train.get_checkpoint_state('checkpoints')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            try:
                while True:
                    batch_output, batch_loss = sess.run([self.output, self.loss])
                    total_output.append(batch_output)
                    total_loss += batch_loss
                    n_batch += 1
            except tf.errors.OutOfRangeError:
                pass
        # total_output = np.concatenate(total_output, axis=0)
        return total_output, total_loss/n_batch

    def train(self, epochs=10, verbose=True, save_intervals=None, retrain=False):
        safe_mkdir('checkpoints')
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('graphs', sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.all_variables())
            if not retrain:
                ckpt = tf.train.get_checkpoint_state('checkpoints')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

            increment_g_step = self.g_step.assign_add(1)
            for epoch in range(epochs):
                sess.run(self.initializer)
                total_loss = 0
                n_batch = 0
                try:
                    while True:
                        summary, batch_loss, _ = sess.run([self.merged, self.loss, self.training_step])
                        writer.add_summary(summary, global_step=self.g_step.eval())
                        sess.run(increment_g_step)
                        total_loss += batch_loss
                        n_batch += 1
                except tf.errors.OutOfRangeError:
                    pass
                if verbose:
                    print(f'Average loss at epoch {epoch+1}/{epochs}: {total_loss/n_batch}')
                if epoch == (epochs-1) or (save_intervals is not None and epoch % save_intervals == 0):
                    saver.save(sess, 'checkpoints/dt_network', write_meta_graph=False)
        writer.close()
