import os
import time
import random
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mnist_utils
if __name__ != '__main__':
    os.chdir(os.path.join('.', 'RNN'))


def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]


def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])


def read_data_lines(filename, vocab):
    lines = [line.strip() for line in open(filename, 'r', encoding='utf8').readlines()]
    lines.sort(key=len, reverse=True)
    while True:
        # random.shuffle(lines)
        for text in lines:
            text = vocab_encode(text, vocab)
            yield text


def read_batch(line_stream, batch_size):
    batch = []
    for line in line_stream:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


def batch_zero_padding(batch_stream):
    for batch in batch_stream:
        max_len = max([len(line) for line in batch])
        for line in batch:
            line += [0] * (max_len - len(line))
        yield batch


class CharRNN(object):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        if 'trump' in model:
            self.vocab = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          " '\"_abcdefghijklmnopqrstuvwxyz{|}@#➡📈")
        else:
            self.vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          "\\^_abcdefghijklmnopqrstuvwxyz{|}")

        self.seq = tf.placeholder(tf.int32, [None, None])
        self.temp = tf.constant(1.5)
        self.hidden_sizes = [128, 128]
        self.batch_size = 64
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.lr = 0.0003
        self.skip_step = 100
        self.len_generated = 200
        self.in_state, self.output, self.out_state, self.logits = None, None, None, None
        self.loss, self.sample, self.opt = None, None, None

    def create_rnn(self, seq):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
        # layers = [tf.nn.rnn_cell.DropoutWrapper(layer, output_keep_prob=0.75, variational_recurrent=True,
        #                                         dtype=tf.float32) for layer in layers]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        batch_size = tf.shape(seq)[0]
        self.in_state = cells.zero_state(batch_size, dtype=tf.float32)
        # this line to calculate the real length of seq
        # all seq are padded to be of the same length
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state, parallel_iterations=64)

    def create_model(self):
        seq = tf.one_hot(self.seq, len(self.vocab))
        self.create_rnn(seq)
        self.logits = tf.layers.dense(self.output, len(self.vocab), None)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits[:, :-1],
                                                          labels=seq[:, 1:])
        self.loss = tf.reduce_sum(loss)
        # sample the next character from Maxwell-Boltzmann Distribution with temperature temp.
        # self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0]
        self.sample = tf.argmax(self.logits[:, -1], 1)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def train(self):
        saver = tf.train.Saver()
        start = time.time()
        checkpoint_name = 'checkpoints/' + self.model + '/char-rnn'
        min_loss = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.model + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            line_stream = read_data_lines(self.path, self.vocab)
            batch_stream = read_batch(line_stream, self.batch_size)
            data = batch_zero_padding(batch_stream)
            while True:
                batch = next(data)
                batch_loss, _ = sess.run([self.loss, self.opt], {self.seq: batch})
                if (iteration + 1) % self.skip_step == 0:
                    print(f'Iter {iteration}. \n    Loss {batch_loss}. Time {time.time() - start}. Min Loss {min_loss}')
                    self.online_infer(sess)
                    start = time.time()
                    saver.save(sess, checkpoint_name)
                    if min_loss is None:
                        min_loss = batch_loss
                    elif batch_loss < min_loss:
                        min_loss = batch_loss
                iteration += 1

    def online_infer(self, sess):
        """ Generate sequence one character at a time, based on the previous character
        """
        for seed in ['Hillary', 'I', 'R', 'T', '@', 'N', 'M', '.', 'G', 'A', 'W']:
            sentence = seed
            batch = [vocab_encode(sentence, self.vocab)]
            state = None
            for _ in range(self.len_generated):
                feed = {self.seq: batch}
                if state is not None:  # for the first decoder step, the state is None
                    for i in range(len(state)):
                        feed[self.in_state[i]] = state[i]
                index, state = sess.run([self.sample, self.out_state], feed)
                sentence += vocab_decode(index, self.vocab)
                batch = [vocab_encode(sentence[-1], self.vocab)]
            print('\t' + sentence)


if __name__ == '__main__':
    model = 'trump_tweets'
    mnist_utils.safe_mkdir('checkpoints')
    mnist_utils.safe_mkdir('checkpoints/' + model)

    lm = CharRNN(model, os.path.join('..', 'data', model + '.text'))
    lm.create_model()
    lm.train()
