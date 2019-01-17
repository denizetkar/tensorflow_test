import tensorflow as tf
import multiprocessing as mp
import os


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


class Seq2Seq:
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, unk_sym='<unk>',
                 sos='<s>', eos='<\s>', batch_size=128, embedding_size=512, num_layer=2, beam_width=5,
                 dropout=0.1, output_buffer_size=None, num_cores=mp.cpu_count(),
                 length_penalty_weight=1.0, learning_rate=0.001, decay_rate=0.99, decay_step=None,
                 max_gradient_norm=5.0, name='seq2seq_network'):
        if output_buffer_size is None:
            output_buffer_size = batch_size * 2
        if decay_step is None:
            decay_step = len(src_data)//batch_size
        decay_step = max(1, decay_step)
        self.src_data, self.tgt_data = src_data, tgt_data

        self.src_eos_id = src_vocab[eos]
        self.tgt_sos_id = tgt_vocab[sos]
        self.tgt_eos_id = tgt_vocab[eos]
        self.src_vocab_size = len(src_vocab)
        self.target_vocab_size = len(tgt_vocab)

        self.batch_size, self.embedding_size = batch_size, embedding_size
        self.num_layer, self.beam_width = max(1, num_layer), beam_width
        self.dropout, self.output_buffer_size = dropout, output_buffer_size
        self.num_cores, self.length_penalty_weight = num_cores, length_penalty_weight
        self.learning_rate, self.decay_rate = learning_rate, decay_rate
        self.decay_step, self.max_gradient_norm, self.name = decay_step, max_gradient_norm, name

    def read_data_line(self):
        for src, tgt in zip(self.src_data, self.tgt_data):
            tgt_in, tgt_out = [self.tgt_sos_id], tgt.copy()
            tgt_in.extend(tgt)
            tgt_out.append(self.tgt_eos_id)
            yield src, tgt_in, tgt_out

    def read_batch(self, data_stream):
        src_ids, tgt_input_ids, tgt_output_ids = [], [], []
        for src, tgt_in, tgt_out in data_stream:
            src_ids.append(src)
            tgt_input_ids.append(tgt_in)
            tgt_output_ids.append(tgt_out)
            if len(src_ids) == self.batch_size:
                yield src_ids, tgt_input_ids, tgt_output_ids
                src_ids, tgt_input_ids, tgt_output_ids = [], [], []
        if len(src_ids) > 0:
            yield src_ids, tgt_input_ids, tgt_output_ids

    def batch_processing(self, batch_stream):
        for src_ids, tgt_input_ids, tgt_output_ids in batch_stream:
            max_src_len = max([len(line) for line in src_ids])
            max_tgt_len = max([len(line) for line in tgt_input_ids])
            for src in src_ids:
                src.extend([self.src_eos_id] * (max_src_len - len(src)))
            for tgt_in in tgt_input_ids:
                tgt_in.extend([self.tgt_eos_id] * (max_tgt_len - len(tgt_in)))
            for tgt_out in tgt_output_ids:
                tgt_out.extend([self.tgt_eos_id] * (max_tgt_len - len(tgt_out)))
            yield src_ids, tgt_input_ids, tgt_output_ids

    def build_placeholders(self):
        self.src_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.tgt_input_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.tgt_output_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.src_seq_len = tf.tile(tf.reshape(tf.shape(self.src_ids)[1], [1]),
                                   tf.reshape(tf.shape(self.src_ids)[0], [1]))
        self.tgt_seq_len = tf.tile(tf.reshape(tf.shape(self.tgt_input_ids)[1], [1]),
                                   tf.reshape(tf.shape(self.tgt_input_ids)[0], [1]))

    def build_encoder_embedding(self):
        with tf.variable_scope('enc_embed', reuse=tf.AUTO_REUSE):
            self.embedding_encoder = tf.get_variable("embedding_encoder",
                                                     [self.src_vocab_size, self.embedding_size])
            src_ids = tf.transpose(self.src_ids)
            self.encoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_encoder, src_ids)

    def build_encoder(self, is_train=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            self.forward_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
            self.backward_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
            if is_train and self.dropout > 0.0:
                self.forward_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=self.forward_cell, input_keep_prob=(1.0 - self.dropout))
                self.backward_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=self.backward_cell, input_keep_prob=(1.0 - self.dropout))
            if self.num_layer > 1:
                self.forward_cell = tf.contrib.rnn.MultiRNNCell([self.forward_cell for _ in range(self.num_layer)])
                self.backward_cell = tf.contrib.rnn.MultiRNNCell([self.backward_cell for _ in range(self.num_layer)])

            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                self.forward_cell, self.backward_cell, self.encoder_emb_inp,
                sequence_length=self.src_seq_len, dtype=tf.float32,
                parallel_iterations=self.num_cores, time_major=True)
            self.encoder_outputs = tf.concat(bi_outputs, -1)
            self.encoder_state = tf.concat(bi_state, -1)

    def build_decoder_embedding(self):
        with tf.variable_scope('dec_embed', reuse=tf.AUTO_REUSE):
            self.embedding_decoder = tf.get_variable("embedding_decoder",
                                                     [self.target_vocab_size, self.embedding_size * 2])
            tgt_input_ids = tf.transpose(self.tgt_input_ids)
            self.decoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_decoder, tgt_input_ids)

    def build_decoder_cell(self, is_train=True):
        # Ensure memory is batch-major
        memory = tf.transpose(self.encoder_outputs, [1, 0, 2])
        if not is_train and self.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=self.beam_width)
            src_seq_len = tf.contrib.seq2seq.tile_batch(
                self.src_seq_len, multiplier=self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(
                self.encoder_state, multiplier=self.beam_width)
            batch_size = self.batch_size * self.beam_width
        else:
            src_seq_len = self.src_seq_len
            encoder_state = self.encoder_state
            batch_size = self.batch_size
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.embedding_size * 2,
            memory,
            memory_sequence_length=src_seq_len,
            scale=True)
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size * 2)
        if is_train and self.dropout > 0.0:
            self.decoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=self.decoder_cell, input_keep_prob=(1.0 - self.dropout))
        if self.num_layer > 1:
            self.decoder_cell = tf.contrib.rnn.MultiRNNCell([self.decoder_cell for _ in range(self.num_layer)])
        alignment_history = (not is_train and self.beam_width == 0)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell,
            attention_mechanism,
            attention_layer_size=self.embedding_size * 2,
            alignment_history=alignment_history,
            output_attention=True,
            name="attention")
        self.decoder_initial_state = self.decoder_cell.zero_state(tf.shape(self.src_ids)[0], tf.float32).clone(
            cell_state=self.encoder_state)

    def build_decoder(self, is_train=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.build_decoder_cell(is_train=is_train)
            self.output_layer = tf.layers.Dense(
                self.target_vocab_size, use_bias=False, name="output_projection")
            if is_train:
                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_emb_inp, self.tgt_seq_len,
                    time_major=True)
                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.decoder_cell,
                    helper,
                    self.decoder_initial_state)
                # Dynamic decoding
                self.outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=True,
                    swap_memory=True)
                self.sample_id = self.outputs.sample_id
                self.logits = self.output_layer(self.outputs.rnn_output)
            else:
                start_tokens = tf.fill([self.batch_size], self.tgt_sos_id)
                end_token = self.tgt_eos_id
                if self.beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=self.length_penalty_weight)
                else:
                    helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                        self.embedding_decoder, start_tokens, end_token)
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        self.decoder_cell,
                        helper,
                        self.decoder_initial_state,
                        output_layer=self.output_layer  # applied per timestep
                    )
                # Dynamic decoding
                self.outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=True,
                    swap_memory=True)
                if self.beam_width > 0:
                    self.logits = tf.no_op()
                    self.sample_id = self.outputs.predicted_ids
                else:
                    self.logits = self.outputs.rnn_output
                    self.sample_id = self.outputs.sample_id

    def build_loss(self):
        tgt_output_ids = tf.transpose(self.tgt_output_ids)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tgt_output_ids, logits=self.logits)
        target_weights = tf.sequence_mask(
            self.tgt_seq_len, dtype=self.logits.dtype)
        target_weights = tf.transpose(target_weights)
        self.train_loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
        tf.summary.scalar('train_loss', self.train_loss)

    def build_training_step(self):
        # warmup_factor = tf.exp(tf.log(0.01) / self.warmup_steps)
        # inv_decay = warmup_factor ** (
        #     tf.to_float(self.warmup_steps - self.global_step))
        # adv_learning_rate = tf.cond(
        # self.global_step < self.warmup_steps,
        # lambda: inv_decay * self.learning_rate,
        # lambda: self.learning_rate,
        # name="learning_rate_warump_cond")
        epsilon = tf.constant(self.learning_rate / 1000.0)
        decay_factor = tf.pow(tf.constant(self.decay_rate), tf.cast(self.global_step//self.decay_step, tf.float32))
        adv_learning_rate = decay_factor * (self.learning_rate - epsilon) + epsilon
        tf.summary.scalar('learning_rate', adv_learning_rate)
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(adv_learning_rate)
        # Gradients
        gradients = tf.gradients(
            self.train_loss,
            params,
            colocate_gradients_with_ops=True)
        clipped_grads, grad_norm = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.update = opt.apply_gradients(
            zip(clipped_grads, params), global_step=self.global_step)

    def build(self, is_train=True):
        tf.reset_default_graph()
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if is_train:
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
                self.build_placeholders()
                self.build_encoder_embedding()
                self.build_encoder()
                self.build_decoder_embedding()
                self.build_decoder()
                self.build_loss()
                self.build_training_step()
                self.merged = tf.summary.merge_all()
            else:
                pass

    def train(self, epochs=10, verbose=True, save_intervals=None, retrain=False, reset_g_step=False):
        make_dir('checkpoints')
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('graphs')    # , sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            if not retrain:
                ckpt = tf.train.get_checkpoint_state('checkpoints')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
            if reset_g_step:
                sess.run(self.global_step.assign(0))

            for epoch in range(self.global_step.eval(), epochs):
                total_loss = 0
                n_batch = 0
                data_stream = self.read_data_line()
                batch_stream = self.read_batch(data_stream)
                batch_generator = self.batch_processing(batch_stream)
                for src, tgt_in, tgt_out in batch_generator:
                    summary, batch_loss, _ = sess.run([self.merged, self.train_loss, self.update],
                                                      feed_dict={self.src_ids: src,
                                                                 self.tgt_input_ids: tgt_in,
                                                                 self.tgt_output_ids: tgt_out})
                    writer.add_summary(summary, global_step=self.global_step.eval())
                    total_loss += batch_loss
                    n_batch += 1
                if verbose:
                    print(f'Average loss at epoch {epoch+1}/{epochs}: {total_loss/n_batch}')
                if epoch == (epochs-1) or (save_intervals is not None and epoch % save_intervals == 0):
                    saver.save(sess, 'checkpoints/seq2seq_network', write_meta_graph=False)
        writer.close()

    def eval(self):
        # TODO: complete eval method
        pass
