import tensorflow as tf
from data_processing import DataProcessor
from model import Seq2Seq
import data_processing
import argparse
import os

# parameters for processing the dataset
DATA_PATH = '../data/cornell movie-dialogs corpus'
CONV_FILE = 'movie_conversations.text'
LINE_FILE = 'movie_lines.text'
PROCESSED_PATH = 'processed'

TESTSET_SIZE = 25000
THRESHOLD = 2

NUM_LAYERS = 1
HIDDEN_SIZE = 256
BATCH_SIZE = 64

UNK_SYM = '<unk>'
SOS = '<s>'
EOS = '<\s>'


def data_preprocess():
    with DataProcessor() as dp:
        line_file_path = os.path.join(DATA_PATH, LINE_FILE)
        conv_file_path = os.path.join(DATA_PATH, CONV_FILE)
        dp.prepare_text_data(line_file_path, conv_file_path, PROCESSED_PATH, TESTSET_SIZE)
        dp.process_data(PROCESSED_PATH, THRESHOLD, UNK_SYM, SOS, EOS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if not os.path.isdir(PROCESSED_PATH):
        data_preprocess()
    if args.mode == 'train':
        vocab_enc = data_processing.load_vocab(os.path.join(PROCESSED_PATH, 'vocab.enc'))
        vocab_dec = data_processing.load_vocab(os.path.join(PROCESSED_PATH, 'vocab.dec'))
        train_enc = data_processing.load_ids(PROCESSED_PATH, 'train', 'enc')
        train_dec = data_processing.load_ids(PROCESSED_PATH, 'train', 'dec')
        model = Seq2Seq(train_enc, train_dec, vocab_enc, vocab_dec, UNK_SYM, SOS, EOS,
                        batch_size=BATCH_SIZE, embedding_size=HIDDEN_SIZE, num_layer=NUM_LAYERS)
        model.build()
        model.train(epochs=5, save_intervals=1, reset_g_step=True)
    elif args.mode == 'chat':
        # TODO: complete chat mode
        pass
