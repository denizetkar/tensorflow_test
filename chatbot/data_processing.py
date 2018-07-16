from sklearn.model_selection import train_test_split
from collections import Counter
import multiprocessing as mp
import pickle
import os
import re


def line_to_id_map_func(line):
    parts = line.strip().split(' +++$+++ ')
    if len(parts) == 5:
        return parts[0], parts[4]
    elif len(parts) == 4:
        return parts[0], ''


def get_lines(pool, file_path):
    line_id_map = pool.imap(line_to_id_map_func, open(file_path, 'r'), 4096)
    id2line = [i for i in line_id_map if i is not None]
    return dict(id2line)


def conv_line_to_list_map_func(line):
    parts = line.strip().split(' +++$+++ ')
    if len(parts) == 4:
        conv = []
        for line in parts[3][1:-1].split(', '):
            conv.append(line[1:-1])
        return conv


def get_convs(pool, file_path):
    convs = []
    conv_map = pool.imap(conv_line_to_list_map_func, open(file_path, 'r'), 4096)
    for conv in conv_map:
        if conv is not None:
            convs.append(conv)
    return convs


class ConvListToQaMapFunc:
    def __init__(self, id2line):
        self.id2line = id2line

    def __call__(self, conv):
        questions, answers = [], []
        for index, _ in enumerate(conv[:-1]):
            questions.append(self.id2line[conv[index]])
            answers.append(self.id2line[conv[index + 1]])
        return questions, answers


def question_answers(pool, id2line, convs):
    qa = pool.map(ConvListToQaMapFunc(id2line), convs)
    qa = tuple(zip(*qa))
    questions = [question for conv_questions in qa[0] for question in conv_questions]
    answers = [answer for conv_answers in qa[1] for answer in conv_answers]
    return questions, answers


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def prepare_dataset(questions, answers, processed_path, testset_size):
    # create path to store all the train & test encoder & decoder
    make_dir(processed_path)

    data_size = len(questions)
    splitted_data = train_test_split(questions, answers, train_size=data_size - testset_size)

    filenames = ['train.enc', 'test.enc', 'train.dec', 'test.dec']
    for i, filename in enumerate(filenames):
        with open(os.path.join(processed_path, filename), 'wb') as fp:
            pickle.dump(splitted_data[i], fp)


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def line_to_token_map_func(line):
    vocab = Counter()
    for token in basic_tokenizer(line):
        if token not in vocab:
            vocab[token] = 0
        vocab[token] += 1
    return vocab


def build_vocab(filename, processed_path, threshold, unk_sym, sos, eos):
    in_path = os.path.join(processed_path, filename)
    _, extension = os.path.splitext(filename)
    out_path = os.path.join(processed_path, 'vocab{}'.format(extension))
    vocab = {}
    for line in pickle.load(open(in_path, 'rb')):
        for token in basic_tokenizer(line):
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1
    vocab[unk_sym], vocab[sos], vocab[eos] = threshold, threshold, threshold
    i = 0
    for key in list(vocab):
        if vocab[key] < threshold:
            del vocab[key]
        else:
            vocab[key] = i
            i += 1
    with open(out_path, 'wb') as fp:
        pickle.dump(vocab, fp)


def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as fp:
        vocab = pickle.load(fp)
    return vocab


def sentence2id(vocab, line, unk_sym):
    return [vocab.get(token, vocab[unk_sym]) for token in basic_tokenizer(line)]


class SentenceToIdMapFunc:
    def __init__(self, vocab, unk_sym):
        self.vocab = vocab
        self.unk_sym = unk_sym

    def __call__(self, line):
        return sentence2id(self.vocab, line, self.unk_sym)


def token2id(pool, data, mode, processed_path, unk_sym):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    vocab = load_vocab(os.path.join(processed_path, vocab_path))
    in_file = open(os.path.join(processed_path, in_path), 'rb')
    out_file = open(os.path.join(processed_path, out_path), 'wb')

    line_ids = pool.map(SentenceToIdMapFunc(vocab, unk_sym), pickle.load(in_file))
    pickle.dump(line_ids, out_file)


class DataProcessor:
    def __init__(self):
        self.pool = mp.Pool(mp.cpu_count())

    def __enter__(self):
        return self

    def close(self):
        self.pool.close()
        self.pool.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def prepare_text_data(self, line_file_path, conv_file_path, processed_path, testset_size):
        id2line = get_lines(self.pool, line_file_path)
        convs = get_convs(self.pool, conv_file_path)
        questions, answers = question_answers(self.pool, id2line, convs)
        prepare_dataset(questions, answers, processed_path, testset_size)

    def process_data(self, processed_path, threshold, unk_sym, sos, eos):
        build_vocab('train.enc', processed_path, threshold, unk_sym, sos, eos)
        build_vocab('train.dec', processed_path, threshold, unk_sym, sos, eos)
        token2id(self.pool, 'train', 'enc', processed_path, unk_sym)
        token2id(self.pool, 'test', 'enc', processed_path, unk_sym)
        token2id(self.pool, 'train', 'dec', processed_path, unk_sym)
        token2id(self.pool, 'test', 'dec', processed_path, unk_sym)


def load_ids(processed_path, data, mode):
    with open(os.path.join(processed_path, data + '_ids.' + mode), 'rb') as fp:
        res = pickle.load(fp)
    return res


def load_text_data(processed_path, data, mode):
    text_data_path = os.path.join(processed_path, data + '.' + mode)
    with open(text_data_path, 'rb') as fp:
        res = pickle.load(fp)
    return res, text_data_path
