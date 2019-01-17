from os import listdir
from os.path import isfile, join

DATA_PATH = r'D:\Users\deniz\Desktop\DOSYALAR\workspace\tensorflow_test\data\docs_tr'


def token_generator():
    path_list = [join(DATA_PATH, f) for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and f.endswith('.text')]
    for doc_path in path_list:
        with open(doc_path, 'r', encoding="utf8") as doc_f:
            yield doc_f.read()


if __name__ == '__main__':
    with open(join(DATA_PATH, 'corpus1.text'), 'r') as in_f, open(join(DATA_PATH, 'corpus.text', 'w')) as out_f:
        pass
