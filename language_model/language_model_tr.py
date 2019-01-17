from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

from . import lang_data


@registry.register_problem
class LanguageModelTr(text_problems.Text2SelfProblem):

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        for token in lang_data.token_generator():
            yield {'targets': token}

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return False

    @property
    def num_generate_tasks(self):
        pass

    def prepare_to_generate(self, data_dir, tmp_dir):
        pass

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER
