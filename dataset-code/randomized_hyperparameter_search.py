import random

import numpy as np


def embedding_parameter_space():
    return {
        #"domain": ["clinical", "mixed_news_clinical", "pubmed", "pubmed_wiki"],
        #"dimension": list(range(50, 850, 50)),
        "dimension": [200],
        "win_size": list(range(1, 11)),
        #"min_freq": list(range(1, 11)),
        "min_freq": [200],
        "neg_samples": list(range(1, 11)),
        "model": ["skipgram", "cbow"]
    }


def stanford_reader_parameter_space():
    return {
    "hidden_size": (16, 160),
    "dropout_rate": (0.2, 0.9999)
    #"optimizer": ["sgd", "rmsprop"]
    #"grad_clipping":
    }


class RandomizedSearch:
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space()

    def size_parameter_space(self):
        size = 1
        for param, val in self.parameter_space.items():
            size *= len(val)
        return size

    def sample(self):
        sampled = {}
        for param, val in self.parameter_space.items():
            if isinstance(val, list):
                sampled[param] = self.draw(val)
            elif isinstance(val, tuple) and len(val) == 2:
                assert val[1] >= val[0]
                if param == "dropout_rate":
                    sampled[param] = self.uniform_draw(val[0], val[1])
                else:
                    # use Bergstra and Bengio's geometric draw
                    sampled[param] = self.geometric_draw(val[0], val[1])
            else:
                raise TypeError

        return sampled

    def draw(self, lst):
        return random.choice(lst)

    def uniform_draw(self, min_val, max_val):
        return np.float32(np.random.uniform(min_val, max_val))

    def geometric_draw(self, min_val, max_val):
        def formatted(x):
            assert type(min_val) == type(max_val)
            if isinstance(min_val, int):
                return int(x)
            elif isinstance(min_val, float):
                return np.float32(x)
            else:
                raise TypeError

        return formatted(np.exp(np.random.uniform(np.log(min_val), np.log(max_val))))
