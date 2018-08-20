import os

import numpy as np

from util import save_json, line_reader


def down(_, downcase=False):
    return _.lower() if downcase else _


def remove_concept_marks(txt, marker1="BEG__", marker2="__END"):
    return txt.replace(marker1, "").replace(marker2, "")


class VocabBuild():
    def __init__(self, filename, sep=" ", downcase=False):
        self.filename = filename
        self.sep = sep
        self.downcase = downcase  # whether the embs have been lowercased

        self.w_index = {}
        self.inv_w_index = {}
        self.W = None

    def read(self):
        """
        Reads word2vec-format embeddings.
        """
        ws = []
        with open(self.filename) as in_f:
            m, n = map(eval, in_f.readline().strip().split())
        e_m = np.zeros((m, n))
        for c, l in enumerate(line_reader(self.filename, skip=1)):  # skip dimensions
            w, *e = l.strip().split()
            #assert len(e) == n
            if len(e) != n:
                print("Incorrect embedding dimension, skipping.")
                continue
            if not w or not e:
                print("Empty w or e.")
            ws.append(w)
            e_m[c] = e
        #assert len(ws) == e_m.shape[0]
        self.w_index = {w: c for c, w in enumerate(ws)}
        self.inv_w_index = {v: k for k, v in self.w_index.items()}
        self.W = e_m

    def lookup(self, w, output_nan=False):
        if down(w, self.downcase) in self.w_index:
            idx = self.w_index[down(w, self.downcase)]
        else:
            if output_nan:
                idx = 0
            else:
                idx = None

        return idx

    def line_to_seq(self, line, output_nan=False):
        seq = []
        for w in line.strip().split(self.sep):
            idx = self.lookup(w, output_nan=output_nan)
            if idx is None:
                continue
            seq.append(idx)

        return seq

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_json(self.w_index, "{}/w_index.json".format(save_dir))
        save_json(self.inv_w_index, "{}/inv_w_index.json".format(save_dir))
