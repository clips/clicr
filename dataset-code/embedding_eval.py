import numpy as np

from text import VocabBuild
from util import cosines


class EmbeddingEval:
    def __init__(self, ground, predictions, embeddings_file, downcase):
        self.eval = {}
        self.ground = ground
        self.predictions = predictions
        self.embeddings_file = embeddings_file
        self.v = VocabBuild(embeddings_file, downcase=downcase)
        self.v.read()

    def evaluate(self):
        assert len(self.ground) == len(self.predictions)
        scorers = [(Average(), "emb-average"),
                   (Greedy(), "emb-greedy"),
                   (Extrema(), "emb-extrema")]

        for scorer, method in scorers:
            score, scores = self.compute_score(scorer, self.ground, self.predictions)
            self.eval[method] = score

    def compute_score(self, scorer, ground, predictions):
        scores = []
        for id, ground_ans in ground.items():
            predicted_ans = predictions[id]
            scores.append(scorer.calculate_score(self.v, ground_ans, predicted_ans))

        average_score = np.mean(np.array(scores))
        return average_score, np.array(scores)


class Average:
    @staticmethod
    def calculate_score(v, ground_answers, predicted_ans):
        assert type(predicted_ans) is list
        assert len(predicted_ans) == 1
        assert type(ground_answers) is list
        assert len(ground_answers) > 0

        ground_answers_means = []
        for ground_ans in ground_answers:
            ground_ans_idx = v.line_to_seq(ground_ans, output_nan=True)
            if not ground_ans_idx:
                continue
            ground_answers_means.append(v.W[np.array(ground_ans_idx)].mean(axis=0))
        assert ground_answers_means

        predicted_ans_idx = v.line_to_seq(predicted_ans[0], output_nan=True)
        if not predicted_ans_idx:
            score = 0
        else:
            predicted_ans_mean = v.W[np.array(predicted_ans_idx)].mean(axis=0)
            score = cosines(np.array(ground_answers_means), predicted_ans_mean).max()

        return score


class Extrema:
    @staticmethod
    def calculate_score(v, ground_answers, predicted_ans):
        assert type(predicted_ans) is list
        assert len(predicted_ans) == 1
        assert type(ground_answers) is list
        assert len(ground_answers) > 0

        ground_answers_extrema_vecs = []
        for ground_ans in ground_answers:
            ground_ans_idx = v.line_to_seq(ground_ans, output_nan=True)
            if not ground_ans_idx:
                continue
            ground_ans_vecs = v.W[np.array(ground_ans_idx)]
            extrema_arr = abs(ground_ans_vecs).argmax(axis=0)
            extrema_vec = ground_ans_vecs[extrema_arr, list(range(len(extrema_arr)))]
            ground_answers_extrema_vecs.append(extrema_vec)
        assert ground_answers_extrema_vecs

        predicted_ans_idx = v.line_to_seq(predicted_ans[0], output_nan=True)
        if not predicted_ans_idx:
            score = 0
        else:
            predicted_ans_vecs = v.W[np.array(predicted_ans_idx)]
            extrema_arr = abs(predicted_ans_vecs).argmax(axis=0)
            predicted_extrema_vec = predicted_ans_vecs[extrema_arr, list(range(len(extrema_arr)))]

            score = cosines(np.array(ground_answers_extrema_vecs), predicted_extrema_vec).max()

        return score


class Greedy:
    def calculate_score(self, v, ground_answers, predicted_ans):
        assert type(predicted_ans) is list
        assert len(predicted_ans) == 1
        assert type(ground_answers) is list
        assert len(ground_answers) > 0

        scores = []
        predicted_ans_idx = v.line_to_seq(predicted_ans[0], output_nan=True)
        for ground_ans in ground_answers:
            ground_ans_idx = v.line_to_seq(ground_ans, output_nan=True)
            if not ground_ans_idx:
                continue
            # direction 1
            if not predicted_ans_idx:
                avg_max_cosines_dir1 = 0
            else:
                max_cosines = []
                for id in ground_ans_idx:
                    max_cosines.append(self.get_max_cosine(v, id, predicted_ans_idx))
                avg_max_cosines_dir1 = np.mean(max_cosines)

            # direction 2
            max_cosines = []
            for id in predicted_ans_idx:
                max_cosines.append(self.get_max_cosine(v, id, ground_ans_idx))
            avg_max_cosines_dir2 = np.mean(max_cosines)

            scores.append(np.mean(np.array([avg_max_cosines_dir1, avg_max_cosines_dir2])))
        assert scores

        return np.max(scores)

    @staticmethod
    def get_max_cosine(v, id1, ids2):
        max_cosine = cosines(v.W[np.array(ids2)], v.W[id1]).max()
        return max_cosine

