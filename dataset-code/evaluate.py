""" Evaluation script for CliCR. Exact match and F1 are based on the evaluation script of the SQuAD dataset, v1.1. """
import argparse
from collections import Counter
import string
import re
import sys

import pexpect

from describe_data import *
from build_json_dataset import to_id_answertxt


def normalize_answer(s, lemmatizer_comm=None):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def lemma(text):
        if lemmatizer_comm is None:
            lemmatized = text
        else:
            lemmatized = lemmatize(text, lemmatizer_comm)
        return lemmatized

    return white_space_fix(remove_articles(remove_punc(lower(lemma(s)))))


def lemmatize(s, c):
    """
    Use luiNorm to lemmatize.

    :param c: spawned command (luiNorm)
    """
    def extract(s):
        """
        :param s: output string, e.g. '\r\ndrs|dr\r\n'
        """
        return s.split("|")[-1][:-len(eol)]
    if s.strip() == "" or s.strip() == "-":
        return s
    eol = "\r\n"
    c.sendline(s)
    c.expect(eol+".*"+eol)
    lemma = extract(c.after)

    return lemma


def lemmatizer(path="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/lvg2017/bin/luiNorm"):
    return pexpect.spawnu(path)


def f1_score(prediction, ground_truth, comm=None):
    prediction_tokens = normalize_answer(prediction, comm).split()
    ground_truth_tokens = normalize_answer(ground_truth, comm).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, comm=None):
    return normalize_answer(prediction, comm) == normalize_answer(ground_truth, comm)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, comm=None):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, comm)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, lemmatizer_path=None, extended=False, embeddings_file=None, downcase=False):
    data = dataset[DATA_KEY]
    comm = lemmatizer(lemmatizer_path) if lemmatizer_path else None
    f1 = exact_match = total = 0

    n_unanswered = 0
    datum_count = 0
    for datum in data:
        for qa in datum[DOC_KEY][QAS_KEY]:
            total += 1
            if qa[ID_KEY] not in predictions:
                n_unanswered += 1
                continue
            ground_truths = list(map(lambda x: x[TXT_KEY], qa[ANS_KEY]))
            prediction = predictions[qa[ID_KEY]]
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths, comm=comm)
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths, comm=comm)
        datum_count += 1
    print("There were {} unanswered instances".format(n_unanswered))
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    assert exact_match <= f1
    scores = {'exact_match': exact_match, 'f1': f1}

    if extended:
        from pycocoevalcap.eval import COCOEvalCap
        from embedding_eval import EmbeddingEval

        # COCO
        ground = {}
        for id, ans in to_id_answertxt(dataset).items():
            normalized_ans = []
            for a in ans:
                normalized_ans.append(normalize_answer(a, comm))
            ground[id] = normalized_ans
        _predictions = {id: [normalize_answer(ans, comm)] for id, ans in predictions.items()}
        cocoEval = COCOEvalCap(ground, _predictions)
        cocoEval.evaluate()

        # Embeddings evaluation
        embEval = EmbeddingEval(ground, _predictions, embeddings_file, downcase)
        embEval.evaluate()

        #scores = {**scores, **cocoEval.eval, **embEval.eval}  # only python3.5
        scores.update(cocoEval.eval)
        scores.update(embEval.eval)

    return scores


def print_scores(scores):
    """
    :param scores: {"method1": score, ...}
    """
    print("{}\t{:.1f}".format("exact_match", scores["exact_match"]))
    print("{}\t{:.1f}".format("f1", scores["f1"]))

    for method, score in sorted(scores.items()):
        if method == "exact_match" or method == "f1":
            continue
        else:
            print("{}\t{:.3f}".format(method, score))


if __name__ == '__main__':
    expected_version = '1.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for CliCR ' + expected_version)
    parser.add_argument('-test_file', help='Dataset file with ground truth.')
    parser.add_argument('-prediction_file', help='Prediction file of the format {"id1234": "Answer", ...}')
    parser.add_argument("-lemmatizer_path", help="Use luiNorm if path provided.")
    parser.add_argument("-extended", help="Whether to use extended evaluation using Bleu, Rouge and embeddings.",
                        action="store_true")
    parser.add_argument('-embeddings_file', help='Embeddings in w2v txt format. Only needed for extended evaluation.')
    parser.add_argument('-downcase',
                        help="Only for extended evaluation. Should be set to true if the embedding vocabulary is lowercased.",
                        action="store_true")
    args = parser.parse_args()
    if args.extended:
        assert args.embeddings_file is not None

    dataset = load_json(args.test_file)

    predictions = load_json(args.prediction_file)
    print_scores(evaluate(dataset, predictions, lemmatizer_path=args.lemmatizer_path, extended=args.extended,
                          embeddings_file=args.embeddings_file, downcase=args.downcase))
