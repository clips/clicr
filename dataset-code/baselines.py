import argparse
import os
from collections import Counter

import numpy as np

from describe_data import *
from evaluate import evaluate, print_scores, f1_score
from text import VocabBuild
from util import load_json, save_json, random_instance_from_list, cosines


def random_word_baseline(dataset):
    def random_word(text):
        text_split = text.split()

        return random_instance_from_list(text_split)

    data = dataset[DATA_KEY]
    predictions = {}
    for datum in data:
        title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        for qa in datum[DOC_KEY][QAS_KEY]:
            id = qa[ID_KEY]
            # query = qa[QUERY_KEY]
            predictions[id] = random_word(remove_concept_marks(title_and_passage))

    return predictions


def read_concepts(text):
    concept_list = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = w_stripped.split("_")[2]
            concept_list.append([concept])
        elif w_stripped.startswith("BEG__"):
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]
        elif w_stripped.endswith("__END"):
            concept.append(w_stripped.rsplit("_", 2)[0])
            concept_list.append(concept)
            inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                continue
    return concept_list


def random_concept_baseline(dataset):
    def random_concept(text):
        """
        :param text: contains concepts annotated as 'w_i BEG__w_k w_j__END w_l',
        where 'w_k w_k' is a concept.
        :return: a concept
        """
        concept_list = read_concepts(text)

        return " ".join(random_instance_from_list(concept_list))

    data = dataset[DATA_KEY]
    predictions = {}
    for datum in data:
        title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        for qa in datum[DOC_KEY][QAS_KEY]:
            id = qa[ID_KEY]
            # query = qa[QUERY_KEY]
            predictions[id] = random_concept(title_and_passage)

    return predictions


def max_score_ood(metric_fn, dataset):
    """
    Calculate predictions obtained with the best eval score on the part of the dataset where answers (answer concepts) don't occur
    among the document concepts. While exact match would give a score of 0, other metric can account
    for partially correct answers.
    :param metric_fn: any of f1, bleu, embs etc. See evaluate.py
    :return: predictions
    """
    def best_concept(metric_fn, text, answer):
        """
        :param text: contains concepts annotated as 'w_i BEG__w_k w_j__END w_l',
        where 'w_k w_k' is a concept.
        """
        concept_set = {" ".join(concept).lower() for concept in read_concepts(text)}
        assert len(concept_set) > 0

        # skip if answer is found in document
        if answer.lower() in concept_set:
            return None

        max_score = .0
        max_concept = None
        for concept in concept_set:
            score = metric_fn(concept, answer)
            if score > max_score:
                max_concept = concept

        if max_concept is None:
            max_concept = concept_set.pop()
        return max_concept

    data = dataset[DATA_KEY]
    predictions = {}
    n_in_doc = 0
    for datum in data:
        title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        for qa in datum[DOC_KEY][QAS_KEY]:
            id = qa[ID_KEY]
            ans = qa[ANS_KEY]
            a = ""
            for _a in ans:
                if _a[ORIG_KEY] == "dataset":
                    a = _a[TXT_KEY]
            assert a

            max_concept = best_concept(metric_fn, title_and_passage, a)
            if max_concept is not None:
                predictions[id] = max_concept
            else:  # concept occurs in the document, skip
                n_in_doc += 1
                continue
    print("Out-of-document answers: {}/{}".format(len(predictions), len(predictions)+n_in_doc))
    return predictions


def maxfreq_concept_baseline(dataset):
    def maxfreq_concept(text):
        """
        :param text: contains concepts annotated as 'w_i BEG__w_k w_j__END w_l',
        where 'w_k w_k' is a concept.
        :return: a concept
        """
        concept_counts = count_concepts(text)

        return concept_counts.most_common(1).pop()[0]

    def count_concepts(text):
        concept_counter = Counter()
        inside = False
        for w in text.split():
            w_stripped = w.strip()
            if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
                concept = w_stripped.split("_")[2]
                concept_counter[concept] += 1
            elif w_stripped.startswith("BEG__"):
                inside = True
                concept = [w_stripped.split("_", 2)[-1]]
            elif w_stripped.endswith("__END"):
                concept.append(w_stripped.rsplit("_", 2)[0])
                concept_counter[" ".join(concept)] += 1
                inside = False
            else:
                if inside:
                    concept.append(w_stripped)
                else:
                    continue

        return concept_counter

    data = dataset[DATA_KEY]
    predictions = {}
    for datum in data:
        title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        for qa in datum[DOC_KEY][QAS_KEY]:
            id = qa[ID_KEY]
            # query = qa[QUERY_KEY]
            predictions[id] = maxfreq_concept(title_and_passage)

    return predictions


def vectorize_contexts_of_concepts(text, v, win_size=3):
    targets = []
    T = []  # n_words*(2*win_size)
    for line in text.split("\n"):
        idxs_start = [match.start() for match in re.finditer("BEG__", line)]
        idxs_end = [match.end() for match in re.finditer("__END", line)]
        for i_start, i_end in zip(idxs_start, idxs_end):
            concept = line[i_start + len("BEG__"):i_end - len("__END")]
            txt_left = line[:i_start].strip()
            lst_left = txt_left.split()
            txt_right = line[i_end:].strip()
            lst_right = txt_right.split()
            lst = lst_left + lst_right
            i = len(lst_left)
            window_start = max(0, i - win_size)
            window_end = min(len(lst), i + win_size)
            contexts = []
            # go over contexts
            for j in range(window_start, window_end):
                w = lst[j]
                w = remove_concept_marks(w)
                c_idx = v.lookup(w, output_nan=True)
                contexts.append(c_idx)
            for _ in range(2 * win_size - len(contexts)):  # padding for start/end sent
                contexts.append(0)  # special out of seq idx
            assert len(contexts) == 2 * win_size
            targets.append(concept)
            T.append(contexts)

    assert T
    T_w_summed = v.W[np.array(T)].sum(axis=1)  # n_words*d

    assert len(targets) > 0
    assert len(targets) == T_w_summed.shape[0]

    return targets, T_w_summed


def vectorize_contexts_of_words(text, v, win_size=3):
    targets = []
    T = []  # n_words*(2*win_size)
    for line in remove_concept_marks(text).split("\n"):
        seq = v.line_to_seq(line, output_nan=True)
        if len(seq) < 2:
            continue
        for i, w_idx in enumerate(seq):
            window_start = max(0, i - win_size)
            window_end = min(len(seq), i + win_size + 1)
            contexts = []
            # go over contexts
            for j in range(window_start, window_end):
                if j != i:
                    c_idx = seq[j]
                    contexts.append(c_idx)
            for _ in range(2 * win_size - len(contexts)):  # padding for start/end sent
                contexts.append(0)  # special out of seq idx
            targets.append(w_idx)
            T.append(contexts)

    T_w_summed = v.W[np.array(T)].sum(axis=1)  # n_words*d
    assert len(targets) == T_w_summed.shape[0]

    return targets, T_w_summed


def vectorize_query(q, v, win_size=3):
    q_line = ""
    for line in q.split("\n"):
        if PLACEHOLDER_KEY in line:
            q_line = line
    assert q_line
    q_line_lst = remove_concept_marks(q_line).split()
    q_line = " ".join(q_line_lst)
    idx_start = q_line.find(PLACEHOLDER_KEY)
    idx_end = idx_start + len(PLACEHOLDER_KEY)
    if len(q_line) > idx_end:
        if q_line[idx_end] != " ":
            q_line = q_line[:idx_end] + " " + q_line[idx_end:]
    txt_left = q_line[:idx_start].rstrip()
    seq_left = v.line_to_seq(txt_left, output_nan=True) if txt_left else []
    txt_right = q_line[idx_end:].lstrip()
    seq_right = v.line_to_seq(txt_right, output_nan=True) if txt_right else []
    seq = seq_left + seq_right
    assert len(seq) == len(q_line.strip().split()) - 1, q_line  # removed placeholder

    i = len(seq_left)
    window_start = max(0, i - win_size)
    window_end = min(len(seq), i + win_size)
    contexts = []
    # go over contexts
    for j in range(window_start, window_end):
        c_idx = seq[j]
        contexts.append(c_idx)
    for _ in range(2 * win_size - len(contexts)):  # padding for start/end sent
        contexts.append(0)  # special out of seq idx
    assert len(contexts) == 2 * win_size
    q_w_summed = v.W[np.array(contexts)].sum(axis=0)  # d*1

    return q_w_summed


def distance_baseline(dataset, embeddings_file, downcase, context_vectorize_fun, win_size):
    v = VocabBuild(embeddings_file, downcase=downcase)
    v.read()
    data = dataset[DATA_KEY]
    predictions = {}
    for datum in data:
        title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        targets, C = context_vectorize_fun(title_and_passage, v, win_size=win_size)  # n_words*d
        for qa in datum[DOC_KEY][QAS_KEY]:
            id = qa[ID_KEY]
            query = qa[QUERY_KEY]
            query_repr = vectorize_query(query, v)
            idx = best_answer(C, query_repr)
            if context_vectorize_fun == vectorize_contexts_of_concepts:
                predictions[id] = targets[idx]
            else:
                predictions[id] = v.inv_w_index[targets[idx]]

    return predictions


def best_answer(context_matrix, query_vector):
    return cosines(context_matrix, query_vector).argmax()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply some simple baselines.')
    parser.add_argument('-test_file',
                        help='Json test file with concept annotations.',
                        default='/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json')
    parser.add_argument('-embeddings_file', help='Embeddings in w2v txt format.')
    parser.add_argument('-downcase',
                        help="Only for distance baselines. Should be set to true if the embedding vocabulary is lowercased.",
                        action="store_true")
    parser.add_argument('-eval_embeddings_file', help='Embeddings in w2v txt format for embedding-based evaluation. Can be the same as the ones used in the baseline.')
    parser.add_argument('-eval_downcase',
                        help="Should be set to true if the embedding vocabulary for embedding-based evaluation is lowercased.",
                        action="store_true")
    parser.add_argument('-predictions_dir', help='Directory to write predictions to.')
    parser.add_argument("-lemmatizer_path", help="Evaluation: Use luiNorm if path provided.")
    parser.add_argument("-win_size", help="Window size to each side for the embedding baselines.", default=3, type=int)
    parser.add_argument("-extended",
                        help="Evaluation: Whether to use extended evaluation using Bleu, Rouge and embeddings.",
                        action="store_true")

    args = parser.parse_args()
    save_json(vars(args), "{}/args.json".format(args.predictions_dir))

    print(args.test_file)
    print(args.embeddings_file)

    print("Obtaining baseline predictions...")
    dataset = load_json(args.test_file)
    #predictions_max_ood = max_score_ood(f1_score, dataset_marked)
    #scores_maxood_concepts = evaluate(intersect_on_ids(dataset_marked, predictions_max_ood), predictions_max_ood,
    #                                   lemmatizer_path=args.lemmatizer_path,
    #                                   extended=args.extended, embeddings_file=args.eval_embeddings_file,
    #                                   downcase=args.eval_downcase)
    #print(scores_maxood_concepts)

    predictions_words = random_word_baseline(dataset)
    print("rand-word OK")
    predictions_concepts = random_concept_baseline(dataset)
    print("rand-concept OK")
    predictions_maxfreq_concepts = maxfreq_concept_baseline(dataset)
    print("maxfreq-concept OK")
    predictions_distance_words = distance_baseline(dataset, args.embeddings_file, args.downcase,
                                                   vectorize_contexts_of_words, win_size=args.win_size)
    print("sim-word OK")

    predictions_distance_concepts = distance_baseline(dataset, args.embeddings_file, args.downcase,
                                                      vectorize_contexts_of_concepts, win_size=args.win_size)
    print("sim-entity OK")

    print("Evaluating baseline predictions...")
    scores_words = evaluate(dataset, predictions_words, lemmatizer_path=args.lemmatizer_path, extended=args.extended,
                            embeddings_file=args.eval_embeddings_file, downcase=args.eval_downcase)
    print("\nrand-word:")
    print_scores(scores_words)
    scores_concepts = evaluate(dataset, predictions_concepts, lemmatizer_path=args.lemmatizer_path,
                               extended=args.extended, embeddings_file=args.eval_embeddings_file,
                               downcase=args.eval_downcase)
    print("\nrand-concept:")
    print_scores(scores_concepts)
    scores_maxfreq_concepts = evaluate(dataset, predictions_maxfreq_concepts, lemmatizer_path=args.lemmatizer_path,
                                       extended=args.extended, embeddings_file=args.eval_embeddings_file,
                                       downcase=args.eval_downcase)
    print("\nmaxfreq-concept:")
    print_scores(scores_maxfreq_concepts)
    scores_distance_words = evaluate(dataset, predictions_distance_words, lemmatizer_path=args.lemmatizer_path,
                                     extended=args.extended, embeddings_file=args.eval_embeddings_file,
                                     downcase=args.eval_downcase)
    print("\nsim-word:")
    print_scores(scores_distance_words)

    scores_distance_concepts = evaluate(dataset, predictions_distance_concepts,
                                        lemmatizer_path=args.lemmatizer_path, embeddings_file=args.eval_embeddings_file,
                                        downcase=args.eval_downcase, extended=args.extended)

    print("\nsim-entity:")
    print_scores(scores_distance_concepts)

    print("\nSaving predictions and scores files...")
    if args.predictions_dir is not None:
        save_json(predictions_words, "{}/preds_words_{}".format(args.predictions_dir, os.path.basename(args.test_file)))
        save_json(scores_words,
                  "{}/preds_words_{}.scores".format(args.predictions_dir, os.path.basename(args.test_file)))

        save_json(predictions_concepts, "{}/preds_concepts_{}".format(args.predictions_dir,
                                                                      os.path.basename(args.test_file)))
        save_json(scores_concepts, "{}/preds_concepts_{}.scores".format(args.predictions_dir,
                                                                        os.path.basename(args.test_file)))
        save_json(scores_maxfreq_concepts, "{}/preds_maxfreq_concepts_{}.scores".format(args.predictions_dir,
                                                                        os.path.basename(args.test_file)))
        save_json(predictions_distance_words, "{}/preds_distance_{}".format(args.predictions_dir,
                                                                            os.path.basename(args.test_file)))
        save_json(scores_distance_words, "{}/preds_distance_{}.scores".format(args.predictions_dir,
                                                                              os.path.basename(args.test_file)))
        save_json(predictions_distance_concepts, "{}/preds_distance_concepts{}".format(args.predictions_dir,
                                                                                       os.path.basename(
                                                                                           args.test_file)))
        save_json(scores_distance_concepts, "{}/preds_distance_concepts{}.scores".format(args.predictions_dir,
                                                                                         os.path.basename(
                                                                                             args.test_file)))

