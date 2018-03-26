import argparse
import os
import subprocess
import uuid

from baselines import distance_baseline, vectorize_contexts_of_concepts
from evaluate import evaluate
from randomized_hyperparameter_search import RandomizedSearch, embedding_parameter_space
from util import load_json, save_json


def train_embeddings(params):
    def get_train_file():
        # pubmed + clicr
        filename = "/nas/corpora/accumulate/pubmed_pmc/processed/tokens/all_clicrtrain.lower"
        #filename = "/nas/corpora/accumulate/clicr/wiki_nyt_lower_shuf"

        return filename

    def get_embs_file():
        dirname = create_unique_dir("/nas/corpora/accumulate/clicr/embeddings/")
        filename = dirname + "embeddings"
        return dirname, filename

    def write_info(cmd):
        with open(embs_dir + "info", "w") as outfile:
            outfile.write("{}".format(cmd))

    train_file = get_train_file()
    embs_dir, embs_file = get_embs_file()
    dimension = params["dimension"]
    win_size = params["win_size"]
    min_freq = params["min_freq"]
    neg_samples = params["neg_samples"]
    model = params["model"]

    cmd = "~/Apps/word2vec/bin/word2vec -train {} -output {} -size {dimension} -window {win_size} -negative \
    {neg_samples} -hs 0 -binary 0 -cbow {model} -min-count {min_count} -threads {threads}".format(
        train_file, embs_file, dimension=dimension, win_size=win_size, neg_samples=neg_samples,
        model=1 if model == "cbow" else 0, min_count=min_freq, threads=11)
    print(cmd)
    cmd_open = subprocess.call(cmd, shell=True)
    write_info(cmd)

    return embs_file


def train_and_eval_embeddings(test_file, params, downcase, lemmatizer_path=None, extended=False):
    embeddings_file = train_embeddings(params)

    dataset = load_json(test_file)
    predictions_distance_concepts = distance_baseline(dataset, embeddings_file, downcase,
                                                      vectorize_contexts_of_concepts)
    scores = evaluate(dataset, predictions_distance_concepts, lemmatizer_path=lemmatizer_path, extended=extended)

    return scores


def get_best_params(all_scores):
    best_scores = {}
    best_runs = {}
    for eval_method in all_scores[0].keys():
        best_scores[eval_method] = 0

    for n_run, results in all_scores.items():
        for eval_method, score in results.items():
            if results[eval_method] > best_scores[eval_method]:
                best_scores[eval_method] = results[eval_method]
                best_runs[eval_method] = n_run

    return best_runs, best_scores


def create_unique_dir(basedir):
    dirname = basedir+"{}/".format(uuid.uuid1())
    os.makedirs(dirname)

    return dirname


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply some simple baselines.')
    parser.add_argument('-test_file',
                        help='Json test file with concept annotations.',
                        default='/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json')
    parser.add_argument('-n_times', help='Number of times to train the embbedings (for randomized search).', type=int,
                        default=10)
    parser.add_argument("-lemmatizer_path", help="Evaluation: Use luiNorm if path provided.")
    parser.add_argument("-extended", help="Evaluation: Whether to use extended evaluation using Bleu and Rouge.",
                        action="store_true")
    args = parser.parse_args()

    downcase = True  # fixed

    all_params = {}
    all_scores = {}
    random_search = RandomizedSearch(embedding_parameter_space)
    for n in range(args.n_times):
        print("Run {}...".format(n))
        params = random_search.sample()
        scores = train_and_eval_embeddings(test_file=args.test_file,
                                           params=params,
                                           downcase=downcase,
                                           extended=args.extended)
        all_params[n] = params
        all_scores[n] = scores
        print(params)
        print(scores)
    best_runs, best_scores = get_best_params(all_scores)

    out_dir = create_unique_dir(
        "/nas/corpora/accumulate/clicr/randomized_search_runs/")
    save_json(best_runs, out_dir + "best_runs")
    save_json(best_scores, out_dir + "best_scores")
    save_json(all_params, out_dir + "all_params")
    save_json(all_scores, out_dir + "all_scores")

    for eval_method, n_run in best_runs.items():
        print("{}: {}={}".format(eval_method, n_run, all_params[n_run]))

