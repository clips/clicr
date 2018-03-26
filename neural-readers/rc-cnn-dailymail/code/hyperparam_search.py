import argparse
import subprocess

from randomized_hyperparam_search import RandomizedSearch, stanford_reader_parameter_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_times', help='Number of times to choose random parameters and train the model with them.',
                        type=int, default=10)
    parser.add_argument('-log_file', help='General log name. To this we append a run descriptor.',
                        default="~/Apps/rc-cnn-dailymail/experiments/hypersearch")
    search_args = parser.parse_args()

    random_search = RandomizedSearch(stanford_reader_parameter_space)
    for n in range(search_args.n_times):
        print("Run {}...".format(n))
        params = random_search.sample()
        print(params)
        command = " ".join(["THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python", "main.py"] +
            ["--train_file /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/train1.0.json --dev_file /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json --embedding_file /nas/corpora/accumulate/clicr/embeddings/de004a58-6eef-11e7-ac2f-901b0e5592c8/embeddings"] + \
            ["--{} {}".format(k, v) for k, v in params.items()] + \
            ["--log_file {}{}.log".format(
                search_args.log_file,
                "_".join(map(lambda k: "{}{}".format(k, params[k]), list(params)))
                )])
        subprocess.call(command, shell=True)
