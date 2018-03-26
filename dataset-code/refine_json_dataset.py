import argparse

from build_json_dataset import document_instance, datum_instance, dataset_instance
from describe_data import *
from util import load_json, save_json


def remove_exact_nwords(dataset, max_n_words):
    """
    remove those queries where either of left/right side of the query occurs in its exact form in the doc.
    limit the left/right side length to max_n_words to remove more aggressively.
    """
    n_matched = 0
    n_all = 0
    new_data = []
    data = dataset[DATA_KEY]
    for datum in data:
        doc = datum[DOC_KEY]
        txt = doc[CONTEXT_KEY]
        qas = []
        for qa in doc[QAS_KEY]:
            q = qa[QUERY_KEY]
            ans = ""
            for a in qa[ANS_KEY]:
                if a[ORIG_KEY] == "dataset":
                    ans = a[TXT_KEY]
            assert ans
            q_line = ""
            for line in q.split("\n"):
                if PLACEHOLDER_KEY in line:
                    q_line = line
            assert q_line
            idx_start = q.find(PLACEHOLDER_KEY)
            idx_end = idx_start + len(PLACEHOLDER_KEY)
            if len(q_line) > idx_end:
                if q_line[idx_end] != " ":
                    q_line = q_line[:idx_end] + " " + q_line[idx_end:]
            txt_left = q_line[:idx_start].rstrip()
            txt_left_shortened = " ".join(txt_left.split()[-max_n_words:])
            txt_right = q_line[idx_end:].lstrip()
            txt_right_shortened = " ".join(txt_right.split()[:max_n_words])
            #if (len(txt_left_shortened) == max_n_words and txt_left_shortened in txt) or \
            #        (len(txt_right_shortened) == max_n_words and txt_right_shortened in txt):
            #    n_matched += 1
            if len(txt_left_shortened.split()) == max_n_words and txt_left_shortened in txt:
                n_matched += 1
            elif len(txt_right_shortened.split()) == max_n_words and txt_right_shortened in txt:
                n_matched += 1
            else:
                qas.append(qa)
            n_all += 1
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    print("Number of matched queries: {}".format(n_matched))
    print("Number of all queries: {}".format(n_all))
    print("Percentage matched: {}".format((n_matched / n_all) * 100))
    return dataset_instance(dataset[VERSION_KEY], new_data)


def remove_exact_longermatch(dataset):
    """
    remove those queries where the longer of left/right side of the query occurs in its exact form in the doc
    """
    n_matched = 0
    n_all = 0
    new_data = []
    data = dataset[DATA_KEY]
    for datum in data:
        doc = datum[DOC_KEY]
        txt = doc[CONTEXT_KEY]
        qas = []
        for qa in doc[QAS_KEY]:
            q = qa[QUERY_KEY]
            ans = ""
            for a in qa[ANS_KEY]:
                if a[ORIG_KEY] == "dataset":
                    ans = a[TXT_KEY]
            assert ans
            q_line = ""
            for line in q.split("\n"):
                if PLACEHOLDER_KEY in line:
                    q_line = line
            assert q_line
            idx_start = q.find(PLACEHOLDER_KEY)
            idx_end = idx_start + len(PLACEHOLDER_KEY)
            if len(q_line) > idx_end:
                if q_line[idx_end] != " ":
                    q_line = q_line[:idx_end] + " " + q_line[idx_end:]
            txt_left = q_line[:idx_start].rstrip()
            txt_right = q_line[idx_end:].lstrip()
            #sent_query_with_ans = txt_left + ans + txt_right
            # get the longer of txt_left or txt_right
            longer_context = txt_right if len(txt_right) > len(txt_left) else txt_left

            #if sent_query_with_ans in txt:
            #    pass
            if longer_context in txt:
                n_matched += 1
            else:
                qas.append(qa)
            n_all += 1
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    print("Number of matched queries: {}".format(n_matched))
    print("Number of all queries: {}".format(n_all))
    print("Percentage matched: {}".format((n_matched / n_all) * 100))
    return dataset_instance(dataset[VERSION_KEY], new_data)


if __name__ == "__main__":
    """
    cd ~/Apps/bmj_case_reports
    DATA=/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-json_dataset", help="Path to train/dev/test json dataset.", required=True)
    args = parser.parse_args()

    dataset = load_json(args.json_dataset)

    new_dataset = remove_exact_longermatch(dataset)
    print("overwriting the old dataset")
    save_json(new_dataset, args.json_dataset)
    #save_json(new_dataset, "{}.without_longermatch.json".format(args.json_dataset.rsplit(".", 1)[0]))
    #new_dataset = remove_exact_nwords(dataset, max_n_words=2)
    #save_json(new_dataset, "{}.without_nwordmatch{}.json".format(args.json_dataset.rsplit(".", 1)[0], 2))
    #new_dataset = remove_exact_nwords(dataset, max_n_words=3)
    #save_json(new_dataset, "{}.without_nwordmatch{}.json".format(args.json_dataset.rsplit(".", 1)[0], 3))
    #new_dataset = remove_exact_nwords(dataset, max_n_words=4)
    #save_json(new_dataset, "{}.without_nwordmatch{}.json".format(args.json_dataset.rsplit(".", 1)[0], 4))
    #new_dataset = remove_exact_nwords(dataset, max_n_words=5)
    #save_json(new_dataset, "{}.without_nwordmatch{}.json".format(args.json_dataset.rsplit(".", 1)[0], 5))
