import argparse
import os
from os.path import basename

from build_queries import build_queries
from describe_data import *
from expand_answers import expand, conn
from util import get_file_list, load_json, save_json


def get_answers(q, umls_cur=None):
    def answer_instance(text, cui, sem_type, origin):
        return {"text": text, "cui": cui, "sem_type": sem_type, "origin": origin}

    answers = []
    a = q[0]
    cui = q[2]
    sem_type = q[1]
    answers.append(answer_instance(a, cui, sem_type, "dataset"))

    expanded_set = expand(cui, umls_cur)
    if a in expanded_set:
        expanded_set.remove(a)  # to separate original answer from the expanded set; both written to output

    for a in expanded_set:
        answers.append(answer_instance(a, cui, sem_type, "UMLS"))

    return answers


def get_source(fn_case):
    ext_id = fn_case.find(".full.struct.tok")

    return os.path.basename(fn_case[:ext_id])


def get_title_and_context(txt_case):
    try:
        title, context = txt_case.split("\n", maxsplit=1)
    except ValueError:
        print("Can't split into TITLE and CONTEXT. Check.")
        title, context = ""

    return title.strip(), context.strip()


def build_dataset(mark_concepts=False, mark_query_concepts=False):
    def consistent(txt):
        is_consistent = True
        inside = False
        for w in txt.split():
            if marker1 in w and marker2 in w:
                inside = False
                if not w.startswith(marker1) or not w.endswith(marker2):
                    is_consistent = False
                continue
            elif marker1 in w:
                if inside or not w.startswith(marker1):
                    is_consistent = False
                if w.startswith(marker1) and not w[len(marker1):]:
                    is_consistent = False
                inside = True
            elif marker2 in w:
                if not inside or not w.endswith(marker2):
                    is_consistent = False
                if w.endswith(marker2) and not w[:-len(marker2)]:
                    is_consistent = False
                inside = False
        return is_consistent

    def qa_instance(query, id, answers):
        return {"query": query, "id": id, "answers": answers}

    data = []
    umls_cur = conn().cursor()

    if mark_concepts:
        marker1 = "BEG__"
        marker2 = "__END"
    else:
        marker1 = ""
        marker2 = ""

    for n_case, fn_case in enumerate(get_file_list(args.dir_cases)):
        if n_case % 1000 == 0:
            print("Number of cases processed: {}".format(n_case))
        fn_proc = args.dir_cases_concepts + basename(fn_case) + ".txt"
        if not os.path.isfile(fn_proc):
            print("Does not exist:" + fn_proc)
            continue
        source = get_source(fn_case)
        queries, txt_case = build_queries(fn_case, fn_proc, marker1=marker1, marker2=marker2,
                                          mark_query_concepts=mark_query_concepts)

        if not consistent(txt_case):
            print("Annotated passage not consistent, skipping.")
            continue
        title, context = get_title_and_context(txt_case)
        qas = []
        for counter, q in enumerate(queries, 1):
            id = "{}.{}".format(source, counter)
            answers = get_answers(q, umls_cur=umls_cur)
            query = q[3]
            if not consistent(query):
                print("Annotated query not consistent, skipping")
                continue
            qas.append(qa_instance(query, id, answers))

        document = document_instance(context, title, qas)
        data.append(datum_instance(document, source))

    return dataset_instance(version, data)


def document_instance(context, title, qas):
    return {"context": context, "title": title, "qas": qas}


def dataset_instance(version, data):
    return {"version": version, "data": data}


def datum_instance(document, source):
        return {"document": document, "source": source}


def intersect_on_ids(dataset, predictions):
    """
    Reduce data to include only those qa ids which occur in predictions.
    """
    new_data = []

    for datum in dataset[DATA_KEY]:
        qas = []
        for qa in datum[DOC_KEY][QAS_KEY]:
            if qa[ID_KEY] in predictions:
                qas.append(qa)
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    return dataset_instance(dataset[VERSION_KEY], new_data)


def sample_dataset(f, f_out,  n=50):
    """
    Reduce the dataset to include only the first n instances from n different case reports.
    """
    dataset = load_json(f)
    new_data = []

    for c, datum in enumerate(dataset[DATA_KEY]):
        if c == n:
            break
        qas = [datum[DOC_KEY][QAS_KEY][0]]
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    save_json(dataset_instance(dataset[VERSION_KEY], new_data), f_out)

    
    
def intersect_datasets_on_ids(dataset1, dataset2):
    """
    Reduce dataset1 to include only those qa ids which occur in dataset2.
    This is useful eg to reduce a marked dataset based on a dataset with applied
    exact-match filters from refine_json_dataset.py
    """
    data1 = load_json(dataset1)
    data2 = load_json(dataset2)
    new_data = []

    # obtain ids from dataset2
    data2_ids = set()
    for datum in data2[DATA_KEY]:
        for qa in datum[DOC_KEY][QAS_KEY]:
            data2_ids.add(qa[ID_KEY])

    # reduce data1 based on ids from data2
    for datum in data1[DATA_KEY]:
        qas = []
        for qa in datum[DOC_KEY][QAS_KEY]:
            if qa[ID_KEY] in data2_ids:
                qas.append(qa)
            else:
                print("reduction")
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    return dataset_instance(data1[VERSION_KEY], new_data)


def split_test(train_file, test_file):
    """
    Split the test set based on whether the answer entity was observed in the training data or not.
    """
    # Get the set of answers in the training set
    stats_tr = GeneralStats(train_file)
    ans_tr = set(stats_tr.most_frequent_answers(origin="dataset").keys())

    data = load_json(test_file)
    new_data_seen, new_data_unseen = [], []
    size_seen, size_unseen = 0, 0

    # reduce data based on answers in answers_seen
    for datum in data[DATA_KEY]:
        qas_seen = []
        qas_unseen = []
        for qa in datum[DOC_KEY][QAS_KEY]:
            ans = ""
            for a in qa[ANS_KEY]:
                if a[ORIG_KEY] == "dataset":
                    ans = a[TXT_KEY]
            assert ans
            if ans in ans_tr:
                qas_seen.append(qa)
            else:
                qas_unseen.append(qa)
        assert qas_seen + qas_unseen
        if qas_seen:
            size_seen += len(qas_seen)
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas_seen)
            new_data_seen.append(datum_instance(new_doc, datum[SOURCE_KEY]))
        if qas_unseen:
            size_unseen += len(qas_unseen)
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas_unseen)
            new_data_unseen.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    print("Size of the seen test dataset: {}".format(size_seen))
    print("Size of the unseen test dataset: {}".format(size_unseen))
    dataset_seen = dataset_instance(data[VERSION_KEY], new_data_seen)
    dataset_unseen = dataset_instance(data[VERSION_KEY], new_data_unseen)

    return dataset_seen, dataset_unseen


def is_intersect_same(dataset1, dataset2):
    """
    Check
    a) whether dataset1 and dataset2 include exactly the same qa ids
    b) set of ids in dataset1 that does not occur in dataset2
    c) set of ids in dataset2 that does not occur in dataset1
    """
    data1 = load_json(dataset1)
    data2 = load_json(dataset2)
    new_data = []

    # obtain ids from dataset1
    data1_ids = set()
    for datum in data1[DATA_KEY]:
        for qa in datum[DOC_KEY][QAS_KEY]:
            data1_ids.add(qa[ID_KEY])

    # obtain ids from dataset2
    data2_ids = set()
    for datum in data2[DATA_KEY]:
        for qa in datum[DOC_KEY][QAS_KEY]:
            data2_ids.add(qa[ID_KEY])

    return data1_ids == data2_ids, data1_ids - data2_ids, data2_ids - data1_ids


def to_id_answertxt(dataset):
    """
    Convert the ground-truth dataset to the concise form as used for predictions, ie
    including only instance id and the answer text.

    :return: {"id1234": ["Answer1", "Answer2, ...], ...}
    """
    new_data = {}

    for datum in dataset[DATA_KEY]:

        for qa in datum[DOC_KEY][QAS_KEY]:
            answers = []
            for a in qa[ANS_KEY]:
                answers.append(a[TXT_KEY])
            new_data[qa[ID_KEY]] = answers

    return new_data


if __name__ == "__main__":
    """
    cd ~/Apps/bmj_case_reports
    DATA=/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data

    python3 build_json_dataset.py -dir_cases $DATA/data_out_tok/ \
    -dir_cases_concepts $DATA/data_proc/clamp/ -file_output $DATA/dataset
    """
    version = "1.0"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dir_cases", help="Path to directory containing preprocessed case files.", required=True)
    parser.add_argument("-dir_cases_concepts",
                        help="Path to directory containing case files with concepts extracted by Clamp.", required=True)
    parser.add_argument("-file_output")
    parser.add_argument("-mark_concepts", help="Whether to mark concepts in document passages as annotated by Clamp",
                        action="store_true")
    parser.add_argument("-mark_query_concepts", help="Whether to mark concepts in queries as annotated by Clamp",
                        action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(args.mark_concepts, args.mark_query_concepts)
    filename = args.file_output + version + ".json"
    save_json(dataset, filename)

