import json
import os
import subprocess

DATA_KEY = "data"
VERSION_KEY = "version"
DOC_KEY = "document"
QAS_KEY = "qas"
ANS_KEY = "answers"
TXT_KEY = "text"  # the text part of the answer
ORIG_KEY = "origin"
ID_KEY = "id"
TITLE_KEY = "title"
CONTEXT_KEY = "context"
SOURCE_KEY = "source"
QUERY_KEY = "query"
CUI_KEY = "cui"
SEMTYPE_KEY = "sem_type"

PLACEHOLDER_KEY = "@placeholder"


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)


def to_entities(text):
    """
    Text includes entities marked as BEG__w1 w2 w3__END. Transform to a single entity @entityw1_w2_w3.
    """
    word_list = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = [w_stripped.split("_")[2]]
            word_list.append("@entity" + "_".join(concept))
            if inside:  # something went wrong, leave as is
                print("Inconsistent markup.")
        elif w_stripped.startswith("BEG__"):
            assert not inside
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]
        elif w_stripped.endswith("__END"):
            if not inside:
                return None
            assert inside
            concept.append(w_stripped.rsplit("_", 2)[0])
            word_list.append("@entity" + "_".join(concept))
            inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                word_list.append(w_stripped)

    return " ".join(word_list)


def write_preds(preds, file_name):
    """
    :param preds: {q_id: answer, ...}

    Write predictions as a json file.
    """
    save_json(preds, file_name)


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))


def to_output_preds(preds):
    """
    """

    def prepare_answer(txt):
        if txt.startswith("@entity"):
            return txt[len("@entity"):].replace("_", " ")
        else:
            return txt

    return {q_id: prepare_answer(answer) for q_id, answer in preds.items()}


def external_eval(preds_file, file_name, eval_dataset, extended=False):
    print("External evaluation, penalizing unanswered...")
    cmd = "python3 ~/Apps/bmj_case_reports/evaluate.py -test_file {} -prediction_file {} -embeddings_file /nas/corpora/accumulate/clicr/embeddings/b2257916-6a9f-11e7-aa74-901b0e5592c8/embeddings -downcase {}".format(eval_dataset, preds_file, "-extended" if extended else "")
    cmd_open = subprocess.check_output(cmd, shell=True)
    with open(file_name, "w") as fh:
        fh.write(cmd_open.decode("ascii"))


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


def get_file_list(topdir, identifiers=None, all_levels=False):
    """
    :param identifiers: a list of strings, any of which should be in the filename
    :param all_levels: get filenames recursively
    """
    if identifiers is None:
        identifiers = [""]
    filelist = []
    for root, dirs, files in os.walk(topdir):
        if not all_levels and (root != topdir):  # don't go deeper
            continue
        for filename in files:
            get = False
            for i in identifiers:
                if i in filename:
                    get = True
            if get:
                fullname = os.path.join(root, filename)
                filelist.append(fullname)

    return filelist
