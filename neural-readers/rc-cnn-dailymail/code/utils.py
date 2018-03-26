import subprocess
import webbrowser

import lasagne
import numpy as np

import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
import os
import json

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

PLACEHOLDER_KEY = "@placeholder"


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
                logging.info("Inconsistent markup.")
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


def load_data(in_file, max_example=None, relabeling=True, remove_notfound=False):
    """
        load Clicr train/dev/test data from json files

    :param relabeling: relabels entity names in such a way that entities are unique only to the document.
    :param remove_notfound: whether to ignore the instances whose answers are not found in the document.
    This will bring down the output entity label space; a single entity name can stand for different concepts (anonymization).
    """

    documents = []
    questions = []
    answers = []
    ids = []  # [(qa_id: entity_dict)]
    num_examples = 0
    num_all = 0
    dataset = load_json(in_file)

    for datum in dataset[DATA_KEY]:
        document = to_entities(datum[DOC_KEY][CONTEXT_KEY] + " " + datum[DOC_KEY][TITLE_KEY])  # TODO: move title to front
        document = document.lower()
        _d_words = document.split()

        assert document
        for qa in datum[DOC_KEY][QAS_KEY]:
            num_all += 1
            question = to_entities(qa[QUERY_KEY]).lower()
            assert question
            answer = ""
            for ans in qa[ANS_KEY]:
                if ans[ORIG_KEY] == "dataset":
                    answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
            assert answer
            # check if UMLS answer can be found in d
            if remove_notfound:
                if answer not in _d_words:
                    found_umls = False
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "UMLS":
                            umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                            if umls_answer in _d_words:
                                found_umls = True
                                answer = umls_answer
                    if not found_umls:
                        continue
            if relabeling:
                #assert answer in _d_words
                _q_words = question.split()
                entity_dict = {}
                entity_id = 0
                lst = _d_words + _q_words
                if not remove_notfound:
                    lst.append(answer)
                for word in lst:
                    if (word.startswith('@entity')) and (word not in entity_dict):
                        entity_dict[word] = '@entity' + str(entity_id)
                        entity_id += 1
                q_words = [entity_dict[w] if w in entity_dict else w for w in _q_words]
                d_words = [entity_dict[w] if w in entity_dict else w for w in _d_words]
                question = " ".join(q_words)
                document = " ".join(d_words)
                answer = entity_dict[answer]
                inv_entity_dict = {ent_id: ent_ans for ent_ans, ent_id in entity_dict.items()}
                assert len(entity_dict) == len(inv_entity_dict)
                ids.append((qa[ID_KEY], inv_entity_dict))
            else:
                ids.append((qa[ID_KEY], None))
            documents.append(document)
            questions.append(question)
            answers.append(answer)
            num_examples += 1
            if (max_example is not None) and (num_examples >= max_example):
                break
        if (max_example is not None) and (num_examples >= max_example):
            break

    logging.info('#Examples: %d' % len(documents))

    return documents, questions, answers, ids


def load_cnn_data(in_file, max_example=None, relabeling=True, has_ids=False):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    if has_ids:
        ids = []
    num_examples = 0
    f = open(in_file, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip()
        document = f.readline().strip().lower()
        if has_ids:
            id = f.readline().strip()

        if relabeling:
            q_words = question.split(' ')
            d_words = document.split(' ')
            #assert answer in d_words
            entity_dict = {}
            entity_id = 0
            for word in d_words + q_words + ([answer] if answer not in d_words else []):
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
            d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
            answer = entity_dict[answer]
            question = ' '.join(q_words)
            document = ' '.join(d_words)
            inv_entity_dict = {ent_id: ent_ans for ent_ans, ent_id in entity_dict.items()}
            assert len(entity_dict) == len(inv_entity_dict)
            if has_ids:
                ids.append((id, inv_entity_dict))

        questions.append(question)
        answers.append(answer)
        documents.append(document)
        if not relabeling and has_ids:
            ids.append(id)
        num_examples += 1

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, answers, ids if has_ids else None)


def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def vectorize(examples, word_dict, entity_dict,
              sort_by_len=True, verbose=True, remove_notfound=False, relabeling=False):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(examples[0]), len(entity_dict))).astype(config._floatX)
    in_y = []
    ids = None
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
        seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            #if not remove_notfound and not relabeling:
            #    if a in d_words:
            #        in_l[idx, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0
            #    else:  # if a not in d, assume enlarge the a set to all possible answers
            #        in_l[idx, :] = 1.0
            #else:
            in_l[idx, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0

            in_y.append(entity_dict[a] if a in entity_dict else 0)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[0])))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        if len(examples) == 4:
            ids = examples[3]
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_l = in_l[sorted_index]
        in_y = [in_y[i] for i in sorted_index]
        if ids is not None:
            ids = [ids[i] for i in sorted_index]
    return in_x1, in_x2, in_l, in_y, ids


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(config._floatX)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def get_dim(in_file):
    fh = open(in_file)
    fh.readline()
    line = fh.readline()
    return len(line.split()) - 1


def gen_embeddings(word_dict, dim, in_file=None,
                   init=lasagne.init.Uniform()):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
    embeddings = init((num_words, dim))
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            sp = line.split()
            if len(sp) == 2:  # loading w2v-style
                continue
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))


def write_preds(preds, file_name):
    """
    :param preds: {q_id: answer, ...}

    Write predictions as a json file.
    """
    save_json(preds, file_name)


def write_att(atts, file_name):
    """
    :param atts: {q_id: [(w,att),...], ...}

    Write as a json file.
    """
    save_json(atts, file_name)


def att_html(preds_file_name, preds_att_file_name, qid, html_file):
    """
    :param preds_file_name: contains predictions for all queries (qids)
    :param preds_att_file_name: contains attention weights for all queries/docs
    :param qid: the query id that we want to obtain html for
    :param html_file: name of the html output file
    """

    header = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <meta name="robots" content="noindex, nofollow">
    <meta name="googlebot" content="noindex, nofollow">
    <script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.js"></script>
    <link rel="stylesheet" type="text/css" href="/css/result-light.css">
    <style type="text/css">
    </style>
    <title></title>
    <script type="text/javascript">//<![CDATA[
    window.onload=function(){
    var words = ["""

    footer_start = """];

    $('#text').html($.map(words, function(w) {
    return '<span style="background-color:hsl(360,100%,' + ((1-w.attention) * 50 + 50) + '%)">' + w.word + ' </span>'
    }))

    }//]]>

    </script>


    </head>

    <body>
    <div id="text">text goes here</div><br>
    """

    footer_end = """
    <script>
    // tell the embed parent frame the height of the content
    if (window.parent && window.parent.parent){
window.parent.parent.postMessage(["resultsFrame", {
height: document.body.getBoundingClientRect().height,
slug: "r3o4mgum"
}], "*")
}
</script>

</body>

</html>
    """

    def js_format(w, att):
        return "{{\'word\': \'{}\', \'attention\': {}}}".format(w.replace("\'", "&quot;").replace("\"", "&quot;"), att)

    def rescale(atts):
        atts = np.array(atts)
        max_v = np.max(atts)
        min_v = np.min(atts)
        diff = max_v - min_v
        return (atts - min_v) / diff

    preds = load_json(preds_file_name)
    atts = load_json(preds_att_file_name)
    # get the predicted answer for the right query id
    a = preds[qid]
    # get attention weights
    d_att = atts[qid]["d_att"]
    # get the query
    q = atts[qid]["q"]

    core_ws = []
    core_atts = []
    for w, att in d_att:
        core_ws.append(w)
        core_atts.append(att)

    core_atts = rescale(core_atts)
    core_lst = [js_format(w, att) for w, att in zip(core_ws, core_atts)]
    core = ",\n".join(core_lst).replace("@entity", "@")

    footer_qa = """
        <div id="q">{}</div><br>
        <div id="a">{}</div>
        """.format(" ".join(q).replace("@entity", "@").replace("@placeholder", "<span style=\"font-weight:bold\">@placeholder</span>"),
                   "<span style=\"font-weight:bold\">{}</span>".format(a))
    with open(html_file, "w") as fh:
        fh.write(header)
        fh.write(core)
        fh.write(footer_start + footer_qa + footer_end)


def external_eval(preds_file, file_name, eval_data="dev"):
    eval_dataset = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/" + "{}1.0.json".format(eval_data)
    logging.info("External evaluation, penalizing unanswered...")
    cmd = "python3 ~/Apps/bmj_case_reports/evaluate.py -test_file {} -prediction_file {} -embeddings_file /nas/corpora/accumulate/clicr/embeddings/b2257916-6a9f-11e7-aa74-901b0e5592c8/embeddings -downcase -extended".format(eval_dataset, preds_file)
    cmd_open = subprocess.check_output(cmd, shell=True)
    with open(file_name, "w") as fh:
        fh.write(cmd_open)


def update_plot(eval_iter, train_accs, dev_accs, file_name):
    header = """
        <html>
        <head>
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <script type="text/javascript">
        google.charts.load('current', {
        packages: ['corechart', 'line']
        });
        google.charts.setOnLoadCallback(drawCrosshairs);
        function drawCrosshairs() {
        var data = new google.visualization.DataTable();
        data.addColumn('number', 'X');
        data.addColumn('number', 'Train');
        data.addColumn('number', 'Dev');

        data.addRows([
        """

    footer = """  ]);
        var options = {
        hAxis: {
          title: 'N updates'
        },
        vAxis: {
          title: 'Accuracy'
        },
        colors: ['#a52714', '#097138'],
        crosshair: {
          color: '#000',
          trigger: 'selection'
        }
        };
        var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
        var options = {
        'width': 2000,
        'height': 1200
        };
        chart.draw(data, options);
        }
        </script>
        </head>
        <body>
        <div id="chart_div" style="width: 2000px; height: 1200px;"></div>
        </body>
        </html>
        """
    with open(file_name, "w") as fh:
        fh.write(header)
        steps = range(eval_iter, (eval_iter * len(train_accs))+eval_iter, eval_iter)
        for step, train_acc, dev_acc in zip(steps, train_accs, dev_accs):
            fh.write(("[{},{},{}],\n".format(step, train_acc, dev_acc)))
        fh.write(footer)
    if len(train_accs) == 5:
        url = "file://" + os.path.abspath(file_name)
        #webbrowser.get('firefox').open_new_tab(url)
