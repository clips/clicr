import argparse
import os
import re
from collections import Counter, defaultdict

from util import load_json, get_file_list
from text import remove_concept_marks

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


def to_lower(w, low):
    return w.lower() if low else w


def print_data_format():
    data_format = """
    {'version': STR,
     'data': [{'document': {'context': STR,
                            'title': STR,
                            'qas': [{'query': STR,
                                     'id': STR,
                                     'answers: [{'text': STR,
                                                 'origin': 'dataset'|'UMLS',
                                                 'cui': STR,
                                                 'sem_type': STR
                                                },...]
                                    },...]
                           },
               'source': STR
              },...]
    }

    """

    return data_format


def dataset_instance(version, data):
    return {"version": version, "data": data}


def get_different_cuis(dataset_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json"):
    dataset = load_json(dataset_file)
    data = dataset[DATA_KEY]
    cuis = set()
    for datum in data:
        for qa in datum[DOC_KEY][QAS_KEY]:
            for ans in qa[ANS_KEY]:
                if ans[ORIG_KEY] == "dataset":
                    if ans[CUI_KEY]:
                        cuis.add(int(ans[CUI_KEY][1:]))

    print("Number of different cuis: {}".format(len(cuis)))
    return cuis


def get_contexts(dataset_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/train1.0.json",
                 output_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/train1.0.txt",
                 downcase=False):
    """
    Gets passage text with no concept annotations.
    """
    dataset = load_json(dataset_file)
    data = dataset[DATA_KEY]
    n_all = 0
    all_contexts = ""

    for datum in data:
        new_context = "\n" + datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        all_contexts += remove_concept_marks(new_context)
        curr_queries = set()
        for qa in datum[DOC_KEY][QAS_KEY]:
            a = ""
            for ans in qa[ANS_KEY]:
                if ans[ORIG_KEY] == "dataset":
                    a = ans[TXT_KEY]
            assert a
            curr_queries.add(remove_concept_marks(qa[QUERY_KEY]).replace(PLACEHOLDER_KEY, a))
        all_contexts += "\n" + "\n".join(curr_queries)
        n_all += 1
    print(n_all)

    all_contexts = all_contexts.replace("\n\n", "\n")
    with open(output_file, "w") as fh:
        fh.write(all_contexts.lower() if downcase else all_contexts)


def get_doc_ids(dataset_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dataset1.0.json"):
    doc_ids = set()
    dataset = load_json(dataset_file)
    data = dataset[DATA_KEY]
    for datum in data:
        doc_ids.add(datum[SOURCE_KEY])

    return doc_ids


def get_article_series(dir="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/BMJ_Case_Reports/", dataset_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dataset1.0.json"):
    """
    :return: {article_id: series_name}
    """
    from describe_data import DATA_KEY, SOURCE_KEY

    series = {}
    for f in get_file_list(dir):
        if f.endswith(".full"):
            ls = open(f).readlines()
            for c, l in enumerate(ls):
                if '<ul class="series-titles">' in l:
                    s = ls[c+1].strip()
                    if s.startswith("<li>") and s.endswith("</li>"):
                        series[os.path.basename(f)[:-5]] = s[4:-5].lower()
                        break

    # filter out those not in json dataset
    from describe_data import get_doc_ids
    doc_ids = get_doc_ids(dataset_file)

    return {doc_id: s for doc_id, s in series.items() if doc_id in doc_ids}


def get_article_series_table(cutoff=230):
    def apply_cutoff(cs):
        new_cs = Counter()
        for series, count in cs.items():
            if count < cutoff:
                new_cs["other"] += count
            else:
                new_cs[series] = count
        return new_cs

    counts = Counter(get_article_series().values())
    rephrasing = {
        "case reports": "case report",
        "images in…": "images in...",
        "images in..": "images in...",
        "images in ….": "images in...",
        "images in …": "images in...",
        "image in...": "images in...",
        "novel treatment": "novel treatment (new drug/intervention; established drug/procedure in new situation)",
        "novel treatment (new drug/interventions; established drug/procedure in new situation)" : "novel treatment (new drug/intervention; established drug/procedure in new situation)",
        "full cases": "other full case",
        "unexpected outcome": "unexpected outcome (positive or negative) including adverse drug reactions",
        "unusual presentation of more common disease": "unusual presentation of more common disease/injury"
        }
    fixed_counts = Counter()
    for series, count in counts.items():
        if series in rephrasing:
            fixed_counts[rephrasing[series]] += count
        else:
            fixed_counts[series] += count
    fixed_counts = apply_cutoff(fixed_counts)
    total = sum(fixed_counts.values())
    for series, count in fixed_counts.most_common():
        print(r"{} & {} ({:.0f}) \\".format(series, count, 100*count/total))


def get_article_specialty(dir="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/BMJ_Case_Reports/", dataset_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dataset1.0.json"):
    """
    :return: {article_id: {spec1, ...}}
    """

    specs = {}
    for f in get_file_list(dir):
        if f.endswith(".full"):
            ls = open(f).readlines()
            for c, l in enumerate(ls):
                if '<meta content="' in l and 'name="DC.subject"' in l:# or ('name="DC.subject"' in ls[c+1])):
                    match = re.search('<meta content=\"(.*)\" name=\"DC\.subject', l)
                    if match:
                        specialties = match.group(1)
                        specs[os.path.basename(f)[:-5]] = set(specialties.lower().split("; "))
                        break
                elif '<meta content="' in l and ('name="DC.subject"' not in l and 'name="DC.subject"' in ls[c+1]):
                    match = re.search('<meta content=\"(.*)\"', l)
                    if match:
                        specialties = match.group(1)
                        specs[os.path.basename(f)[:-5]] = set(specialties.lower().split("; "))
                        break

    # filter out those not in json dataset
    from describe_data import get_doc_ids
    doc_ids = get_doc_ids(dataset_file)
    #print(len(specs))
    #print(Counter(i for s in specs.values() for i in s))

    return {doc_id: s for doc_id, s in specs.items() if doc_id in doc_ids}


def get_article_specialty_table(cutoff=390):
    def apply_cutoff(cs):
        new_cs = Counter()
        for series, count in cs.items():
            if count < cutoff:
                new_cs["other"] += count
            else:
                new_cs[series] = count
        return new_cs

    counts = Counter(i for s in get_article_specialty().values() for i in s)

    fixed_counts = apply_cutoff(counts)
    total = sum(fixed_counts.values())
    for spec, count in fixed_counts.most_common():
        print(r"{} & {} ({:.0f}) \\".format(spec, count, 100*count/total))


def get_article_specialty_table_R(top=50):
    counts = Counter(i for s in get_article_specialty().values() for i in s)
    total = sum(counts.values())
    print("\t".join([spec for spec, count in counts.most_common()][3:top+3]))
    perc = [str(100 * count / total) for spec, count in counts.most_common()][3:top+3]
    print("\t".join(perc))


def get_article_specialty_table_R_vertical(top=50):
    counts = Counter(i for s in get_article_specialty().values() for i in s)
    total = sum(counts.values())
    for spec, count in counts.most_common()[0:top+0]:
        print("{}\t{}".format(spec, 100 * count / total))


def plot_article_specialty():
    header = """
    <html>
    <head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {

        var data = google.visualization.arrayToDataTable([
          ['Specialty', 'Percentage'],"""

    footer = """]);
    data.sort([{column: 1}]);
    var options = {
          title: 'Clicr specialties',
          sliceVisibilityThreshold: .01,
          pieSliceText: 'label'
        };
    var chart = new google.visualization.PieChart(document.getElementById('piechart'));
    chart.draw(data, options);
      }
    </script>
    </head>
    <body>
    <div id="piechart" style="width: 900px; height: 500px;"></div>
    </body>
    </html>
    """
    specs = get_article_specialty()
    specs_c = Counter(i for s in specs.values() for i in s)
    specs_lst = []
    for spec, freq in specs_c.items():
        if not spec:
            spec = "unspecified"
        specs_lst.append("['{0}', {1}]".format(spec.replace("\'", "&quot;").replace("\"", "&quot;"), freq))
    core = ",\n".join(specs_lst)

    return header + core + footer


def plot_article_series():
    header = """
    <html>
    <head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {

        var data = google.visualization.arrayToDataTable([
          ['Specialty', 'Percentage'],"""

    footer = """]);
    data.sort([{column: 1}]);
    var options = {
          title: 'Clicr series',
          sliceVisibilityThreshold: .01,
          pieSliceText: 'label'
        };
    var chart = new google.visualization.PieChart(document.getElementById('piechart'));
    chart.draw(data, options);
      }
    </script>
    </head>
    <body>
    <div id="piechart" style="width: 900px; height: 500px;"></div>
    </body>
    </html>
    """
    series = get_article_series()
    series_c = Counter(series.values())
    series_lst = []
    for spec, freq in series_c.items():
        if not spec:
            spec = "unspecified"
        series_lst.append("['{0}', {1}]".format(spec.replace("\'", "&quot;").replace("\"", "&quot;"), freq))
    core = ",\n".join(series_lst)

    return header + core + footer


class GeneralStats:
    def __init__(self, dataset_file=None):
        self.dataset_file = dataset_file
        self.dataset = load_json(self.dataset_file) if self.dataset_file is not None else None

    def n_cases(self):
        return len(self.dataset[DATA_KEY])

    def n_queries(self):
        n_q = 0
        for datum in self.dataset[DATA_KEY]:
            n_q += len(datum[DOC_KEY][QAS_KEY])
        return n_q

    def avg_n_queries(self, n_q, n_c):
        return n_q / n_c

    def n_words(self):
        """
        Effective size of the dataset.

        N.B. this includes recounting of passages
        """
        n_w = 0
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                n_w += len(datum[DOC_KEY][TITLE_KEY].split())
                n_w += len(datum[DOC_KEY][CONTEXT_KEY].split())
                n_w += len(qa[QUERY_KEY].split())
                for a in qa[ANS_KEY]:
                    if a["origin"] == "dataset":
                        n_w += len(a[TXT_KEY].split())
        return n_w

    def n_words_passages(self, recounting=True):
        """
        Effective size of the dataset in tokens of passages.

        :param recounting: whether to include recounting of passages
        """
        n_w = 0
        for datum in self.dataset[DATA_KEY]:
            if recounting:
                for qa in datum[DOC_KEY][QAS_KEY]:
                    n_w += len(datum[DOC_KEY][CONTEXT_KEY].split()) + len(datum[DOC_KEY][TITLE_KEY].split())
            else:
                n_w += len(datum[DOC_KEY][CONTEXT_KEY].split()) + len(datum[DOC_KEY][TITLE_KEY].split())
        return n_w

    def n_words_passages_dist(self):
        """
        Distribution of effective size of the dataset in tokens of passages.

        N.B. this includes recounting of passages
        """
        #len_to_n = Counter()
        #for datum in self.dataset[DATA_KEY]:
        #    for qa in datum[DOC_KEY][QAS_KEY]:
        #        len_to_n[len(datum[DOC_KEY][CONTEXT_KEY].split()) + len(datum[DOC_KEY][TITLE_KEY].split())] += 1
        #total = sum(len_to_n.values())

        #return {length: n/total for length, n in len_to_n.items()}

        len_to_n = []
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                len_to_n.append(len(datum[DOC_KEY][CONTEXT_KEY].split()) + len(datum[DOC_KEY][TITLE_KEY].split()))

        return len_to_n

    def n_words_passages_year_dist(self):
        """
        Distribution of effective size of the dataset in tokens of passages. Per year.

        N.B. this includes recounting of passages
        """
        year_to_n = {}
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                m = re.search("[.-](2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016)[.-]", datum[SOURCE_KEY])
                if m is not None:
                    year = m.group()[1:-1]
                    if year not in year_to_n:
                        year_to_n[year] = []
                    year_to_n[year].append(len(datum[DOC_KEY][CONTEXT_KEY].split()) + len(datum[DOC_KEY][TITLE_KEY].split()))

        return year_to_n

    def avg_len_case(self, n_q, n_w):
        """
        Average length of case in tokens.
        """
        return n_w / n_q

    def avg_len_passage(self, n_q, n_w_passages):
        """
        Average length of passage in tokens.
        """
        return n_w_passages / n_q

    def vocabulary_passage(self, lowercase=True):
        v = Counter()
        for datum in self.dataset[DATA_KEY]:
            for w in remove_concept_marks(datum[DOC_KEY][TITLE_KEY]).split():
                v[to_lower(w, lowercase)] += 1
            for w in remove_concept_marks(datum[DOC_KEY][CONTEXT_KEY]).split():
                v[to_lower(w, lowercase)] += 1
        return v

    def vocabulary(self, lowercase=True, include_extended=False, remove_mark=False):
        def remove_marker(w):
            return w.replace("BEG__", "").replace("__END", "") if remove_mark else w
        v = Counter()
        for datum in self.dataset[DATA_KEY]:
            for w in datum[DOC_KEY][TITLE_KEY].split():
                v[to_lower(remove_marker(w), lowercase)] += 1
            for w in datum[DOC_KEY][CONTEXT_KEY].split():
                v[to_lower(remove_marker(w), lowercase)] += 1
            for qa in datum[DOC_KEY][QAS_KEY]:
                for w in qa[QUERY_KEY]:
                    v[to_lower(remove_marker(w), lowercase)] += 1
                for a in qa[ANS_KEY]:
                    if a[ORIG_KEY] == "dataset" or include_extended:
                        for w in a[TXT_KEY].split():
                            v[to_lower(remove_marker(w), lowercase)] += 1
        return v

    def vocabulary_size(self, v):
        return len(v)

    def entities(self, lowercase=True, include_extended=False):
        from baselines import read_concepts
        e = Counter()
        for datum in self.dataset[DATA_KEY]:
            title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
            e.update([to_lower(" ".join(concept), lowercase) for concept in read_concepts(title_and_passage)])
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                e.update([to_lower(" ".join(concept), lowercase) for concept in read_concepts(qa[QUERY_KEY])])
                for a in qa[ANS_KEY]:
                    if a[ORIG_KEY] == "dataset" or include_extended:
                        e[to_lower(a[TXT_KEY], lowercase)] += 1
        return e

    def entities_passage(self, lowercase=True):
        from baselines import read_concepts
        e = Counter()
        for datum in self.dataset[DATA_KEY]:
            title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
            e.update([to_lower(" ".join(concept), lowercase) for concept in read_concepts(title_and_passage)])
        return e

    def n_entities(self, e):
        return len(e)

    def get_answer_texts(self, origin=None, what=TXT_KEY):
        """
        :param origin: Set whether to take into account only original answer (origin="dataset"), the expanded
        answers (origin="UMLS") or any (origin=None).
        """
        def to_yield():
            if origin is None:
                return True
            if (a[ORIG_KEY] == "dataset" and origin == "dataset") or (a[ORIG_KEY] == "UMLS" and origin == "UMLS"):
                return True
            return False

        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                for a in qa[ANS_KEY]:
                    if to_yield():
                        yield a[what]

    def answer_length(self, origin="dataset"):
        ans_lengths = Counter()

        for a_txt in self.get_answer_texts(origin=origin):
            ans_lengths[len(a_txt.split())] += 1

        return ans_lengths

    def most_frequent_answers(self, origin="dataset"):
        ans_freq = Counter()

        for a_txt in self.get_answer_texts(origin=origin):
            ans_freq[a_txt.lower()] += 1

        return ans_freq

    def most_frequent_answer_types(self, origin="dataset"):
        ans_freq = Counter()

        for a_type in self.get_answer_texts(origin=origin, what=SEMTYPE_KEY):
            ans_freq[a_type] += 1

        return ans_freq

    def percentage_of_ans_in_docs(self, include_extended=False):
        """
        Find out what proportion of answers actually occur in documents.
        NB: this is based on pure word matching. This is not the same as the percentage of
        entity (concept) answers found in text.

        @param include_extended: whether to use expanded answers in counting.
        """
        n_all = 0
        n_found = 0
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                n_all += 1
                text = remove_concept_marks(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                for ans in qa[ANS_KEY]:
                    if include_extended:
                        if re.search(re.escape(ans[TXT_KEY]), text) is not None:
                            n_found += 1
                            break
                    else:
                        if ans[ORIG_KEY] == "dataset":
                            if re.search(re.escape(ans[TXT_KEY]), text) is not None:
                                n_found += 1
        assert n_all > 0
        assert n_found <= n_all
        return 100 * n_found / n_all

    def percentage_of_concept_ans_in_docs(self, include_extended=False):
        """
        Where answer is found in any passage.
        """
        from baselines import read_concepts
        n_all = 0
        n_found = 0
        doc_concept_set = set()
        for datum in self.dataset[DATA_KEY]:
            title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
            concept_set = {" ".join(concept).lower() for concept in read_concepts(title_and_passage)}
            doc_concept_set.update(concept_set)
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                ans = qa[ANS_KEY]
                a = ""
                for _a in ans:
                    if _a[ORIG_KEY] == "dataset":
                        a = _a[TXT_KEY]
                assert a
                if a.lower() in doc_concept_set:
                    n_found += 1
                elif include_extended:
                    is_found = False
                    for _a in ans:
                        if _a[ORIG_KEY] == "UMLS":
                            a = _a[TXT_KEY]
                            if a.lower() in doc_concept_set:
                                is_found = True
                    if is_found:
                        n_found += 1
                n_all += 1
        return 100 * n_found / n_all

    def percentage_of_concept_ans_in_doc(self, include_extended=False):
        """
        Where answer is found in the relevant passage.
        """
        from baselines import read_concepts
        n_all = 0
        n_found = 0
        for datum in self.dataset[DATA_KEY]:
            title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
            concept_set = {" ".join(concept).lower() for concept in read_concepts(title_and_passage)}
            for qa in datum[DOC_KEY][QAS_KEY]:
                ans = qa[ANS_KEY]
                a = ""
                for _a in ans:
                    if _a[ORIG_KEY] == "dataset":
                        a = _a[TXT_KEY]
                assert a
                if a.lower() in concept_set:
                    n_found += 1
                elif include_extended:
                    is_found = False
                    for _a in ans:
                        if _a[ORIG_KEY] == "UMLS":
                            a = _a[TXT_KEY]
                            if a.lower() in concept_set:
                                is_found = True
                    if is_found:
                        n_found += 1
                n_all += 1

        return 100 * n_found / n_all


def ratio_ans(train_file, dev_file, test_file):
    """
    How many answers in dev/test were observed as answers in the training set?
    """
    stats_tr = GeneralStats(train_file)
    stats_de = GeneralStats(dev_file)
    stats_te = GeneralStats(test_file)

    ans_tr = set(stats_tr.most_frequent_answers(origin="dataset").keys())
    ans_de = set(stats_de.most_frequent_answers(origin="dataset").keys())
    ans_te = set(stats_te.most_frequent_answers(origin="dataset").keys())

    common_de = len(ans_tr & ans_de)
    ratio_de = common_de / len(ans_de)
    common_te = len(ans_tr & ans_te)
    ratio_te = common_te / len(ans_te)

    return ratio_de, ratio_te

def ratio_ans_fq(train_file, dev_file, test_file):
    """
    For the answers from dev/test that were observed also in the train, what is the mean frequency in the train set,
    and what is the std?
    """
    import numpy as np

    stats_tr = GeneralStats(train_file)
    stats_de = GeneralStats(dev_file)
    stats_te = GeneralStats(test_file)

    ans_tr = set(stats_tr.most_frequent_answers(origin="dataset").keys())
    ans_de = set(stats_de.most_frequent_answers(origin="dataset").keys())
    ans_te = set(stats_te.most_frequent_answers(origin="dataset").keys())

    common_de = ans_tr & ans_de
    fq_wrt_de = [fq for ans, fq in stats_tr.most_frequent_answers(origin="dataset").items() if ans in common_de]
    common_te = ans_tr & ans_te
    fq_wrt_te = [fq for ans, fq in stats_tr.most_frequent_answers(origin="dataset").items() if ans in common_te]

    de_mean, de_median, de_std = np.mean(fq_wrt_de), np.median(fq_wrt_de), np.std(fq_wrt_de)
    te_mean, te_median, te_std = np.mean(fq_wrt_te), np.median(fq_wrt_te), np.std(fq_wrt_te)

    return de_mean, de_median, de_std, te_mean, te_median, te_std

def print_general_stats(train_file, dev_file, test_file):
    # combine all train/dev/test first
    stats = GeneralStats()
    train = load_json(train_file)
    dev = load_json(dev_file)
    test = load_json(test_file)
    stats.dataset = dataset_instance(train[VERSION_KEY], train[DATA_KEY] + dev[DATA_KEY] + test[DATA_KEY])

    n_cases = stats.n_cases()
    print("N of cases: {}".format(n_cases))
    n_queries = stats.n_queries()
    print("N of queries: {}".format(n_queries))
    #print("Avg n of q: {}".format(stats.avg_n_queries(n_queries, n_cases)))
    #n_words = stats.n_words()
    #print("N of tokens in cases: {}".format(n_words))
    n_words_passages = stats.n_words_passages(recounting=False)
    print("N of tokens in passages without recounting: {}".format(n_words_passages))
    n_words_passages = stats.n_words_passages(recounting=True)
    print("N of tokens in passages with recounting: {}".format(n_words_passages))
    #print("Avg len of case: {}".format(stats.avg_len_case(n_queries, n_words)))
    print("Avg len of passage: {}".format(stats.avg_len_passage(n_queries, n_words_passages)))
    #print("Vocabulary size (w/ extended): {}".format(stats.vocabulary_size(stats.vocabulary(include_extended=True))))
    #print("Vocabulary size (w/o extended): {}".format(stats.vocabulary_size(stats.vocabulary(include_extended=False))))
    print("N of word types in passages: {}".format(stats.vocabulary_size(stats.vocabulary_passage(lowercase=True))))
    #print("N of entity types (w/ extended): {}".format(stats.n_entities(stats.entities(include_extended=True))))
    #print("N of entity types (w/o extended): {}".format(stats.n_entities(stats.entities(include_extended=False))))
    print("N of entity types in passages: {}".format(stats.n_entities(stats.entities_passage(lowercase=True))))

    stats_train = GeneralStats(train_file)
    print("N of queries in train: {}".format(stats_train.n_queries()))
    stats_dev = GeneralStats(dev_file)
    print("N of queries in dev: {}".format(stats_dev.n_queries()))
    stats_test = GeneralStats(test_file)
    print("N of queries in test: {}".format(stats_test.n_queries()))

    ans_l = stats.answer_length()
    total = sum(ans_l.values())
    print("Distribution of ans length: {}".format([(size, n, round(100*n/total)) for size, n in ans_l.most_common()]))
    most_freq_ans = stats.most_frequent_answers()
    print("Most freq ans: {}".format(most_freq_ans.most_common(20)))
    #print("N of distinct answers: {} ({})".format(len(most_freq_ans), sum(most_freq_ans.values())))
    print("N of distinct answers: {}".format(len(most_freq_ans)))
    most_freq_ans = stats.most_frequent_answers(origin=None)
    #print("N of distinct answers (incl. extended): {} ({})".format(len(most_freq_ans), sum(most_freq_ans.values())))
    print("N of distinct answers (incl. extended): {}".format(len(most_freq_ans)))
    most_freq_ans_types = stats.most_frequent_answer_types()
    total = sum(most_freq_ans_types.values())
    print("Most freq ans types: {}".format([(typ, n, round(100*n/total)) for typ, n in most_freq_ans_types.most_common()]))

    print("% answers found in passage: {}".format(stats.percentage_of_concept_ans_in_doc(include_extended=True)))
    print("% answers found in any passage: {}".format(stats.percentage_of_concept_ans_in_docs(include_extended=True)))


def print_dist():
    stats = GeneralStats()
    for i in sorted(stats.n_words_passages_dist()):
        print(i)


def print_year_dist():
    stats = GeneralStats()
    for year, len_lst in stats.n_words_passages_year_dist().items():
        print("{}: {}".format(year, sum(len_lst)/len(len_lst)))


def misc():
    stats = GeneralStats()
    most_freq_ans = stats.most_frequent_answers()
    for w, f in most_freq_ans.most_common():
        if len(w.split()) == 3:
            print(w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Statistics for the CliCR dataset')
    parser.add_argument('-train_file', help='Path to the training set in json format.')
    parser.add_argument('-dev_file', help='Path to the dev set in json format.')
    parser.add_argument('-test_file', help='Path to the test set in json format.')
    args = parser.parse_args()

    #print(print_data_format())
    #print(percentage_of_ans_in_docs(include_extended=False))
    #get_contexts(downcase=True)
    #p=percentage_of_concept_ans_in_doc(include_extended=False)
    #print("% found in doc: {}, extended={}".format(p, False))
    #p=percentage_of_concept_ans_in_doc(include_extended=True)
    #print("% found in doc: {}, extended={}".format(p, True))
    #p=percentage_of_concept_ans_in_docs(include_extended=False)
    #print("% found in docs: {}, extended={}".format(p, False))
    #p=percentage_of_concept_ans_in_docs(include_extended=True)
    #print("% found in docs: {}, extended={}".format(p, True))


    #get_article_series()
    #get_article_specialty_table()
    #get_article_specialty_table_R_vertical(top=25)
    #print(plot_article_specialty())
    #print(plot_article_series())

    print_general_stats(args.train_file, args.dev_file, args.test_file)
    #print(ratio_ans(args.train_file, args.dev_file, args.test_file))
    #print(ratio_ans_fq(args.train_file, args.dev_file, args.test_file))
    #print_dist()
    #print_year_dist()
    #misc()
