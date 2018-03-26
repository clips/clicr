import codecs
import glob
import os

from config import MAX_WORD_LEN
from utils.utils import load_json, DATA_KEY, DOC_KEY, TITLE_KEY, CONTEXT_KEY, QUERY_KEY, QAS_KEY, ANS_KEY, ORIG_KEY, \
    TXT_KEY, to_entities, ID_KEY

SYMB_BEGIN = "@begin"
SYMB_END = "@end"


class Data:
    def __init__(self, dictionary, num_entities, training, validation, test, train_relabeling_dicts=None,
                 val_relabeling_dicts=None, test_relabeling_dicts=None):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}
        self.train_relabeling_dicts = train_relabeling_dicts
        self.val_relabeling_dicts = val_relabeling_dicts
        self.test_relabeling_dicts = test_relabeling_dicts


class DataPreprocessorNovice:
    """ Uses plain format for CNN training part, and json format for Clicr dev and test parts."""

    def preprocess(self, question_dir, ent_setup, no_training_set=False, use_chars=True):
        """
        preprocess all data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set is True.
        """
        vocab_f = os.path.join(question_dir,"vocab_novice.txt")
        word_dictionary, char_dictionary, num_entities = \
                self.make_dictionary(question_dir, vocab_file=vocab_f, ent_setup=ent_setup, remove_notfound=False)
        dictionary = (word_dictionary, char_dictionary)
        if no_training_set:
            training = None
        else:
            print("preparing training data ...")
            training = self.parse_all_files(question_dir + "/training", dictionary, use_chars)
        train_relabeling_dicts = None
        print("preparing validation data ...")
        validation, val_relabeling_dicts = self.parse_file(question_dir + "/dev1.0.json", dictionary, use_chars, ent_setup, remove_notfound=False)
        print("preparing test data ...")
        test, test_relabeling_dicts = self.parse_file(question_dir + "/test1.0.json", dictionary, use_chars, ent_setup, remove_notfound=False)

        data = Data(dictionary, num_entities, training, validation, test, train_relabeling_dicts, val_relabeling_dicts, test_relabeling_dicts)
        return data

    def make_dictionary(self, question_dir, vocab_file, ent_setup, remove_notfound):
        if os.path.exists(vocab_file):
            print("loading vocabularies from " + vocab_file + " ...")
            vocabularies = list(map(lambda x: x.strip(), codecs.open(vocab_file, encoding="utf-8").readlines()))
        else:
            print("no " + vocab_file + " found, constructing the vocabulary list ...")

            fnames = glob.glob(question_dir + "/training/*.question")
            dataset_dev = load_json(question_dir + "dev1.0.json")
            dataset_test = load_json(question_dir + "test1.0.json")

            # first TRAINING ****************************************
            vocab_set = set()
            n = 0.
            for fname in fnames:

                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline().split()
                fp.readline()
                query = fp.readline().split()
                fp.close()

                vocab_set |= set(document) | set(query)

                # show progress
                n += 1
                if n % 10000 == 0:
                    print('%3d%%' % int(100 * n / len(fnames)))

            # DEV + TEST *******************************************
            assert ent_setup == "ent-anonym" or ent_setup == "ent"
            for datum in dataset_test[DATA_KEY] + dataset_dev[DATA_KEY]:
                document = to_entities(
                    datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                document = document.lower()

                assert document
                for qa in datum[DOC_KEY][QAS_KEY]:
                    doc_raw = document.split()
                    question = to_entities(qa[QUERY_KEY]).lower()
                    assert question
                    qry_raw = question.split()
                    ans_raw = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                    assert ans_raw
                    if remove_notfound:
                        if ans_raw not in doc_raw:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                                    if umls_answer in doc_raw:
                                        found_umls = True
                                        ans_raw = umls_answer
                            if not found_umls:
                                continue
                    if ent_setup == "ent-anonym":
                        entity_dict = {}
                        entity_id = 0
                        lst = doc_raw + qry_raw
                        lst.append(ans_raw)
                        for word in lst:
                            if (word.startswith('@entity')) and (word not in entity_dict):
                                entity_dict[word] = '@entity' + str(entity_id)
                                entity_id += 1
                        qry_raw = [entity_dict[w] if w in entity_dict else w for w in qry_raw]
                        doc_raw = [entity_dict[w] if w in entity_dict else w for w in doc_raw]
                        ans_raw = entity_dict[ans_raw]
                    vocab_set |= set(qry_raw)
                    vocab_set |= set(doc_raw)
                    vocab_set.add(ans_raw)
                    # show progress
                    n += 1
                    if n % 10000 == 0:
                        print(n)

            entities = set(e for e in vocab_set if e.startswith('@entity'))
            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities) + list(tokens)

            print("writing vocabularies to " + vocab_file + " ...")
            vocab_fp = codecs.open(vocab_file, "w", encoding="utf-8")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print("vocab_size = %d" % vocab_size)
        print("num characters = %d" % len(char_set))
        print("%d anonymoused entities" % num_entities)
        print("%d other tokens (including @placeholder, %s and %s)" % (
            vocab_size - num_entities, SYMB_BEGIN, SYMB_END))

        return word_dictionary, char_dictionary, num_entities

    def parse_file(self, file_path, dictionary, use_chars, ent_setup, remove_notfound):
        """
        parse a *.json dataset file into a list of questions, where each element is tuple(document, query, answer, filename, query_id)
        """
        questions = []
        w_dict, c_dict = dictionary[0], dictionary[1]
        relabeling_dicts = {}
        raw = load_json(file_path)
        for datum in raw[DATA_KEY]:
            document = to_entities(
                datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
            document = document.lower()

            assert document
            for qa in datum[DOC_KEY][QAS_KEY]:
                if ent_setup in ["ent-anonym", "ent"]:
                    doc_raw = document.split()
                    question = to_entities(qa[QUERY_KEY]).lower()
                    qry_id = qa[ID_KEY]
                    assert question
                    ans_raw = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                    assert ans_raw
                    if remove_notfound:
                        if ans_raw not in doc_raw:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                                    if umls_answer in doc_raw:
                                        found_umls = True
                                        ans_raw = umls_answer
                            if not found_umls:
                                continue
                    qry_raw = question.split()
                    if ent_setup == "ent-anonym":
                        entity_dict = {}
                        entity_id = 0
                        lst = doc_raw + qry_raw
                        lst.append(ans_raw)
                        for word in lst:
                            if (word.startswith('@entity')) and (word not in entity_dict):
                                entity_dict[word] = '@entity' + str(entity_id)
                                entity_id += 1
                        qry_raw = [entity_dict[w] if w in entity_dict else w for w in qry_raw]
                        doc_raw = [entity_dict[w] if w in entity_dict else w for w in doc_raw]
                        ans_raw = entity_dict[ans_raw]
                        inv_entity_dict = {ent_id: ent_ans for ent_ans, ent_id in entity_dict.items()}
                        assert len(entity_dict) == len(inv_entity_dict)
                        relabeling_dicts[qa[ID_KEY]] = inv_entity_dict

                    cand_e = [w for w in doc_raw if w.startswith('@entity')]
                    cand_raw = [[e] for e in cand_e]
                    # wrap the query with special symbols
                    qry_raw.insert(0, SYMB_BEGIN)
                    qry_raw.append(SYMB_END)
                    try:
                        cloze = qry_raw.index('@placeholder')
                    except ValueError:
                        print('@placeholder not found in ', qry_raw, '. Fixing...')
                        at = qry_raw.index('@')
                        qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                        cloze = qry_raw.index('@placeholder')

                    # tokens/entities --> indexes
                    doc_words = list(map(lambda w: w_dict[w], doc_raw))

                    # tokens/entities --> indexes
                    qry_words = list(map(lambda w: w_dict[w], qry_raw))
                    if use_chars:
                        qry_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                                                list(w)[:MAX_WORD_LEN])), qry_raw))
                    else:
                        qry_chars = []
                    ans = list(map(lambda w: w_dict.get(w, 0), ans_raw.split()))
                    cand = [list(map(lambda w: w_dict.get(w, 0), c)) for c in cand_raw]

                    if use_chars:
                        doc_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                                                list(w)[:MAX_WORD_LEN])), doc_raw))
                    else:
                        doc_chars = []

                    questions.append((doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze, qry_id))

                elif ent_setup == "no-ent":
                    # collect candidate ents using @entity marks
                    cand_e = [w for w in
                              to_entities(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]).lower().split() if
                              w.startswith('@entity')]
                    cand_raw = [e[len("@entity"):].split("_") for e in cand_e]
                    document = remove_entity_marks(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                    document = document.lower()
                    doc_raw = document.split()
                    question = remove_entity_marks(qa[QUERY_KEY]).lower()
                    qry_id = qa[ID_KEY]
                    assert question
                    qry_raw = question.split()
                    ans_raw = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            ans_raw = ans[TXT_KEY].lower()
                    assert ans_raw
                    if remove_notfound:
                        if ans_raw not in doc_raw:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = ans[TXT_KEY].lower()
                                    if umls_answer in doc_raw:
                                        found_umls = True
                                        ans_raw = umls_answer
                            if not found_umls:
                                continue

                    relabeling_dicts[qa[ID_KEY]] = None
                    # wrap the query with special symbols
                    qry_raw.insert(0, SYMB_BEGIN)
                    qry_raw.append(SYMB_END)
                    try:
                        cloze = qry_raw.index('@placeholder')
                    except ValueError:
                        print('@placeholder not found in ', qry_raw, '. Fixing...')
                        at = qry_raw.index('@')
                        qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                        cloze = qry_raw.index('@placeholder')

                    # tokens/entities --> indexes
                    doc_words = list(map(lambda w: w_dict[w], doc_raw))

                    # tokens/entities --> indexes
                    qry_words = list(map(lambda w: w_dict[w], qry_raw))
                    if use_chars:
                        qry_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                                                list(w)[:MAX_WORD_LEN])), qry_raw))
                    else:
                        qry_chars = []
                    ans = list(map(lambda w: w_dict.get(w, 0), ans_raw.split()))
                    cand = [list(map(lambda w: w_dict.get(w, 0), c)) for c in cand_raw]

                    if use_chars:
                        doc_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                                                list(w)[:MAX_WORD_LEN])), doc_raw))
                    else:
                        doc_chars = []

                    questions.append((doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze, qry_id))

                else:
                    raise ValueError

        return questions, relabeling_dicts

    def parse_one_file(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()
        doc_raw = raw[2].split() # document
        qry_raw = raw[4].split() # query
        ans_raw = raw[6].strip() # answer
        cand_raw = map(lambda x:x.strip().split(':')[0].split(), raw[8:]) # candidate answers

        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)
        try:
            cloze = qry_raw.index('@placeholder')
        except ValueError:
            print('@placeholder not found in ', fname, '. Fixing...')
            at = qry_raw.index('@')
            qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at+2])] + qry_raw[at+2:]
            cloze = qry_raw.index('@placeholder')

        # tokens/entities --> indexes
        doc_words = list(map(lambda w:w_dict[w], doc_raw))
        qry_words = list(map(lambda w:w_dict[w], qry_raw))
        if use_chars:
            doc_chars = list(map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']),
                list(w)[:MAX_WORD_LEN]), doc_raw))
            qry_chars = list(map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']),
                list(w)[:MAX_WORD_LEN]), qry_raw))
        else:
            doc_chars, qry_chars = [], []
        ans = list(map(lambda w:w_dict.get(w,0), ans_raw.split()))
        cand = [list(map(lambda w:w_dict.get(w,0), c)) for c in cand_raw]

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze

    def parse_all_files(self, directory, dictionary, use_chars):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        all_files = glob.glob(directory + '/*.question')
        questions = [self.parse_one_file(f, dictionary, use_chars) + (os.path.basename(f)[:-len(".question")],) for f in all_files]
        return questions


class DataPreprocessorClicr:
    def preprocess(self, question_dir, ent_setup, no_training_set=False, use_chars=True, remove_notfound=True):
        """
        preprocess all Clicr data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set is True.
        """
        vocab_f = os.path.join(question_dir, "vocab.txt")
        word_dictionary, char_dictionary, num_entities = \
            self.make_dictionary(question_dir, vocab_file=vocab_f, ent_setup=ent_setup, remove_notfound=remove_notfound)
        dictionary = (word_dictionary, char_dictionary)
        if no_training_set:
            training = None
            train_relabeling_dicts = None
        else:
            print("preparing training data ...")
            training, train_relabeling_dicts = self.parse_file(question_dir + "/train1.0.json", dictionary, use_chars, ent_setup, remove_notfound)
        print("preparing validation data ...")
        validation, val_relabeling_dicts = self.parse_file(question_dir + "/dev1.0.json", dictionary, use_chars, ent_setup, remove_notfound=False)
        print("preparing test data ...")
        test, test_relabeling_dicts = self.parse_file(question_dir + "/test1.0.json", dictionary, use_chars, ent_setup, remove_notfound=False)

        data = Data(dictionary, num_entities, training, validation, test, train_relabeling_dicts, val_relabeling_dicts, test_relabeling_dicts)
        return data

    def make_dictionary(self, question_dir, vocab_file, ent_setup, remove_notfound):
        vocab_file = "{}_stp{}_remove{}_py3".format(vocab_file, ent_setup, remove_notfound)
        if os.path.exists(vocab_file):
            print("loading vocabularies from " + vocab_file + " ...")
            vocabularies = list(map(lambda x: x.strip(), codecs.open(vocab_file, encoding="utf-8").readlines()))
        else:
            print("no " + vocab_file + " found, constructing the vocabulary list ...")
            vocab_set = set()
            n = 0.
            dataset_train = load_json(question_dir + "train1.0.json")
            dataset_dev = load_json(question_dir + "dev1.0.json")
            dataset_test = load_json(question_dir + "test1.0.json")

            if ent_setup in ["ent-anonym", "ent"]:  # treats each entity as a single token
                # train here (remove_notfound=True|False), dev/test below
                for datum in dataset_train[DATA_KEY]:
                    document = to_entities(
                        datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                    document = document.lower()

                    assert document
                    for qa in datum[DOC_KEY][QAS_KEY]:
                        doc_raw = document.split()
                        question = to_entities(qa[QUERY_KEY]).lower()
                        assert question
                        qry_raw = question.split()
                        ans_raw = ""
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "dataset":
                                ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                        assert ans_raw
                        if remove_notfound:
                            if ans_raw not in doc_raw:
                                found_umls = False
                                for ans in qa[ANS_KEY]:
                                    if ans[ORIG_KEY] == "UMLS":
                                        umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                                        if umls_answer in doc_raw:
                                            found_umls = True
                                            ans_raw = umls_answer
                                if not found_umls:
                                    continue
                        if ent_setup == "ent-anonym":  # anonymize
                            entity_dict = {}
                            entity_id = 0
                            lst = doc_raw + qry_raw
                            lst.append(ans_raw)
                            for word in lst:
                                if (word.startswith('@entity')) and (word not in entity_dict):
                                    entity_dict[word] = '@entity' + str(entity_id)
                                    entity_id += 1
                            qry_raw = [entity_dict[w] if w in entity_dict else w for w in qry_raw]
                            doc_raw = [entity_dict[w] if w in entity_dict else w for w in doc_raw]
                            ans_raw = entity_dict[ans_raw]
                        vocab_set |= set(qry_raw)
                        vocab_set |= set(doc_raw)
                        vocab_set.add(ans_raw)
                        # show progress
                        n += 1
                        if n % 10000 == 0:
                            print(n)
                # treat dev/test separately to allow remove_notfound=False
                for datum in dataset_test[DATA_KEY] + dataset_dev[DATA_KEY]:
                    document = to_entities(
                        datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                    document = document.lower()

                    assert document
                    for qa in datum[DOC_KEY][QAS_KEY]:
                        doc_raw = document.split()
                        question = to_entities(qa[QUERY_KEY]).lower()
                        assert question
                        qry_raw = question.split()
                        ans_raw = ""
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "dataset":
                                ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                        assert ans_raw
                        if ent_setup == "ent-anonym":
                            entity_dict = {}
                            entity_id = 0
                            lst = doc_raw + qry_raw
                            lst.append(ans_raw)
                            for word in lst:
                                if (word.startswith('@entity')) and (word not in entity_dict):
                                    entity_dict[word] = '@entity' + str(entity_id)
                                    entity_id += 1
                            qry_raw = [entity_dict[w] if w in entity_dict else w for w in qry_raw]
                            doc_raw = [entity_dict[w] if w in entity_dict else w for w in doc_raw]
                            ans_raw = entity_dict[ans_raw]
                        vocab_set |= set(qry_raw)
                        vocab_set |= set(doc_raw)
                        vocab_set.add(ans_raw)
                        # show progress
                        n += 1
                        if n % 10000 == 0:
                            print(n)

                entities = set(e for e in vocab_set if e.startswith('@entity'))
                # @placehoder, @begin and @end are included in the vocabulary list
                tokens = vocab_set.difference(entities)
                tokens.add(SYMB_BEGIN)
                tokens.add(SYMB_END)

                vocabularies = list(entities) + list(tokens)

            elif ent_setup == "no-ent":  # ignore entity markings
                # train here (remove_notfound=True|False), dev/test below
                for datum in dataset_train[DATA_KEY]:
                    document = remove_entity_marks(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                    document = document.lower()
                    assert document
                    doc_raw = document.split()
                    for qa in datum[DOC_KEY][QAS_KEY]:
                        question = remove_entity_marks(qa[QUERY_KEY]).lower()
                        assert question
                        qry_raw = question.split()
                        ans_raw = ""
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "dataset":
                                ans_raw = ans[TXT_KEY].lower()
                        assert ans_raw
                        if remove_notfound:
                            if ans_raw not in doc_raw:
                                found_umls = False
                                for ans in qa[ANS_KEY]:
                                    if ans[ORIG_KEY] == "UMLS":
                                        umls_answer = ans[TXT_KEY].lower()
                                        if umls_answer in document:
                                            found_umls = True
                                            ans_raw = umls_answer
                                if not found_umls:
                                    continue
                        vocab_set |= set(qry_raw)
                        vocab_set |= set(doc_raw)
                        vocab_set.add(ans_raw)
                        # show progress
                        n += 1
                        if n % 10000 == 0:
                            print(n)
                # treat dev/test separately to allow remove_notfound=False
                for datum in dataset_test[DATA_KEY] + dataset_dev[DATA_KEY]:
                    document = remove_entity_marks(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                    document = document.lower()
                    assert document
                    doc_raw = document.split()
                    for qa in datum[DOC_KEY][QAS_KEY]:
                        question = remove_entity_marks(qa[QUERY_KEY]).lower()
                        assert question
                        qry_raw = question.split()
                        ans_raw = ""
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "dataset":
                                ans_raw = ans[TXT_KEY].lower()
                        assert ans_raw
                        vocab_set |= set(qry_raw)
                        vocab_set |= set(doc_raw)
                        vocab_set.add(ans_raw)
                        # show progress
                        n += 1
                        if n % 10000 == 0:
                            print(n)

                entities = set(e for e in vocab_set if e.startswith('@entity'))
                # @placehoder, @begin and @end are included in the vocabulary list
                tokens = vocab_set.difference(entities)
                tokens.add(SYMB_BEGIN)
                tokens.add(SYMB_END)

                vocabularies = list(entities) + list(tokens)

            else:
                raise ValueError

            print("writing vocabularies to " + vocab_file + " ...")
            vocab_fp = codecs.open(vocab_file, "w", encoding="utf-8")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print("vocab_size = %d" % vocab_size)
        print("num characters = %d" % len(char_set))
        print("%d anonymoused entities" % num_entities)
        print("%d other tokens (including @placeholder, %s and %s)" % (
            vocab_size - num_entities, SYMB_BEGIN, SYMB_END))

        return word_dictionary, char_dictionary, num_entities

    def parse_file(self, file_path, dictionary, use_chars, ent_setup, remove_notfound):
        """
        parse a *.json dataset file into a list of questions, where each element is tuple(document, query, answer, filename, query_id)
        """
        questions = []
        w_dict, c_dict = dictionary[0], dictionary[1]
        relabeling_dicts = {}
        raw = load_json(file_path)
        for datum in raw[DATA_KEY]:
            document = to_entities(
                datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
            document = document.lower()

            assert document
            for qa in datum[DOC_KEY][QAS_KEY]:
                if ent_setup in ["ent-anonym", "ent"]:
                    doc_raw = document.split()
                    question = to_entities(qa[QUERY_KEY]).lower()
                    qry_id = qa[ID_KEY]
                    assert question
                    ans_raw = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            ans_raw = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                    assert ans_raw
                    if remove_notfound:
                        if ans_raw not in doc_raw:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                                    if umls_answer in doc_raw:
                                        found_umls = True
                                        ans_raw = umls_answer
                            if not found_umls:
                                continue
                    qry_raw = question.split()
                    if ent_setup == "ent-anonym":
                        entity_dict = {}
                        entity_id = 0
                        lst = doc_raw + qry_raw
                        lst.append(ans_raw)
                        for word in lst:
                            if (word.startswith('@entity')) and (word not in entity_dict):
                                entity_dict[word] = '@entity' + str(entity_id)
                                entity_id += 1
                        qry_raw = [entity_dict[w] if w in entity_dict else w for w in qry_raw]
                        doc_raw = [entity_dict[w] if w in entity_dict else w for w in doc_raw]
                        ans_raw = entity_dict[ans_raw]
                        inv_entity_dict = {ent_id: ent_ans for ent_ans, ent_id in entity_dict.items()}
                        assert len(entity_dict) == len(inv_entity_dict)
                        relabeling_dicts[qa[ID_KEY]] = inv_entity_dict


                    cand_e = [w for w in doc_raw if w.startswith('@entity')]
                    cand_raw = [[e] for e in cand_e]
                    # wrap the query with special symbols
                    qry_raw.insert(0, SYMB_BEGIN)
                    qry_raw.append(SYMB_END)
                    try:
                        cloze = qry_raw.index('@placeholder')
                    except ValueError:
                        print('@placeholder not found in ', qry_raw, '. Fixing...')
                        at = qry_raw.index('@')
                        qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                        cloze = qry_raw.index('@placeholder')

                    # tokens/entities --> indexes
                    doc_words = list(map(lambda w: w_dict[w], doc_raw))

                    # tokens/entities --> indexes
                    qry_words = list(map(lambda w: w_dict[w], qry_raw))
                    if use_chars:
                        qry_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                              list(w)[:MAX_WORD_LEN])), qry_raw))
                    else:
                        qry_chars = []
                    ans = list(map(lambda w: w_dict.get(w, 0), ans_raw.split()))
                    cand = [list(map(lambda w: w_dict.get(w, 0), c)) for c in cand_raw]

                    if use_chars:
                        doc_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                                      list(w)[:MAX_WORD_LEN])), doc_raw))
                    else:
                        doc_chars = []

                    questions.append((doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze, qry_id))

                elif ent_setup == "no-ent":
                    # collect candidate ents using @entity marks
                    cand_e = [w for w in to_entities(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]).lower().split() if w.startswith('@entity')]
                    cand_raw = [e[len("@entity"):].split("_") for e in cand_e]
                    document = remove_entity_marks(datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY])
                    document = document.lower()
                    doc_raw = document.split()
                    question = remove_entity_marks(qa[QUERY_KEY]).lower()
                    qry_id = qa[ID_KEY]
                    assert question
                    qry_raw = question.split()
                    ans_raw = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            ans_raw = ans[TXT_KEY].lower()
                    assert ans_raw
                    if remove_notfound:
                        if ans_raw not in doc_raw:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = ans[TXT_KEY].lower()
                                    if umls_answer in doc_raw:
                                        found_umls = True
                                        ans_raw = umls_answer
                            if not found_umls:
                                continue

                    relabeling_dicts[qa[ID_KEY]] = None
                    # wrap the query with special symbols
                    qry_raw.insert(0, SYMB_BEGIN)
                    qry_raw.append(SYMB_END)
                    try:
                        cloze = qry_raw.index('@placeholder')
                    except ValueError:
                        print('@placeholder not found in ', qry_raw, '. Fixing...')
                        at = qry_raw.index('@')
                        qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
                        cloze = qry_raw.index('@placeholder')

                    # tokens/entities --> indexes
                    doc_words = list(map(lambda w: w_dict[w], doc_raw))

                    # tokens/entities --> indexes
                    qry_words = list(map(lambda w: w_dict[w], qry_raw))
                    if use_chars:
                        qry_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                                                list(w)[:MAX_WORD_LEN])), qry_raw))
                    else:
                        qry_chars = []
                    ans = list(map(lambda w: w_dict.get(w, 0), ans_raw.split()))
                    cand = [list(map(lambda w: w_dict.get(w, 0), c)) for c in cand_raw]

                    if use_chars:
                        doc_chars = list(map(lambda w: list(map(lambda c: c_dict.get(c, c_dict[' ']),
                                                                list(w)[:MAX_WORD_LEN])), doc_raw))
                    else:
                        doc_chars = []

                    questions.append((doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze, qry_id))

                else:
                    raise ValueError

        return questions, relabeling_dicts


class DataPreprocessor:

    def preprocess(self, question_dir, no_training_set=False, use_chars=True):
        """
        preprocess all data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set is True.
        """
        vocab_f = os.path.join(question_dir,"vocab.txt")
        word_dictionary, char_dictionary, num_entities = \
                self.make_dictionary(question_dir, vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)
        if no_training_set:
            training = None
        else:
            print("preparing training data ...")
            training = self.parse_all_files(question_dir + "/training", dictionary, use_chars)
        print("preparing validation data ...")
        validation = self.parse_all_files(question_dir + "/validation", dictionary, use_chars)
        print("preparing test data ...")
        test = self.parse_all_files(question_dir + "/test", dictionary, use_chars)

        data = Data(dictionary, num_entities, training, validation, test)
        return data

    def make_dictionary(self, question_dir, vocab_file):

        if os.path.exists(vocab_file):
            print("loading vocabularies from " + vocab_file + " ...")
            vocabularies = list(map(lambda x:x.strip(), open(vocab_file).readlines()))
        else:
            print("no " + vocab_file + " found, constructing the vocabulary list ...")

            fnames = []
            fnames += glob.glob(question_dir + "/test/*.question")
            fnames += glob.glob(question_dir + "/validation/*.question")
            fnames += glob.glob(question_dir + "/training/*.question")

            vocab_set = set()
            n = 0.
            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline().split()
                fp.readline()
                query = fp.readline().split()
                fp.close()

                vocab_set |= set(document) | set(query)

                # show progress
                n += 1
                if n % 10000 == 0:
                    print('%3d%%' % int(100 * n / len(fnames)))

            entities = set(e for e in vocab_set if e.startswith('@entity'))

            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities)+list(tokens)

            print("writing vocabularies to " + vocab_file + " ...")
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print("vocab_size = %d" % vocab_size)
        print("num characters = %d" % len(char_set))
        print("%d anonymoused entities" % num_entities)
        print("%d other tokens (including @placeholder, %s and %s)" % (
            vocab_size - num_entities, SYMB_BEGIN, SYMB_END))

        return word_dictionary, char_dictionary, num_entities

    def parse_one_file(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()
        doc_raw = raw[2].split() # document
        qry_raw = raw[4].split() # query
        ans_raw = raw[6].strip() # answer
        cand_raw = map(lambda x:x.strip().split(':')[0].split(), raw[8:]) # candidate answers

        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)
        try:
            cloze = qry_raw.index('@placeholder')
        except ValueError:
            print('@placeholder not found in ', fname, '. Fixing...')
            at = qry_raw.index('@')
            qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at+2])] + qry_raw[at+2:]
            cloze = qry_raw.index('@placeholder')

        # tokens/entities --> indexes
        doc_words = list(map(lambda w:w_dict[w], doc_raw))
        qry_words = list(map(lambda w:w_dict[w], qry_raw))
        if use_chars:
            doc_chars = list(map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']),
                list(w)[:MAX_WORD_LEN]), doc_raw))
            qry_chars = list(map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']),
                list(w)[:MAX_WORD_LEN]), qry_raw))
        else:
            doc_chars, qry_chars = [], []
        ans = list(map(lambda w:w_dict.get(w,0), ans_raw.split()))
        cand = [list(map(lambda w:w_dict.get(w,0), c)) for c in cand_raw]

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze

    def parse_all_files(self, directory, dictionary, use_chars):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        all_files = glob.glob(directory + '/*.question')
        questions = [self.parse_one_file(f, dictionary, use_chars) + (os.path.basename(f)[:-len(".question")],) for f in all_files]
        return questions

    def gen_text_for_word2vec(self, question_dir, text_file):

            fnames = []
            fnames += glob.glob(question_dir + "/training/*.question")

            out = open(text_file, "w")

            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline()
                fp.readline()
                query = fp.readline()
                fp.close()
                
                out.write(document.strip())
                out.write(" ")
                out.write(query.strip())

            out.close()


def remove_entity_marks(txt):
    return txt.replace("BEG__", "").replace("__END", "")

if __name__ == '__main__':
    dp = DataPreprocessor()
    dp.gen_text_for_word2vec("cnn/questions", "/tmp/cnn_questions.txt")

