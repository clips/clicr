import argparse
import os
import re
import sys
from collections import defaultdict
from os.path import basename

from expand_answers import conn, expand
from util import get_file_list


def build_query(s_to_e_idx, s_idx, txt_case, s_to_ls, e_to_s_idx, mark=False, marker1="__", marker2="__"):
    def move_boundaries(entity, pre_q_text, post_q_text):
        if not entity.startswith(marker1):
            start_idx = entity.find(marker1)
            pre_q_text += entity[:start_idx]
            entity = entity[start_idx+len(marker1):]
        if not entity.endswith(marker2):
            end_idx = entity.find(marker2)
            post_q_text = entity[end_idx + len(marker2):] + post_q_text
            entity = entity[:end_idx]
        if entity.startswith(marker1):
            entity = entity[entity.find(marker1)+len(marker1):]
        if entity.endswith(marker2):
            entity = entity[:entity.find(marker2)]

        return entity, pre_q_text, post_q_text

    e_idx = s_to_e_idx[s_idx]
    entity = txt_case[s_idx:e_idx]

    left_newline_idx = txt_case[:s_idx][::-1].find("\n")
    start_q_id = s_idx - left_newline_idx
    pre_q_text = txt_case[start_q_id:s_idx]

    end_q_id = txt_case.find("\n", e_idx)
    post_q_text = txt_case[e_idx:end_q_id]

    # ensure @placeholder won't touch other chars or toks
    if pre_q_text:
        if not (pre_q_text.endswith(" ") or pre_q_text.endswith("\n")):
            pre_q_text += " "
    if post_q_text:
        if not (post_q_text.startswith(" ") or post_q_text.startswith("\n")):
            post_q_text = " " + post_q_text

    if mark:
        pre_q_text = mark_query_entities(pre_q_text, s_to_ls, e_to_s_idx, start_q_id, marker1, marker2)
        add_space = False
        if pre_q_text.endswith(" "):
            add_space = True
        pre_q_text_lst = pre_q_text.split()
        if pre_q_text_lst:
            if pre_q_text_lst[-1].startswith(marker1):
                pre_q_text = " ".join(pre_q_text_lst[:-1]) # move the last item to entity  # pre_q_text_lst[-1] = pre_q_text_lst[-1][len(marker1):]
                entity = pre_q_text_lst[-1][len(marker1):] + entity
        if add_space:
            pre_q_text += " "
        post_q_text = mark_query_entities(post_q_text, s_to_ls, e_to_s_idx, e_idx, marker1, marker2)
    # make answer less sparse: func word and parentheticals exclusion
    entity = exclude_fn_words(marker1+entity+marker2)
    entity, pre_q_text, post_q_text = move_boundaries(entity, pre_q_text, post_q_text)
    entity = exclude_parentheticals(marker1 + entity + marker2)
    entity, pre_q_text, post_q_text = move_boundaries(entity, pre_q_text, post_q_text)
    assert marker1 not in entity
    assert marker2 not in entity

    # exclude fn words in query
    pre_q_text = exclude_from_entities(pre_q_text, exclude_fn_words)
    post_q_text = exclude_from_entities(post_q_text, exclude_fn_words)
    # exclude parentheticals in query
    pre_q_text = exclude_from_entities(pre_q_text, exclude_parentheticals)
    post_q_text = exclude_from_entities(post_q_text, exclude_parentheticals)

    if pre_q_text and not pre_q_text.endswith(" "):
        pre_q_text += " "
    if post_q_text and not post_q_text.startswith(" "):
        post_q_text = " " + post_q_text

    q_text = pre_q_text + "@placeholder" + post_q_text
    sem_typ, ass, cui = s_to_ls[s_idx]

    return entity, sem_typ, cui, remove_citation(q_text)


def fix_marks(txt, marker1, marker2):
    """
    Clamp sometimes start the concept mid-word, e.g. at a dash. This function fixes that.
    """
    fixed_txt = []
    for s in txt.split("\n"):
        fixed_s = []
        for w in s.split(" "):
            if marker1 in w and marker2 in w:
                if not (w.startswith(marker1) and w.endswith(marker2)):
                    start_idx_m2 = w.find(marker2)
                    start_idx_m1 = w.find(marker1)
                    if start_idx_m2 > start_idx_m1:
                        w = w[:start_idx_m1] + " " + w[start_idx_m1:start_idx_m2 + len(marker2)] + " " + w[
                                                                                                         start_idx_m2 + len(
                                                                                                             marker2):]
                    else:
                        w = w[:start_idx_m2 + len(marker2)] + " " + w[start_idx_m2 + len(marker2):start_idx_m1] + " " + w[
                                                                                                                    start_idx_m1:]
            elif marker1 in w or marker2 in w:
                if marker1 in w and not w.startswith(marker1):
                    _w = w
                    if not _w.startswith(marker1):
                        start_idx = _w.find(marker1)
                        end_idx = start_idx + len(marker1)
                        w = _w[:start_idx] + " " + marker1 + _w[end_idx:]
                if marker2 in w and not w.endswith(marker2):
                    _w = w
                    if not _w.endswith(marker2):
                        start_idx = _w.find(marker2)
                        end_idx = start_idx + len(marker2)
                        w = _w[:start_idx] + marker2 + " " + _w[end_idx:]
            fixed_s.append(w)
        fixed_txt.append(fixed_s)
    fixed_txt_str = "\n".join([" ".join(s) for s in fixed_txt])

    return fixed_txt_str


def exclude_from_entities(txt, exclude_func):
    marked_txt_list = []
    for s in read_concept_text(txt):
        marked_txt_list.append(" ".join([exclude_func(w) for w in s]))
    return "\n".join(marked_txt_list)


def mark_entities(txt, s_to_ls, e_to_s_idx, marker1="__", marker2="__"):
    def format_label(label):
        """
        :param label: ('problem', 'absent', 'C2945640')
        """
        return label[0][:2].title()

    marked_txt = ""
    inside = False
    for i, char in enumerate(txt):
        if i in s_to_ls:
            if inside:
                print("Incorrect annotation, skipping.")
                continue
            new_char = "{0}{1}".format(marker1, char)
            inside = True
        elif i in e_to_s_idx:
            if not inside:
                print("Incorrect annotation, skipping.")
                continue
            # new_char = "{0}{2}{1}".format(marker2, char, format_label(s_to_ls[e_to_s_idx[i]]))
            new_char = "{0}{1}".format(marker2, char)
            inside = False
        else:
            new_char = char

        marked_txt += new_char

    if marker1 and marker2:
        # fix inconsistent markup
        marked_txt = fix_marks(marked_txt, marker1, marker2)
        # make annotated entities less sparse: func word and parentheticals exclusion
        marked_txt = exclude_from_entities(marked_txt, exclude_fn_words)
        marked_txt = exclude_from_entities(marked_txt, exclude_parentheticals)

    return marked_txt


def mark_query_entities(txt, s_to_ls, e_to_s_idx, s_offset, marker1="__", marker2="__"):
    def format_label(label):
        """
        :param label: ('problem', 'absent', 'C2945640')
        """
        return label[0][:2].title()


    marked_txt = ""

    for i, char in enumerate(txt):
        if i + s_offset in s_to_ls:
            new_char = "{0}{1}".format(marker1, char)
        elif i + s_offset in e_to_s_idx and i != 0:
            # new_char = "{0}{2}{1}".format(marker2, char, format_label(s_to_ls[e_to_s_idx[i]]))
            new_char = "{0}{1}".format(marker2, char)
        else:
            new_char = char

        marked_txt += new_char

    if marker1 and marker2:
        marked_txt = fix_marks(marked_txt, marker1, marker2)
    return marked_txt


def build_queries(fn_case, fn_proc, marker1, marker2, mark_query_concepts=False):
    queries = []
    # get start to end index map and start idx to label map
    s_to_e_idx, s_to_ls = get_idx_maps(fn_proc)
    with open(fn_case) as fh_case:
        txt_case = fh_case.read()
    # find end of the learning point title
    start_lp = txt_case.lower().find("\nlearning point")
    end_lp = txt_case.find("\n", start_lp)

    # mark document with concepts
    e_to_s_idx = {v: k for k, v in s_to_e_idx.items()}
    assert len(s_to_e_idx) == len(e_to_s_idx)
    marked_txt = mark_entities(txt_case[:start_lp], s_to_ls, e_to_s_idx, marker1=marker1, marker2=marker2)
    marked_txt = remove_citation(marked_txt)
    # build queries from learning points
    for s_idx, ls in sorted(s_to_ls.items()):
        # text occurring before learning points
        if s_idx <= end_lp + 1:
            continue
        else:
            queries.append(
                build_query(s_to_e_idx, s_idx, txt_case, s_to_ls, e_to_s_idx, mark=mark_query_concepts, marker1=marker1,
                            marker2=marker2))

    return queries, marked_txt


def read_concept_text(text):
    concept_text_list = []
    for s in text.split("\n"):
        concept_sent_list = []
        inside = False
        for w in s.split():
            if w.startswith("BEG__") and w.endswith("__END"):
                concept_sent_list.append(w)
            elif w.startswith("BEG__"):
                inside = True
                concept = [w]
            elif w.endswith("__END"):
                concept.append(w)
                concept_sent_list.append(" ".join(concept))
                inside = False
            else:
                if inside:
                    concept.append(w)
                else:
                    concept_sent_list.append(w)
        concept_text_list.append(concept_sent_list)
    return concept_text_list


def refine_concepts(dataset_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/train1.0.json"):
    from util import load_json
    from describe_data import DATA_KEY, DOC_KEY, TITLE_KEY, CONTEXT_KEY
    dataset = load_json(dataset_file)
    data = dataset[DATA_KEY]
    for datum in data:
        title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        concept_text_lst = [exclude_parentheticals(exclude_fn_words(word)) for word in read_concept_text(title_and_passage)]
        for c in concept_text_lst:
            print(c)


def exclude_fn_words(txt):
    txt_lst = txt.split()
    if txt_lst[0].startswith("BEG__"):
        assert txt_lst[-1].endswith("__END")
        if txt_lst[0].split("_")[-1].lower() in {"a", "an", "the", "her", "his", "their", "this", "that", "these", "those", "its", "our", "any", "every", "some"}:
            new_txt = "{} BEG__{}".format(txt_lst[0].split("_")[-1], " ".join(txt_lst[1:]))
        else:
            new_txt = txt
    else:
        new_txt = txt
    return new_txt


def exclude_parentheticals(txt):
    if txt.startswith("BEG__"):
        assert txt.endswith("__END")
        txt = txt.replace("BEG__", "BEG__ ").replace("__END", " __END")
        txt_lst = txt.split()
        if txt_lst.count("(") > 1 or txt_lst.count(")") > 1:
            new_txt = " ".join(txt_lst).replace("BEG__ ", "BEG__").replace(" __END", "__END")  # leave as is, avoid messiness
        elif "(" in txt_lst and ")" in txt_lst:
            if txt_lst[-2] == ")":
                """
                B a b ( c ) E -> B a b E ( c )
                """
                start_idx = txt_lst.index("(")
                new_txt_lst = txt_lst[:start_idx] + [txt_lst[-1]] + txt_lst[start_idx:-1]
                new_txt = " ".join(new_txt_lst).replace("BEG__ ", "BEG__").replace(" __END", "__END")
            else:
                """
                B a b ( c ) d E -> same
                """
                new_txt = " ".join(txt_lst).replace("BEG__ ", "BEG__").replace(" __END", "__END")
        elif "(" in txt_lst:
            """
            B a b ( c E -> B a b E ( c
            """
            start_idx = txt_lst.index("(")
            new_txt_lst = txt_lst[:start_idx] + [txt_lst[-1]] + txt_lst[start_idx:-1]
            new_txt = " ".join(new_txt_lst).replace("BEG__ ", "BEG__").replace(" __END", "__END")

        else:
            new_txt = txt.replace("BEG__ ", "BEG__").replace(" __END", "__END")
    else:
        new_txt = txt
    return new_txt


def get_idx_maps(fn_proc):
    start_idx_to_end_idx = {}
    start_idx_to_labels = {}
    with open(fn_proc) as fh_proc:
        for l in fh_proc:
            if l.strip().startswith("NamedEntity"):
                cui = ""
                try:
                    typ, start_idx, end_idx, sem_typ, ass, cui, ident = l.strip().split("\t")
                    assert "cui=" in cui
                    cui = cui[4:]
                except ValueError:
                    typ, start_idx, end_idx, sem_typ, ass, ident = l.strip().split("\t")

                assert "semantic=" in sem_typ
                assert "assertion=" in ass
                assert "ne=" in ident

                start_idx = int(start_idx)
                end_idx = int(end_idx)
                sem_typ = sem_typ.split("=")[-1]
                ass = ass.split("=")[-1]
                ident = ident.split("=")[-1]

                start_idx_to_end_idx[start_idx] = end_idx
                start_idx_to_labels[start_idx] = (sem_typ, ass, cui)

    return start_idx_to_end_idx, start_idx_to_labels


def get_nes(f):
    """get named entities"""
    nes = []
    with open(f) as infile_proc:
        for l in infile_proc:
            if l.strip().startswith("NamedEntity"):
                sem_typ = l.strip().split("\t")[3]
                assert sem_typ.startswith("semantic=")
                sem_typ = sem_typ.split("=")[-1]
                ne = l.strip().split("\t")[-1]
                assert ne.startswith("ne=")
                ne = ne.split("=")[-1]

                nes.append((ne, sem_typ))

    return nes


def mark_query(q):
    marker1 = "\033[01;32m"
    marker2 = "\033[0m"
    # marker1 = ""
    # marker2 = ""
    s_id = q.find("@placeholder")
    e_id = s_id + len("@placeholder")

    return q[:s_id] + marker1 + q[s_id:e_id] + marker2 + q[e_id:]


def mark_answer(a):
    marker1 = "\033[01;32m"
    marker2 = "\033[0m"
    # marker1 = ""
    # marker2 = ""

    return marker1 + a + marker2


def remove_citation(txt):
    """
    Remove citation numbers, which interfere with line breaks, e.g.
    "the most reliable method for treating complications related to PV .13 , 17 "
    or
    "1 , 3 The late onset " when occurring in the beginning of the line. Note that
    in these cases we only remove citations when there are two or more. For one, it is
    too risky, because it would remove also some cases of measurments ("1 g per day") etc.

    Should really have been done before concept extraction, but this is an easy fix.
    """
    try:
        fixed_back = re.sub(r" \.[0-9]+\s(, [0-9]+\s)*", " .\n", txt)
        fixed_front = re.sub(r"^[0-9]+ (, [0-9]+ )+([A-Z])", r"\2", fixed_back, flags=re.MULTILINE)
    except TypeError:
        print()

    return fixed_front


def format_case_txt(txt_case, q, fn_case, answer_type="entity", umls_cur=None):
    ext_id = fn_case.find(".full.struct.tok")
    source = os.path.basename(fn_case[:ext_id])

    if answer_type == "entity":
        a = q[0]
    elif answer_type == "semtype":
        a = q[1]
    elif answer_type == "cui":
        a = q[2]
    elif answer_type == "expanded":
        cui = q[2]
        a = q[0]
        expanded_set = expand(cui, umls_cur)
        expanded_str = ""
        if a in expanded_set:
            expanded_set.remove(a)  # to separate original answer from the expanded set; both written to output
        for expanded_a in expanded_set:
            expanded_str += "\t" + mark_answer(expanded_a)
    else:
        sys.exit("Unkown answer type during formatting")

    instance = "{0}\n\n{1}\n{2}\n\n{3}{4}\n".format(source,
                                                    remove_citation(txt_case),
                                                    remove_citation(mark_query(q[3])),  # query text
                                                    mark_answer(a),
                                                    expanded_str if answer_type == "expanded" else ""
                                                    )
    # return "source:{0}\n\n{1}\nquery:{2}\n@placeholder:{3}\n".format(source, txt_case, q[3], q[0]), source
    return instance, source


def main_txt():
    label_freq = defaultdict(int)
    n_queries = 0
    n_cases = 0
    marker1 = "\033[01;31m"
    marker2 = "\033[0m"
    # marker1 = ""
    # marker2 = ""
    answer_types = ["entity", "semtype", "cui", "expanded"]
    a_type = answer_types[3]
    if a_type == "expanded":
        umls_cur = conn().cursor()
    else:
        umls_cur = None

    print("Using {} answer type".format(a_type))

    for fn_case in get_file_list(args.dir_cases):
        n_cases += 1

        fn_proc = args.dir_cases_concepts + basename(fn_case) + ".txt"
        if not os.path.isfile(fn_proc):
            print("Does not exist:" + fn_proc)
            continue
        queries, txt_case = build_queries(fn_case, fn_proc, marker1=marker1, marker2=marker2)
        n_queries += len(queries)

        for q in queries:
            label_freq[q[1]] += 1
            yield format_case_txt(txt_case, q, fn_case, answer_type=a_type, umls_cur=umls_cur)

        # if n_cases == 1:
        #    break
        if n_cases % 1000 == 0:
            print(n_cases)

    print("Number of cases: {}".format(n_cases))
    print("Number of queries: {}".format(n_queries))


def build_txt(dir_output):
    # keep track of number of queries per case source
    source_count = defaultdict(int)

    for case, source in main_txt():
        source_count[source] += 1
        source_num = source_count[source]

        if args.dir_output is not None:
            if not os.path.exists(dir_output):
                os.makedirs(dir_output)
            write(case, "{}.{}".format(source, source_num), dir_output, ext=".col")
        else:
            print(case)


def write(case, source, outdir, ext=""):
    f_out = open("{0}/{1}{2}".format(outdir, source, ext), "w")
    f_out.write(case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # d = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports/data_out_tok"
    parser.add_argument("-dir_cases", help="Path to directory containing preprocessed case files.", required=True)
    parser.add_argument("-dir_cases_concepts",
                        help="Path to directory containing case files with concepts extracted by Clamp.", required=True)
    parser.add_argument("-dir_output")
    args = parser.parse_args()

    build_txt(args.dir_output)
