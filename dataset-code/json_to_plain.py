import argparse
import sys

from describe_data import *
from util import load_json


def remove_entity_marks(txt):
    return txt.replace("BEG__", "").replace("__END", "")


def to_entities(text, ent_marker="@entity"):
    """
    Text includes entities marked as BEG__w1 w2 w3__END. Transform to a single entity @entityw1_w2_w3.
    """
    word_list = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = [w_stripped.split("_")[2]]
            word_list.append(ent_marker + "_".join(concept))
            if inside:  # something went wrong, leave as is
                print("Inconsistent markup.")
        elif w_stripped.startswith("BEG__"):
            assert not inside
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]
        elif w_stripped.endswith("__END"):
            if not inside:
                word_list.append(w_stripped[:-5])
            else:
                concept.append(w_stripped.rsplit("_", 2)[0])
                word_list.append(ent_marker + "_".join(concept))
                inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                word_list.append(w_stripped)

    return " ".join(word_list)


def ent_to_plain(e):
    """
    :param e: "@entityLeft_hand"
    :return: "Left hand"
    """
    return " ".join(e[len("@entity"):].split("_"))


def plain_to_ent(e):
    """
    :param e: "Left hand"
    :return: "@entityLeft_hand"
    """
    return "@entity" + "_".join(e.split())


def write_gareader(i, f_out):
    """
    :param i: {"id": "",
                  "p": "",
                  "q", "",
                  "a", "",
                  "c", [""]}
    """
    with open(f_out, "w") as fh_out:
        fh_out.write(i["id"] + "\n\n")
        fh_out.write(i["p"] + "\n\n")
        fh_out.write(i["q"] + "\n\n")
        fh_out.write(i["a"] + "\n\n")
        fh_out.write("\n".join(i["c"]) + "\n")


def write_sareader(i, fh_out):
    """
        :param i: {"id": "",
                      "p": "",
                      "q", "",
                      "a", "",
                      "c", [""]}
    """
    fh_out.write(i["q"] + "\n")
    fh_out.write(plain_to_ent(i["a"]) + "\n")
    fh_out.write(i["p"] + "\n")
    fh_out.write(i["id"] + "\n\n")


def write_cnnlike(i, f_out):
    """
            :param i: {"id": "",
                          "p": "",
                          "q", "",
                          "a", "",
                          "c", [""]}
        """
    def rename_ents(txt, c_d):
        out_txt = []
        for tok in txt.split():
            if tok.startswith("@entity"):
                ent_id = c_d[tok[len("@entity"):]]
                out_txt.append("@entity{}".format(ent_id))
            else:
                out_txt.append(tok)
        return " ".join(out_txt)

    c_d = {e[len("@entity"):]: cnt for cnt, e in enumerate(set(i["c"]))}
    with open(f_out, "w") as fh_out:
        fh_out.write(i["id"] + "\n\n")
        p = rename_ents(i["p"], c_d)
        fh_out.write(p + "\n\n")
        q = rename_ents(i["q"], c_d)
        fh_out.write(q + "\n\n")
        # add answer to candidates when the answer not found in passage
        if i["a"][len("@entity"):] not in c_d:
            c_d[i["a"][len("@entity"):]] = len(c_d)
        a = rename_ents(i["a"], c_d)
        fh_out.write(a + "\n\n")
        c = ["{}:{}".format(rename_ents(cand, c_d), cand[len("@entity"):]) for cand in i["c"]]
        fh_out.write("\n".join(c) + "\n")


class JsonDataset:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.dataset = load_json(self.dataset_file)

    def json_to_plain(self, remove_notfound=False, stp="no-ent", include_q_cands=False):
        """
        :param stp: no-ent | ent; whether to mark entities in passage; if ent, a multiword entity is treated as 1 token
        :return: {"id": "",
                  "p": "",
                  "q", "",
                  "a", "",
                  "c", [""]}
        """
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                fields = {}
                qa_txt_option = (" " + qa[QUERY_KEY]) if include_q_cands else ""
                #cand = [w for w in to_entities(datum[DOC_KEY][TITLE_KEY] + " " +
                #                               datum[DOC_KEY][CONTEXT_KEY] + qa_txt_option).lower().split() if w.startswith('@entity')]
                cand = [w for w in to_entities(datum[DOC_KEY][TITLE_KEY] + " " +
                                               datum[DOC_KEY][CONTEXT_KEY]).lower().split() if w.startswith('@entity')]
                cand_q = [w for w in to_entities(qa_txt_option).lower().split() if w.startswith('@entity')]
                if stp == "no-ent":
                    c = {ent_to_plain(e) for e in set(cand)}
                    a = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            a = ans[TXT_KEY].lower()
                    if remove_notfound:
                        if a not in c:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = ans[TXT_KEY].lower()
                                    if umls_answer in c:
                                        found_umls = True
                                        a = umls_answer
                            if not found_umls:
                                continue
                    fields["c"] = list(c)
                    assert a
                    fields["a"] = a
                    document = remove_entity_marks(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace(
                        "\n", " ").lower()
                    fields["p"] = document
                    fields["q"] = remove_entity_marks(qa[QUERY_KEY]).replace("\n", " ").lower()

                elif stp == "ent":
                    c = set(cand)
                    c_q = set(cand_q)
                    a = ""
                    for ans in qa[ANS_KEY]:
                        if ans[ORIG_KEY] == "dataset":
                            a = plain_to_ent(ans[TXT_KEY].lower())
                    if remove_notfound:
                        if a not in c:
                            found_umls = False
                            for ans in qa[ANS_KEY]:
                                if ans[ORIG_KEY] == "UMLS":
                                    umls_answer = plain_to_ent(ans[TXT_KEY].lower())
                                    if umls_answer in c:
                                        found_umls = True
                                        a = umls_answer
                            if not found_umls:
                                continue
                    fields["c"] = list(c) + list(c_q)
                    assert a
                    fields["a"] = a
                    document = to_entities(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace(
                        "\n", " ").lower()
                    fields["p"] = document
                    fields["q"] = to_entities(qa[QUERY_KEY]).replace("\n", " ").lower()
                else:
                    raise NotImplementedError

                fields["id"] = qa[ID_KEY]

                yield fields


def map_to_split_name(f_dataset):
    """
    :param f_dataset: any of "dev1.0.json", "train1.0.json", "test1.0.json"
    :return: any of "training", "validation", "test"
    """
    if f_dataset[:-len("1.0.json")] == "train":
        name = "training"
    elif f_dataset[:-len("1.0.json")] == "test":
        name = "test"
    elif f_dataset[:-len("1.0.json")] == "dev":
        name = "validation"
    else:
        raise ValueError

    return name


def clicr_to_concept_txt(train_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/train1.0.json", out_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/train1.0_concepts.txt"):
    """
    Prepare a single txt file with entities marked as @ent_a_b. One sentence per line, lowercased
    """
    dataset = load_json(train_file)
    with open(out_file, "w") as out_f:
        for datum in dataset[DATA_KEY]:
            for l in datum[DOC_KEY][TITLE_KEY].split("\n"):
                if not l.strip():
                    continue
                out_f.write(to_entities(l, ent_marker="@ent_").lower()+"\n")
            for l in datum[DOC_KEY][CONTEXT_KEY].split("\n"):
                if not l.strip():
                    continue
                out_f.write(to_entities(l, ent_marker="@ent_").lower()+"\n")
            for qa in datum[DOC_KEY][QAS_KEY]:
                q = qa[QUERY_KEY]
                for a in qa[ANS_KEY]:
                    if a["origin"] == "dataset":
                        q = q.replace("@placeholder", a[TXT_KEY])
                out_f.write(to_entities(q, ent_marker="@ent_").lower()+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('-dataset_dir', default="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/")
    parser.add_argument("-out_dir", default="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_plain/")
    parser.add_argument("-stp", help="(ent | no-ent)")
    parser.add_argument("-reader", help="(gareader | sareader | cnnlike)")
    args = parser.parse_args()

    out_dir = "{}/{}/{}/".format(args.out_dir, args.stp, args.reader)
    if args.reader == "gareader":
        if not os.path.exists(out_dir + "test"):
            os.makedirs(out_dir + "test")
        if not os.path.exists(out_dir + "training"):
            os.makedirs(out_dir + "training")
        if not os.path.exists(out_dir + "validation"):
            os.makedirs(out_dir + "validation")

        for f_dataset in ["train1.0.json"]:
            d = JsonDataset(args.dataset_dir + f_dataset)
            remove_notfound = True
            for inst in d.json_to_plain(remove_notfound=remove_notfound, stp=args.stp):
                write_gareader(inst, f_out=out_dir + map_to_split_name(f_dataset) + "/" + inst["id"] + ".question")

        for f_dataset in ["dev1.0.json", "test1.0.json"]:
            d = JsonDataset(args.dataset_dir + f_dataset)
            #remove_notfound = False
            remove_notfound = False
            for inst in d.json_to_plain(remove_notfound=remove_notfound, stp=args.stp):
                write_gareader(inst, f_out=out_dir + map_to_split_name(f_dataset) + "/" + inst["id"] + ".question")

    elif args.reader == "sareader":
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for f_dataset in ["train1.0.json"]:
            d = JsonDataset(args.dataset_dir + f_dataset)
            remove_notfound = True
            with open(out_dir+map_to_split_name(f_dataset), "w") as fh_out:
                for inst in d.json_to_plain(remove_notfound=remove_notfound, stp=args.stp):
                    write_sareader(inst, fh_out=fh_out)
        for f_dataset in ["dev1.0.json", "test1.0.json"]:
            d = JsonDataset(args.dataset_dir + f_dataset)
            remove_notfound = False
            with open(out_dir+map_to_split_name(f_dataset), "w") as fh_out:
                for inst in d.json_to_plain(remove_notfound=remove_notfound, stp=args.stp):
                    write_sareader(inst, fh_out=fh_out)

    elif args.reader == "cnnlike":
        if not os.path.exists(out_dir + "test"):
            os.makedirs(out_dir + "test")
        if not os.path.exists(out_dir + "train"):
            os.makedirs(out_dir + "train")
        if not os.path.exists(out_dir + "dev"):
            os.makedirs(out_dir + "dev")
        for f_dataset in ["train1.0.json"]:
            d = JsonDataset(args.dataset_dir + f_dataset)
            remove_notfound = True
            for inst in d.json_to_plain(remove_notfound=remove_notfound, stp=args.stp, include_q_cands=True):
                write_cnnlike(inst, f_out=out_dir + f_dataset[:-len("1.0.json")] + "/" + inst["id"] + ".question")
        for f_dataset in ["dev1.0.json", "test1.0.json"]:
            d = JsonDataset(args.dataset_dir + f_dataset)
            remove_notfound = True
            for inst in d.json_to_plain(remove_notfound=remove_notfound, stp=args.stp, include_q_cands=True):
                write_cnnlike(inst, f_out=out_dir + f_dataset[:-len("1.0.json")] + "/" + inst["id"] + ".question")
