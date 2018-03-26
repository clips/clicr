import json
import os
import random
import sys
import xml.etree.ElementTree as ET

import numpy as np


KNOWN_TOKEN_TYPES = {
    "org.apache.ctakes.typesystem.type.syntax.NewlineToken",
    "org.apache.ctakes.typesystem.type.syntax.PunctuationToken",
    "org.apache.ctakes.typesystem.type.syntax.WordToken",
    "org.apache.ctakes.typesystem.type.syntax.NumToken",
    "org.apache.ctakes.typesystem.type.syntax.SymbolToken",
    "org.apache.ctakes.typesystem.type.syntax.ContractionToken"
    }

random.seed(1234)


def line_reader(f, skip=0):
    with open(f) as in_f:
        for c, l in enumerate(in_f, 1):
            if c <= skip:
                continue
            yield l


def cosines(W, W2):
    if W2.ndim == 2:
        scores = []
        for w_emb in W2:
            scores.append(cosines(W, w_emb))
        return np.array(scores)
    w_emb_norm = np.linalg.norm(W2)
    return np.dot(W, W2) / (np.linalg.norm(W, axis=1) * w_emb_norm)


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


def ctakes_to_tok(fh, f_xml):

    text = fh.read()

    tree = ET.parse(f_xml)
    root = tree.getroot()
    start_idx, end_idx, sent_start_idx, sent_end_idx = set(), set(), set(), set()
    for child in root:
        if child.tag.endswith("Token") and child.tag.startswith("org.apache.ctakes.typesystem.type.syntax."):
            if child.tag not in KNOWN_TOKEN_TYPES:
                print("Unknown token type: " + child.tag)
            start_idx.add(int(child.attrib["begin"]))
            end_idx.add(int(child.attrib["end"]))
        elif child.tag == "org.apache.ctakes.typesystem.type.textspan.Sentence":
            sent_start_idx.add(int(child.attrib["begin"]))
            sent_end_idx.add(int(child.attrib["end"]))

    toks = ""
    tok = ""
    prev_tok = ""
    for i, c in enumerate(text):
        if i in end_idx:
            if not toks or toks.endswith("\n") or tok == "\n" or prev_tok == "\n":
                toks += tok
            else:
                toks += " " + tok
            prev_tok = tok
            tok = ""

        if i in sent_end_idx and not toks.rstrip().endswith(")") and not toks.rstrip().endswith(":"):
            toks += "\n"  # sentence boundary
            
        if i in start_idx:  # new tok
            if c == "\n" and prev_tok == "\n":
                continue
            tok += c
        else:
            if tok:
                tok += c
            else:
                continue

    return toks


def ctakes_to_tok_batch():
    with open("~/Apps/apache-ctakes-3.2.2/data_in/bcr.01.2009.1411.full.struct.2") as fh:
        f_xml = "~/Apps/apache-ctakes-3.2.2/data_out/bcr.01.2009.1411.full.struct.2.xml"
        print(ctakes_to_tok(fh, f_xml))


def remove_section_markers(fh):
    """
    Remove <h> and <p> markers at the beginning of lines.
    """
    text = ""
    for l in fh:
        if l.startswith("<h>") or l.startswith("<p>"):
            text += l[3:].strip() + "\n\n"
        else:
            if not l.strip():
                continue
            sys.exit("Unknown line type.")

    return text


def ansi_to_tex(f):
    with open(f) as infile, open(f+".tex", "w") as outfile:
        txt = infile.read()
        new_txt = txt. \
            replace("$", r"\$"). \
            replace("%", r"\%").\
            replace("&", r"\&").\
            replace("°", r"$^\circ$").\
            replace("×", r"$\times$").\
            replace("β", r"$\beta$").\
            replace("α", r"$\alpha$"). \
            replace("μ", r"$\mu$"). \
            replace("µ", r"$\mu$"). \
            replace("λ", r"$\lambda$"). \
            replace("⩾", r"$\geq$"). \
            replace("⩾", r"$\geq$"). \
            replace("χ", r"$\chi$"). \
            replace("≥", r"$\geq$"). \
            replace("{", r"\{"). \
            replace("}", r"\}"). \
            replace("#", r"\#"). \
            replace("Ι", r"I"). \
            replace("κ", r"$\kappa$"). \
            replace("δ", r"$\delta$"). \
            replace("γ", r"$\gamma$"). \
            replace("⇓", r"$\downarrow$"). \
            replace("⇑", r"$\uparrow$"). \
            replace("½", r"$1/2$"). \
            replace("−", r"$-$").\
            replace("+", r"$+$"). \
            replace("▶", r"$\rightarrow$"). \
            replace("▸", r"$\rightarrow$"). \
            replace("∼", r"$\sim$"). \
            replace("→", r"$\rightarrow$"). \
            replace("±", r"$\pm$"). \
            replace("τ", r"$\tau$"). \
            replace("″", r"\"").\
            replace("\x1b[01;31m", r"\textbf{ ").\
            replace("\x1b[01;32m", r"\textbf{ "). \
            replace("BEG__", r"\textbf{ "). \
            replace("__END", r"} ").\
            replace("\x1b[0m", "} ")

        outfile.write(r"""\documentclass[a4paper,10pt]{article}
        \usepackage[english]{babel}
        \usepackage[utf8]{inputenc}
        \usepackage{fullpage}
        \usepackage{microtype}

        \begin{document}
        """)
        outfile.write(new_txt)
        outfile.write("""
        \end{document}
        """)


def ansi_files_to_tex(files):
    for f in files:
        ansi_to_tex(f)


def ansi_dir_to_tex(dir):
    files = get_file_list(dir)
    ansi_files_to_tex(files)


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)


def random_instance_from_list(lst):
    if not lst:
        instance = ""
    else:
        instance = lst[random.randint(0, len(lst) - 1)]
    return instance


if __name__ == "__main__":
    d = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports/BMJ_Case_Reports_Structured/"
