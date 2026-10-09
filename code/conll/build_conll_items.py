"""Re-analysis of a published DR audit: Reiss et al. (2020) on the CoNLL-2003 test fold, against the independent
full re-check CoNLL++ (Wang et al. 2019). Item = token; label = entity type (O, PER, LOC, ORG, MISC).

L = original label, A = Reiss's resolved label (label corrections only), Y = CoNLL++ label (reference).
Adjudicated set Delta = tokens covered by any span Reiss reviewed (model-flagged or incidental) or corrected.
Tokens covered by Token/Sentence corrections (which change tokenization) are excluded from the item set.
This build step needs the licensed corpus (data/raw/conll). It writes text-free derived files:
data/derived/conll_items.csv (eng.testb line, document and token index, labels L/A/Y as entity types,
adjudication flags) and data/derived/conll_preds.csv (each model's predicted entity type per item).
"""
import json, re, sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parents[1]))
from drcore import sensitivity_bounds, sensitivity_bounds_diff

R = Path("data/raw/conll")
TYPES = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3, "MISC": 4}


def tag_type(tag):
    return "O" if tag in ("O", "") else tag.split("-")[-1]


def read_tags(path, last_col=True):
    out = []
    for ln in Path(path).read_text(encoding="utf-8", errors="strict").splitlines():
        p = ln.split()
        out.append(None if not p or p[0] == "-DOCSTART-" else (p[-1] if last_col else p[0]))
    return out


lines = Path(R / "eng.testb").read_text().splitlines()
is_tok = [bool(ln.split()) and ln.split()[0] != "-DOCSTART-" for ln in lines]
words = [ln.split()[0] if t else None for ln, t in zip(lines, is_tok)]
orig = read_tags(R / "eng.testb")
reiss = read_tags(R / "reiss_label_corrected_test.txt")
cpp = read_tags(R / "conllpp_test.txt")
for name, seq in [("reiss", reiss), ("conll++", cpp)]:
    seq += [None] * (len(lines) - len(seq))
    assert all((a is None) == (not t) for a, t in zip(seq, is_tok)), f"{name} not line-aligned"
cpp_words = read_tags(R / "conllpp_test.txt", last_col=False) + [None]
assert all(w == c for w, c in zip(words, cpp_words) if w), "CoNLL++ words differ from eng.testb"

# ---------------------------------------------------------------- Reiss's reviewed set, mapped to lines
tok = pd.read_csv(R / "tp_test_tokens.csv", keep_default_na=False)
tok = tok[tok.text != "-DOCSTART-"]
span_re = re.compile(r"\[(\d+),\s*(\d+)\)")


def lines_of(span, doc):
    m = span_re.match(str(span))
    if not m:
        return []
    b, e = int(m.group(1)), int(m.group(2))
    t = tok[(tok.doc == doc) & (tok.begin >= b) & (tok.end <= e)]
    return t.line_num.tolist()


def rd(f):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(f, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            pass


flagged, incidental, excluded = set(), set(), set()
hl = R / "reiss/corrected_labels/human_labels_audited"
n_rows = dict(flagged=0, incidental=0)
for f in ["CoNLL_2_in_gold", "CoNLL_2_not_in_gold", "CoNLL_3_in_gold", "CoNLL_3_not_in_gold", "CoNLL_4_in_gold", "CoNLL_4_not_in_gold"]:
    d = rd(hl / f"{f}.csv")
    d = d[(d.fold == "test") & (d.error_type.str.strip() != "")]
    for r in d.itertuples():
        ls = lines_of(r.corpus_span, int(r.doc_offset)) + lines_of(r.correct_span, int(r.doc_offset))
        if r.error_type.strip() in ("Token", "Sentence"):
            excluded.update(ls)
        target = flagged if str(r.num_models).strip() != "" else incidental
        target.update(ls)
        n_rows["flagged" if str(r.num_models).strip() != "" else "incidental"] += 1
corr = rd(R / "reiss/corrected_labels/all_conll_corrections_combined.csv")
corr = corr[corr.fold == "test"]
corr_lines = set()
for r in corr.itertuples():
    ls = lines_of(r.corpus_span, int(r.doc_offset)) + lines_of(r.correct_span, int(r.doc_offset))
    corr_lines.update(ls)
    if r.error_type in ("Token", "Sentence"):
        excluded.update(ls)

# ---------------------------------------------------------------- item table
items = [i for i, t in enumerate(is_tok) if t and i not in excluded]
idx = np.array(items)
L = np.array([TYPES[tag_type(orig[i])] for i in items])
A = np.array([TYPES[tag_type(reiss[i])] for i in items])
Y = np.array([TYPES[tag_type(cpp[i])] for i in items])
adj_lines = flagged | incidental | corr_lines
Delta = np.array([i in adj_lines for i in items])
Flag = np.array([i in flagged for i in items])
changed_outside = int(((A != L) & ~Delta).sum())

models = {}
for f in sorted((R / "entrants").glob("*.testb")):
    seq = read_tags(f, last_col=False)          # entrant files: one tag per line
    models["entrant:" + f.stem] = np.array([TYPES[tag_type(seq[i])] for i in items])
for f in sorted(Path("data/derived").glob("conll_pred_*.txt")):
    seq = f.read_text().splitlines()
    models["hf:" + f.stem.replace("conll_pred_", "")] = np.array([TYPES[seq[i] or "O"] for i in items])

INV = {v: k for k, v in TYPES.items()}
docpos = tok.set_index("line_num")[["doc", "tok"]]
items_df = pd.DataFrame({"line": idx, "doc": docpos.loc[idx, "doc"].values, "tok": docpos.loc[idx, "tok"].values,
                         "L": [INV[x] for x in L], "A": [INV[x] for x in A], "Y": [INV[x] for x in Y],
                         "adjudicated": Delta.astype(int), "flagged": Flag.astype(int)})
Path("data/derived").mkdir(exist_ok=True)
items_df.to_csv("data/derived/conll_items.csv", index=False)
pd.DataFrame({m: [INV[x] for x in M] for m, M in models.items()}).to_csv("data/derived/conll_preds.csv", index=False)
meta = dict(n_excluded=len(excluded & {i for i, t in enumerate(is_tok) if t}), n_rows=n_rows,
            changed_outside_delta=int(((A != L) & ~Delta).sum()))
Path("data/derived/conll_build_meta.json").write_text(json.dumps(meta, indent=1))
print("wrote", len(items_df), "items and", len(models), "models (no corpus text)")
