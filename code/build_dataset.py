"""Join MMLU-Redux 2.0 reference labels with HELM v1.3.0 per-instance predictions.

Output: data/derived/redux_helm.parquet, one row per Redux item, with
  subject, qid, L (MMLU label index), error_type, Y (reference label index or -1),
  and one column per model holding the predicted option index (0-3, or -1 if unparseable / missing).
"""
import json, re, glob
from pathlib import Path
import pyarrow as pa
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/derived"); OUT.mkdir(parents=True, exist_ok=True)
LETTERS = "ABCD"

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def load_redux():
    rows = []
    for f in sorted(glob.glob(str(RAW / "mmlu-redux-2.0/*/data-*.arrow"))):
        with pa.memory_map(f) as src:
            t = pa.ipc.open_stream(src).read_all()
        sub = f.split("/")[-2]
        for i, r in enumerate(t.to_pylist()):
            r["subject"], r["qid"] = sub, f"{sub}/{i}"
            rows.append(r)
    return pd.DataFrame(rows)

def parse_answer(x):
    """Redux 'correct_answer' is stored inconsistently: '2', 2.0, 'C', None, free text."""
    if x is None:
        return None
    s = str(x).strip()
    if re.fullmatch(r"[0-3](\.0)?", s):
        return int(float(s))
    if re.fullmatch(r"[A-Da-d]", s):
        return LETTERS.index(s.upper())
    return None

def reference_label(r):
    if r.error_type == "ok":
        return r.answer, "ok"
    if r.error_type == "wrong_groundtruth":
        a = parse_answer(r.correct_answer)
        if a is None:
            return -1, "wrong_gt_unparsed"
        if a == r.answer:
            return -1, "wrong_gt_same_as_label"
        return a, "wrong_gt"
    return -1, "excluded:" + r.error_type

def key(question, choices):
    return norm(question) + " || " + " | ".join(norm(c) for c in choices)

redux = load_redux()
ref = redux.apply(reference_label, axis=1, result_type="expand")
redux["Y"], redux["ref_status"] = ref[0].astype(int), ref[1]
redux["L"] = redux["answer"].astype(int)
redux["k"] = [key(q, c) for q, c in zip(redux.question, redux.choices)]

models = sorted(p.name for p in (RAW / "helm/pred").iterdir())
preds = {m: {} for m in models}
consistency = {m: [0, 0] for m in models}
for m in models:
    for pf in (RAW / "helm/pred" / m).glob("*.json"):
        inst = {d["id"]: d for d in json.loads((RAW / "helm/inst" / m / pf.name).read_text())}
        for p in json.loads(pf.read_text()):
            d = inst[p["instance_id"]]
            refs = [x["output"]["text"] for x in d["references"]]
            gold = [i for i, x in enumerate(d["references"]) if "correct" in x["tags"]][0]
            # a bare option letter (optionally followed by punctuation); anything else is invalid,
            # which HELM scores as wrong
            mt = re.match(r"\s*\(?([A-D])\)?(?![A-Za-z])", p["predicted_text"])
            idx = LETTERS.index(mt.group(1)) if mt else -1
            # sanity: HELM exact_match must equal (predicted index == HELM gold index)
            consistency[m][0] += int((idx == gold) == bool(p["stats"]["exact_match"]))
            consistency[m][1] += 1
            preds[m][key(d["input"]["text"], refs)] = (idx, gold)

bad = {m: c for m, c in consistency.items() if c[0] != c[1]}
print("models:", len(models), "exact_match consistency failures:", bad)

for m in models:
    redux[m] = [preds[m].get(k, (-2, None))[0] for k in redux.k]
    # HELM gold must equal the Redux copy of the MMLU label
    golds = [preds[m].get(k, (None, None))[1] for k in redux.k]
    redux["helm_gold_" + m] = golds

gold_cols = ["helm_gold_" + m for m in models]
redux["helm_gold"] = redux[gold_cols].bfill(axis=1).iloc[:, 0]
redux = redux.drop(columns=gold_cols)
redux.loc[(redux[models] == -2).any(axis=1), "ref_status"] = "excluded:no_helm_match"
redux.loc[redux.helm_gold.notna() & (redux.helm_gold != redux.L), "ref_status"] = "excluded:label_differs_from_helm"
missing = (redux[models] == -2).sum()
print("items missing per model (min/max):", missing.min(), missing.max())
print(redux.ref_status.value_counts())
cols = ["qid", "subject", "L", "Y", "error_type", "ref_status"] + models
redux[cols].to_parquet(OUT / "redux_helm.parquet")
print("wrote", len(redux))

stats = {"ref_status": redux.ref_status.value_counts().to_dict(),
         "helm_scoring_disagreements": {m: c[1] - c[0] for m, c in consistency.items()},
         "helm_instances_per_model": {m: c[1] for m, c in consistency.items()},
         "invalid_outputs_in_analysis_set": {m: int((redux.loc[redux.ref_status.isin(["ok", "wrong_gt"]), m] == -1).sum()) for m in models}}
(OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
