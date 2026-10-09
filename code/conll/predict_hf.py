"""Word-level NER predictions of public Hugging Face CoNLL-2003 models on eng.testb (first-subword label).

Output: data/derived/conll_pred_<name>.txt with one entity type (O/PER/LOC/ORG/MISC) per eng.testb line,
blank for blank / -DOCSTART- lines. Weights are loaded from safetensors only.
"""
import sys, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

repo, name = sys.argv[1], sys.argv[2]
lines = Path("data/raw/conll/eng.testb").read_text().splitlines()
sents, cur = [], []
for i, ln in enumerate(lines):
    p = ln.split()
    if not p or p[0] == "-DOCSTART-":
        if cur: sents.append(cur); cur = []
        continue
    cur.append((i, p[0]))
if cur: sents.append(cur)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(repo, add_prefix_space=True) if "roberta" in repo.lower() else AutoTokenizer.from_pretrained(repo)
model = AutoModelForTokenClassification.from_pretrained(repo, use_safetensors=True).to(dev).eval()
id2label = model.config.id2label

def to_type(lab):
    lab = lab.upper()
    if lab == "O": return "O"
    t = re.split(r"[-_]", lab)[-1]
    return {"PER": "PER", "LOC": "LOC", "ORG": "ORG", "MISC": "MISC"}.get(t, "O")

out = [""] * len(lines)
B = 32
with torch.no_grad():
    for s in range(0, len(sents), B):
        batch = sents[s:s + B]
        words = [[w for _, w in sent] for sent in batch]
        enc = tok(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
        logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.argmax(-1).cpu()
        for b, sent in enumerate(batch):
            wid = enc.word_ids(b); seen = set()
            for pos, w in enumerate(wid):
                if w is None or w in seen: continue
                seen.add(w); out[sent[w][0]] = to_type(id2label[int(logits[b, pos])])
            for w, (li, _) in enumerate(sent):
                if out[li] == "": out[li] = "O"          # truncated words (none expected)
Path("data/derived").mkdir(exist_ok=True)
Path(f"data/derived/conll_pred_{name}.txt").write_text("\n".join(out) + "\n")
print(name, "labels:", sorted(set(id2label.values())))
