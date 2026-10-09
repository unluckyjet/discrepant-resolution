"""Run in an isolated env with text-extensions-for-pandas (pandas<2): export tp's token offsets for eng.testb.

Reiss et al. (2020) give their corrections as character spans in tp's document model, so we use tp itself
to map spans to tokens.
"""
import sys
import text_extensions_for_pandas as tp

src, out = sys.argv[1], sys.argv[2]
docs = tp.io.conll.conll_2003_to_dataframes(src, ["pos", "phrase", "ent"], [False, True, True])
rows = []
for d, df in enumerate(docs):
    spans = df["span"].values
    for t in range(len(df)):
        rows.append((d, t, int(df["line_num"].iloc[t]), int(spans.begin[t]), int(spans.end[t]),
                     df["span"].iloc[t].covered_text, df["ent_iob"].iloc[t], df["ent_type"].iloc[t] or ""))
import csv
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["doc", "tok", "line_num", "begin", "end", "text", "iob", "type"]); w.writerows(rows)
print(len(docs), "documents,", len(rows), "tokens")
