"""Table of adjudication designs in published relabelling studies, from data/lit/design_coding.csv."""
import pandas as pd
from pathlib import Path

d = pd.read_csv("data/lit/design_coding.csv")
d = d[d.key != "klie2023annotation"]          # survey, not a relabelling study
BENCH = {
    "northcutt2021pervasive": "10 test sets (ImageNet, CIFAR, \\dots)", "nahum2025are": "TRUE, SummEval",
    "vasudevan2022when": "ImageNet (multi-label)", "beyer2020are": "ImageNet (ReaL)", "shankar2020evaluating": "ImageNet",
    "gema2025are": "MMLU (Redux)", "vendrow2025do": "15 subsets (GSM8K, SQuAD2.0, \\dots)", "wang2024mmlupro": "MMLU-Pro",
    "chong2022detecting": "IMDB, Amazon, \\dots", "wang2019crossweigh": "CoNLL-2003", "reiss2020identifying": "CoNLL-2003",
    "alt2020tacred": "TACRED", "stoica2021retacred": "TACRED", "rucker2023cleanconll": "CoNLL-2003",
    "zhai2026hleverified": "Humanity's Last Exam", "ye2026scalable": "MedCalc-Bench", "land2026auditing": "32 subsets (RewardBench 2, \\dots)",
    "ansari2026how": "4 physics benchmarks", "brunello2026fixing": "FOLIO, MALLS", "sylvestre2026gold": "SciFact",
    "aleithan2024swebench": "SWE-bench", "chowdhury2024introducing": "SWE-bench (Verified)",
}
ORDER = {"DR": 0, "DR-multi": 1, "CRS": 2, "OTHER": 3, "RANDOM": 4, "FULL": 5}
d = d.assign(o=d.design.map(ORDER)).sort_values(["o", "year"])

def yn(x):
    s = str(x).strip().lower()
    if s.startswith("yes"): return "yes"
    if s.startswith("partial"): return "partial"
    if s.startswith("no"): return "no"
    return "--"

rows = []
for r in d.itertuples():
    rows.append(f"\\citet{{{r.key}}} & {BENCH[r.key]} & {r.design} & {yn(r.flagger_also_evaluated)} & "
                f"{yn(r.rereports_accuracy_or_ranking)} & {yn(r.acknowledges_unflagged_errors) if r.design not in ('FULL',) else '--'} \\\\")
Path("paper/tables/tab_literature.tex").write_text(
    "\\begin{tabular}{llllll}\n\\toprule\n"
    "Study & Benchmark & Design & Flagger & Re-reports & Notes unflagged \\\\\n"
    " & & & evaluated & acc./ranking & may be wrong \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
c = d.design.value_counts().to_dict()
print(c, len(d))
dr = d[d.design.isin(["DR", "DR-multi"])]
print("DR/DR-multi:", len(dr), "flagger evaluated yes:", (dr.flagger_also_evaluated.str.lower().str.startswith("yes")).sum(),
      "rereports yes:", (dr.rereports_accuracy_or_ranking.str.lower().str.startswith("yes")).sum(),
      "ack yes:", (dr.acknowledges_unflagged_errors.str.lower().str.startswith("yes")).sum())
