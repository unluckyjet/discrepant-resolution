"""LaTeX tables (paper/tables/*.tex) and number macros (paper/numbers.tex) from results/*.json."""
import json, statistics as st, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from names import short

R = Path("results"); T = Path("paper/tables"); T.mkdir(parents=True, exist_ok=True)
load = lambda n: json.loads((R / f"{n}.json").read_text())
SCEN_B = "B:mistralai_mixtral-8x7b-32kseqlen"
LAB = {"B:mistralai_mixtral-8x7b-32kseqlen": "Mixtral 8x7B", "B:meta_llama-2-70b": "Llama 2 70B",
       "B:openai_gpt-3.5-turbo-0613": "GPT-3.5 Turbo", "B:qwen_qwen1.5-72b": "Qwen1.5 72B"}
macros = {}


def mac(name, val):
    macros[name] = val


def pp(x, d=2):
    return f"{100 * x:.{d}f}"


meta, bias, tp, alloc, lock, sens, rate = (load(n) for n in
                                           ["meta", "bias", "two_phase", "allocation", "lockin", "sensitivity", "rate"])

# ------------------------------------------------------------------ basic counts
A = bias["A"]
mac("nItems", f"{meta['n_items']:,}".replace(",", "{,}"))
mac("nModels", str(meta["n_models"]))
mac("nLabelErrA", str(meta["n_label_errors_A"]))
mac("errRateA", pp(A["label_error_rate"]))
fl = A["flaggers"]
hid = [v["n_hidden"] for v in fl.values()]
mac("hiddenMinA", str(min(hid))); mac("hiddenMaxA", str(max(hid)))
mac("hiddenMedA", f"{st.median(hid):.0f}")
mac("inflMinA", pp(min(v["flagger_bias"] for v in fl.values())))
mac("inflMaxA", pp(max(v["flagger_bias"] for v in fl.values())))
mac("rankGainA", str(max(v["flagger_true_rank"] - v["flagger_dr_rank"] for v in fl.values())))
mac("favFlaggersA", str(sum(v["n_flagger_favoring"] > 0 for v in fl.values())))
mac("indepRatioA", f"{st.median(fl[F]['pJ'] / A['pJ_independence'][F] for F in fl):.1f}")
allp = A["panels"][f"top{meta['n_models']}"]
mac("allPanelAdjA", f"{allp['n_adjudicated']:,}".replace(",", "{,}"))
mac("topTenHiddenA", str(A["panels"]["top10"]["n_hidden"])); mac("topTenAdjA", f"{A['panels']['top10']['n_adjudicated']:,}".replace(",", "{,}"))
mac("allPanelAdjPctA", f"{100 * allp['n_adjudicated'] / meta['n_items']:.0f}")
top5 = sorted(fl, key=lambda F: fl[F]["flagger_true_rank"])[:5]
mac("hiddenTopFiveMinA", str(min(fl[F]["n_hidden"] for F in top5)))
mac("hiddenTopFiveMaxA", str(max(fl[F]["n_hidden"] for F in top5)))
mac("adjTopFiveMinA", str(min(fl[F]["n_adjudicated"] for F in top5)))
mac("adjTopFiveMaxA", str(max(fl[F]["n_adjudicated"] for F in top5)))

Bs = [k for k in bias if k.startswith("B:")]
errB = [bias[k]["label_error_rate"] for k in Bs]
mac("errRateBmin", pp(min(errB), 1)); mac("errRateBmax", pp(max(errB), 1))
inflB = [v["flagger_bias"] for k in Bs for v in bias[k]["flaggers"].values()]
mac("inflMinB", pp(min(inflB), 1)); mac("inflMaxB", pp(max(inflB), 1))
favB = [sum(v["n_flagger_favoring"] > 0 for v in bias[k]["flaggers"].values()) for k in Bs]
mac("favFlaggersBmin", str(min(favB))); mac("favFlaggersBmax", str(max(favB)))
gainB = [max(v["flagger_true_rank"] - v["flagger_dr_rank"] for v in bias[k]["flaggers"].values()) for k in Bs]
mac("rankGainBmin", str(min(gainB))); mac("rankGainBmax", str(max(gainB)))
ratB = [st.median(bias[k]["flaggers"][F]["pJ"] / bias[k]["pJ_independence"][F] for F in bias[k]["flaggers"]) for k in Bs]
mac("indepRatioBmin", f"{min(ratB):.1f}"); mac("indepRatioBmax", f"{max(ratB):.1f}")
b0 = bias[SCEN_B]["flaggers"]["meta_llama-3-70b"]
mac("llamaInflB", pp(b0["flagger_bias"], 1)); mac("llamaTrueRankB", str(b0["flagger_true_rank"]))
mac("llamaDrRankB", str(b0["flagger_dr_rank"])); mac("llamaFavB", str(b0["n_flagger_favoring"]))
mac("errRateBmain", pp(bias[SCEN_B]["label_error_rate"], 1))

# ------------------------------------------------------------------ Table: scenario A flaggers
rows = []
order = sorted(fl, key=lambda F: fl[F]["flagger_true_rank"])
for F in order[:6]:
    v = fl[F]
    rows.append(f"{short(F)} & {v['flagger_true_rank']} & {v['n_adjudicated']} & {v['n_caught']} & {v['n_hidden']} & "
                f"{pp(v['flagger_bias'])} & {v['n_inversions']} & {v['n_flagger_favoring']} \\\\")
rows.append("\\midrule")
for k in ["top3", "top5", "top10", f"top{meta['n_models']}"]:
    v = A["panels"][k]
    name = "all models" if k == f"top{meta['n_models']}" else f"top-{k[3:]} panel"
    rows.append(f"{name} (union) & -- & {v['n_adjudicated']} & {v['n_caught']} & {v['n_hidden']} & "
                f"{pp(v['max_panel_bias'])} & {v['n_inversions']} & {v['n_flagger_favoring']} \\\\")
(T / "tab_flaggers_A.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    "Flagger & True rank & Adjudicated & Errors caught & Hidden $|J|$ & Inflation (pp) & Inversions & Flagger-favouring \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# full appendix table, all flaggers in scenario A
rows = []
for F in order:
    v = fl[F]
    rows.append(f"{short(F)} & {100 * A['true_acc'][F]:.2f} & {v['n_adjudicated']} & {v['n_hidden']} & {pp(v['flagger_bias'])} & "
                f"{v['flagger_true_rank']} & {v['flagger_dr_rank']} & {v['n_flagger_favoring']} \\\\")
(T / "tab_flaggers_A_full.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    "Flagger & True acc.\\ (\\%) & Adjudicated & Hidden $|J|$ & Inflation (pp) & True rank & DR rank & Flagger-favouring \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: scenario B summary
rows = []
for k in Bs:
    v = bias[k]; f = v["flaggers"]
    infl = [x["flagger_bias"] for x in f.values()]
    hidden = [x["n_hidden"] for x in f.values()]
    taus = [x["kendall_tau"] for x in f.values()]
    rows.append(f"{LAB[k]} & {pp(v['label_error_rate'], 1)} & {min(hidden)}--{max(hidden)} & "
                f"{pp(st.median(infl), 1)} [{pp(min(infl), 1)}, {pp(max(infl), 1)}] & "
                f"{sum(x['n_flagger_favoring'] > 0 for x in f.values())}/{len(f)} & "
                f"{max(x['flagger_true_rank'] - x['flagger_dr_rank'] for x in f.values())} & "
                f"{min(taus):.2f}--{max(taus):.2f} & "
                f"{st.median(x['pJ'] / v['pJ_independence'][F] for F, x in f.items()):.1f} \\\\")
rowA_infl = [x["flagger_bias"] for x in fl.values()]
rows.insert(0, f"(A) original MMLU & {pp(A['label_error_rate'], 1)} & {min(hid)}--{max(hid)} & "
               f"{pp(st.median(rowA_infl), 2)} [{pp(min(rowA_infl), 2)}, {pp(max(rowA_infl), 2)}] & "
               f"{sum(x['n_flagger_favoring'] > 0 for x in fl.values())}/{len(fl)} & "
               f"{max(x['flagger_true_rank'] - x['flagger_dr_rank'] for x in fl.values())} & "
               f"{min(x['kendall_tau'] for x in fl.values()):.2f}--{max(x['kendall_tau'] for x in fl.values()):.2f} & "
               f"{st.median(fl[F]['pJ'] / A['pJ_independence'][F] for F in fl):.1f} \\\\\n\\midrule")
(T / "tab_scenarios.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    "Labels & Label err.\\ (\\%) & Hidden $|J|$ & Flagger inflation, pp & Flaggers & Max ranks & Kendall $\\tau$ & $P(J)$ / indep. \\\\\n"
    " & & (range) & median [min, max] & favoured & gained & (range) & (median) \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: two-phase
rows = []
for scen, name in [("A", "A"), (SCEN_B, "B")]:
    v = tp[scen]
    for i, r in enumerate(v["rows"]):
        lab = "DR ($\\pi_C=0$)" if r["pi_C"] == 0 else f"{r['pi_C']:g}"
        cw = "--" if r["pi_C"] == 0 else f"{r['coverage_wald_mean']:.3f}"
        cc = "--" if r["pi_C"] == 0 else f"{r['coverage_cp_mean']:.3f}"
        ww = "--" if r["pi_C"] == 0 else pp(r["width_wald_mean"])
        wc = "--" if r["pi_C"] == 0 else pp(r["width_cp_mean"])
        first = f"\\multirow{{{len(v['rows'])}}}{{*}}{{{name}}}" if i == 0 else ""
        rows.append(f"{first} & {lab} & {r['mean_adjudicated']:.0f} & {pp(r['mean_abs_bias'], 3)} & {pp(r['rmse_mean'], 2)} & "
                    f"{pp(r['rmse_flagger'], 2)} & {cw} & {ww} & {cc} & {wc} \\\\")
    rows.append("\\midrule")
rows = rows[:-1]
(T / "tab_two_phase.tex").write_text(
    "\\begin{tabular}{llrrrrrrrr}\n\\toprule\n"
    " & & & & & & \\multicolumn{2}{c}{Wald} & \\multicolumn{2}{c}{Conservative (CP)} \\\\\n"
    "\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
    "Sc. & $\\pi_C$ & Adjudicated & $|\\text{bias}|$ & RMSE & RMSE$_F$ & cover & width & cover & width \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for scen, tag in [("A", "A"), (SCEN_B, "B")]:
    rr = {r["pi_C"]: r for r in tp[scen]["rows"]}
    mac(f"drFlagRmse{tag}", pp(rr[0.0]["rmse_flagger"]))
    mac(f"tpWaldCovOne{tag}", f"{rr[0.01]['coverage_wald_mean']:.2f}")
    mac(f"tpWaldCovTwenty{tag}", f"{rr[0.2]['coverage_wald_mean']:.2f}")
    # round coverage minima down so the text never overstates coverage
    mac(f"tpCpCovMin{tag}", f"{np.floor(1000 * min(r['coverage_cp_min'] for r in tp[scen]['rows'] if r['pi_C'] > 0)) / 1000:.3f}")
    mac(f"tpMaxBias{tag}", pp(max(r["max_abs_bias"] for r in tp[scen]["rows"] if r["pi_C"] > 0), 2))
    mac(f"tpFlagRmseTen{tag}", pp(rr[0.1]["rmse_flagger"]))
    mac(f"tpAdjTen{tag}", f"{rr[0.1]['mean_adjudicated']:.0f}")
    mac(f"tpAdjDR{tag}", f"{rr[0.0]['mean_adjudicated']:.0f}")
mac("reps", "2{,}000")
mac("hiddenGptA", str(fl["openai_gpt-4o-2024-05-13"]["n_hidden"]))

# ------------------------------------------------------------------ Table: allocation
rows = []
for scen, name in [("A", "A"), (SCEN_B, "B")]:
    v = alloc[scen]
    for i, r in enumerate(v["rows"]):
        first = f"\\multirow{{{len(v['rows'])}}}{{*}}{{{name}}}" if i == 0 else ""
        tpr = pp(r["two_phase_rmse"], 2) if "two_phase_rmse" in r else "--"
        tpi = f"{r['two_phase_pi'][1]:.3f}" if "two_phase_pi" in r else "--"
        rows.append(f"{first} & {r['budget_mult']:g} & {r['budget']:.0f} & {pp(r['uniform_rmse'], 2)} & {tpr} & {tpi} & "
                    f"{pp(r['neyman_rmse'], 2)} & {r['neyman_pi'][0]:.2f} & {r['neyman_pi'][1]:.3f} \\\\")
    rows.append("\\midrule")
rows = rows[:-1]
(T / "tab_allocation.tex").write_text(
    "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
    " & & & Uniform & \\multicolumn{2}{c}{Two-phase ($\\pi_\\Delta=1$)} & \\multicolumn{3}{c}{Neyman} \\\\\n"
    "\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\n"
    "Sc. & $B/|\\Delta|$ & $B$ & RMSE & RMSE & $\\pi_C$ & RMSE & $\\pi_\\Delta$ & $\\pi_C$ \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for scen, tag in [("A", "A"), (SCEN_B, "B")]:
    v = alloc[scen]
    r1 = [r for r in v["rows"] if r["budget_mult"] == 1.0][0]
    mac(f"allocUnifOne{tag}", pp(r1["uniform_rmse"], 2)); mac(f"allocNeyOne{tag}", pp(r1["neyman_rmse"], 2))
    mac(f"allocNeyPiDOne{tag}", f"{r1['neyman_pi'][0]:.2f}")
    mac(f"eD{tag}", f"{v['eD']:.3f}"); mac(f"eC{tag}", f"{v['eC']:.4f}")
    mac(f"nD{tag}", str(v["ND"])); mac(f"nC{tag}", str(v["NC"]))

# ------------------------------------------------------------------ lock-in
lb = lock[SCEN_B]["rows"]
mac("lockGptOneB", pp(abs(lb[0]["new_bias"]["openai_gpt-4o-2024-05-13"]), 1))
assert lb[0]["new_bias"]["openai_gpt-4o-2024-05-13"] < 0 and lb[0]["new_bias"]["anthropic_claude-3-opus-20240229"] < 0
mac("lockOpusOneB", pp(abs(lb[0]["new_bias"]["anthropic_claude-3-opus-20240229"]), 1))
mac("lockHiddenOneB", str(lb[0]["n_hidden"])); mac("lockHiddenLastB", str(lb[-1]["n_hidden"]))
mac("lockAdjOneB", str(lb[0]["n_adjudicated"])); mac("lockAdjLastB", str(lb[-1]["n_adjudicated"]))
mac("lockPanelLastB", str(lb[-1]["panel_size"]))
mac("lockGptCorrectOneB", str(lb[0]["new_correct_on_J"]["openai_gpt-4o-2024-05-13"]))
mac("lockGptRepeatOneB", str(lb[0]["new_repeat_on_J"]["openai_gpt-4o-2024-05-13"]))
mac("lockGptCorrectPpOneB", pp(lb[0]["new_correct_on_J"]["openai_gpt-4o-2024-05-13"] / meta["n_items"], 1))
mac("lockGptRepeatPpOneB", pp(lb[0]["new_repeat_on_J"]["openai_gpt-4o-2024-05-13"] / meta["n_items"], 1))
mac("lockOpusCorrectOneB", str(lb[0]["new_correct_on_J"]["anthropic_claude-3-opus-20240229"]))
mac("lockOpusRepeatOneB", str(lb[0]["new_repeat_on_J"]["anthropic_claude-3-opus-20240229"]))
la = lock["A"]["rows"]
mac("lockHiddenLastA", str(la[-1]["n_hidden"])); mac("lockAdjLastA", str(la[-1]["n_adjudicated"]))

# ------------------------------------------------------------------ sensitivity
for scen, tag in [("A", "A"), (SCEN_B, "B")]:
    v = sens[scen]
    mac(f"sensTrueRate{tag}", f"{v['true_hidden_rate']:.4f}")
    mac(f"sensPairTrue{tag}", pp(v["pair_true"]))
    r0 = v["rows"][0]
    mac(f"sensPairDR{tag}", pp(r0["pair_bounds"][0])); mac(f"sensPairDR{tag}abs", pp(abs(r0["pair_bounds"][0])))
    # smallest eps whose identified set contains the true difference
    ok = [r["eps"] for r in v["rows"] if r["pair_bounds"][0] - 1e-12 <= v["pair_true"] <= r["pair_bounds"][1] + 1e-12]
    mac(f"sensEpsCover{tag}", f"{min(ok):g}" if ok else "none")
    signok = [r["eps"] for r in v["rows"] if r["pair_bounds"][1] > 0]
    mac(f"sensEpsSign{tag}", f"{min(signok):g}" if signok else "none")
rows = []
for scen, name in [("A", "A"), (SCEN_B, "B")]:
    v = sens[scen]
    keep = [r for r in v["rows"] if r["eps"] in (0.0, 0.01, 0.05, 0.1, 0.2)]
    for m in v["top"]:
        cells = " & ".join(f"[{100 * r['bounds'][m][0]:.1f}, {100 * r['bounds'][m][1]:.1f}]" for r in keep)
        rows.append(f"{name} & {short(m)}{' (F)' if m == v['flagger'] else ''} & {100 * v['true'][m]:.1f} & {cells} \\\\")
    rows.append("\\midrule")
rows = rows[:-1]
(T / "tab_sensitivity.tex").write_text(
    "\\begin{tabular}{lll" + "c" * 5 + "}\n\\toprule\n"
    "Sc. & Model & True & " + " & ".join(f"$\\varepsilon={e:g}$" for e in (0.0, 0.01, 0.05, 0.1, 0.2)) + " \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ rate
mac("rateEps", f"{rate['eps']:.3f}"); mac("rateGamma", f"{rate['gamma']:.3f}")
ratio = [x["rmse"] / x["theory_se"] for x in rate["rows"]]
mac("rateRatioMin", f"{min(ratio):.2f}"); mac("rateRatioMax", f"{max(ratio):.2f}")

# ------------------------------------------------------------------ data processing
bs = json.loads(Path("data/derived/build_stats.json").read_text())
rs = bs["ref_status"]
for k, name in [("ok", "nOk"), ("wrong_gt", "nWrongGt"), ("excluded:bad_question_clarity", "nBadQ"),
                ("excluded:multiple_correct_answers", "nMulti"), ("excluded:no_correct_answer", "nNoCorrect"),
                ("excluded:expert", "nExpert"), ("excluded:bad_options_clarity", "nBadOpt"),
                ("excluded:no_helm_match", "nNoMatch"), ("wrong_gt_unparsed", "nUnparsed"),
                ("wrong_gt_same_as_label", "nSameAsLabel"), ("excluded:label_differs_from_helm", "nLabelDiffers")]:
    mac(name, f"{rs.get(k, 0):,}".replace(",", "{,}"))
d = bs["helm_scoring_disagreements"]
mac("helmDisagreeTotal", str(sum(d.values()))); mac("helmDisagreeMax", str(max(d.values())))
mac("helmInstances", f"{next(iter(bs['helm_instances_per_model'].values())):,}".replace(",", "{,}"))
inv = bs["invalid_outputs_in_analysis_set"]
mac("invalidMax", str(max(inv.values()))); mac("invalidMaxModel", short(max(inv, key=inv.get)))
mac("invalidZero", str(sum(v == 0 for v in inv.values())))

# ------------------------------------------------------------------ simultaneous exact intervals
sim = load("simultaneous")
rows = []
for scen, name in [("A", "A"), (SCEN_B, "B")]:
    v = sim[scen]; tpr = {r["pi_C"]: r for r in tp[scen]["rows"]}
    for i, r in enumerate(v["rows"]):
        t = tpr[r["pi_C"]]
        first = f"\\multirow{{{len(v['rows'])}}}{{*}}{{{name}}}" if i == 0 else ""
        rows.append(f"{first} & {r['pi_C']:g} & {100 * r['frac_reps_no_sampled_error']:.0f} & {np.floor(1000 * r['event_prob']) / 1000:.3f} & "
                    f"{pp(r['width_single'], 1)} & {pp(t['width_cp_mean'], 1)} & {pp(r['width_pair'], 1)} & "
                    f"{np.mean(t['pair_coverage']):.2f} & {100 * r['certified_adjacent']:.0f} & {100 * r['wrongly_certified']:.0f} \\\\")
    rows.append("\\midrule")
rows = rows[:-1]
(T / "tab_simultaneous.tex").write_text(
    "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
    " & & No error & Simult. & \\multicolumn{2}{c}{Width, accuracy (pp)} & \\multicolumn{2}{c}{Pairs} & \\multicolumn{2}{c}{Adjacent pairs (\\%)} \\\\\n"
    "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
    "Sc. & $\\pi_C$ & sampled (\\%) & coverage & Thm & CP & width & Wald cov. & certified & wrong \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
for scen, tag in [("A", "A"), (SCEN_B, "B")]:
    v = sim[scen]; rr = {r["pi_C"]: r for r in v["rows"]}
    mac(f"simCovMin{tag}", f"{np.floor(1000 * min(r['event_prob'] for r in v['rows'])) / 1000:.3f}")
    mac(f"simWidthOne{tag}", pp(rr[0.01]["width_single"], 1)); mac(f"simWidthFive{tag}", pp(rr[0.05]["width_single"], 1))
    mac(f"simWidthHalf{tag}", pp(rr[0.5]["width_single"], 1))
    mac(f"simPairWidthFive{tag}", pp(rr[0.05]["width_pair"], 1)); mac(f"simPairWidthHalf{tag}", pp(rr[0.5]["width_pair"], 1))
    mac(f"simCertFive{tag}", f"{100 * rr[0.05]['certified_adjacent']:.0f}"); mac(f"simCertHalf{tag}", f"{100 * rr[0.5]['certified_adjacent']:.0f}")
    mac(f"simNoErrOne{tag}", f"{100 * rr[0.01]['frac_reps_no_sampled_error']:.0f}")
    mac(f"simTrueCert{tag}", f"{100 * v['true_certifiable']:.0f}"); mac(f"simNadj{tag}", str(v["n_adjacent"]))
    t = {r["pi_C"]: r for r in tp[scen]["rows"]}
    mac(f"cpWidthOne{tag}", pp(t[0.01]["width_cp_mean"], 1)); mac(f"waldPairCovOne{tag}", f"{min(t[0.01]['pair_coverage']):.2f}--{max(t[0.01]['pair_coverage']):.2f}")
mac("simWrongMax", f"{100 * max(r['wrongly_certified'] for v in sim.values() for r in v['rows']):.0f}")

# ------------------------------------------------------------------ CoNLL-2003 re-analysis (Reiss et al. vs CoNLL++)
cn = load("conll")
NICE = {"hf:bert_large_dbmdz": "BERT-large (dbmdz)", "hf:bert_large_dslim": "BERT-large (dslim)",
        "hf:bert_base_dslim": "BERT-base (dslim)", "hf:distilbert_elastic": "DistilBERT (elastic)",
        "hf:roberta_large_jb": "RoBERTa-large (J.-B.)"}
ENTRANT = {"mccallum": "McCallum", "demeulder": "De Meulder", "carrerasa": "Carreras (a)", "carrerasb": "Carreras (b)"}
nice = lambda m: NICE.get(m, ENTRANT.get(m.split(":")[1], m.split(":")[1].capitalize()) + " (2003)")
rows = []
for m in sorted(cn["models"], key=lambda m: -cn["models"][m]["true"]):
    v = cn["models"][m]
    rows.append(f"{nice(m)} & {100 * v['true']:.2f} & {100 * v['dr']:.2f} & {100 * v['bias']:+.2f} & "
                f"{100 * v['hidden_part']:+.3f} & {100 * v['adj_part']:+.3f} & {v['hidden_correct']} & {v['hidden_repeat']} & "
                f"{cn['rank_true'][m]} & {cn['rank_dr'][m]} \\\\")
(T / "tab_conll.tex").write_text(
    "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
    " & \\multicolumn{2}{c}{Token accuracy (\\%)} & \\multicolumn{3}{c}{DR bias (pp)} & \\multicolumn{2}{c}{Hidden errors} & \\multicolumn{2}{c}{Rank} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
    "Model & CoNLL++ & DR & total & hidden & disagr. & correct & repeat & true & DR \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
mac("cnItems", f"{cn['n_items']:,}".replace(",", "{,}")); mac("cnExcluded", str(cn["n_excluded"]))
mac("cnAdj", f"{cn['n_adjudicated']:,}".replace(",", "{,}")); mac("cnAdjPct", f"{100 * cn['n_adjudicated'] / cn['n_items']:.1f}")
mac("cnFlagged", f"{cn['n_flagged']:,}".replace(",", "{,}")); mac("cnIncidental", str(cn["n_incidental_only"]))
mac("cnRefErr", str(cn["ref_errors"])); mac("cnRefErrIn", str(cn["ref_errors_in_delta"])); mac("cnHidden", str(cn["hidden"]))
mac("cnHiddenPct", f"{100 * cn['hidden'] / cn['ref_errors']:.0f}")
mac("cnReissChanges", str(cn["reiss_changes"])); mac("cnAgreeChanges", str(cn["agree_both_change"]))
mac("cnDisagreeIn", str(cn["adjudicator_disagree_in_delta"]))
mac("cnTau", f"{cn['kendall_tau']:.3f}"); mac("cnNinv", str(len(cn["inversions"]))); mac("cnNmodels", str(len(cn["models"])))
hp = [v["hidden_part"] for v in cn["models"].values()]; ap = [v["adj_part"] for v in cn["models"].values()]
mac("cnHiddenAbsMax", f"{100 * max(abs(x) for x in hp):.2f}")
mac("cnHiddenMin", f"{100 * min(hp):+.2f}"); mac("cnHiddenMax", f"{100 * max(hp):+.2f}")
mac("cnAdjMin", f"{100 * min(ap):+.2f}"); mac("cnAdjMax", f"{100 * max(ap):+.2f}")
mac("cnAllHiddenPos", "every" if min(hp) > 0 else "not every")
mac("cnConcRate", f"{100 * cn['true_concordant_error_rate']:.2f}")
ent = [v["hidden_part"] for m, v in cn["models"].items() if m.startswith("entrant")]
hf = [v["hidden_part"] for m, v in cn["models"].items() if m.startswith("hf")]
mac("cnHiddenEntrantMean", f"{100 * np.mean(ent):+.3f}"); mac("cnHiddenHfMean", f"{100 * np.mean(hf):+.3f}")
mac("cnNhf", str(len(hf))); mac("cnNentrant", str(len(ent)))
if cn["inversions"]:
    a, b = cn["inversions"][0]
    mac("cnInvA", nice(a)); mac("cnInvB", nice(b))
    mac("cnInvGap", f"{100 * abs(cn['models'][a]['true'] - cn['models'][b]['true']):.2f}")
pb = cn["pair_bounds"]
k = max(pb, key=lambda k: pb[k]["true"] - pb[k]["dr"])     # pair whose gap DR understates most
a, b = k.split("|"); mac("cnPairA", nice(a)); mac("cnPairB", nice(b))
mac("cnPairDR", f"{100 * pb[k]['dr']:.3f}"); mac("cnPairTrue", f"{100 * pb[k]['true']:.3f}")
eps_sign = [e for e, lo, hi in pb[k]["bounds"] if lo <= 0 <= hi]
mac("cnPairEpsAmbig", f"{min(eps_sign):g}" if eps_sign else "none")
bd = cn["bounds"]; top = max(cn["models"], key=lambda m: cn["models"][m]["true"])
w = [(e, hi - lo) for e, lo, hi in bd[top] if abs(e - 0.002) < 1e-12][0][1]
mac("cnWidthTwo", pp(w, 2))

es = cn["entity_subset"]; rows = []
for m in sorted(es["models"], key=lambda m: -es["models"][m]["true"]):
    v = es["models"][m]
    rows.append(f"{nice(m)} & {100 * v['true']:.2f} & {100 * v['dr']:.2f} & {100 * v['bias']:+.2f} & "
                f"{100 * v['hidden_part']:+.2f} & {100 * v['adj_part']:+.2f} & {es['rank_true'][m]} & {es['rank_dr'][m]} \\\\")
(T / "tab_conll_entity.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    " & \\multicolumn{2}{c}{Accuracy (\\%)} & \\multicolumn{3}{c}{DR bias (pp)} & \\multicolumn{2}{c}{Rank} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n"
    "Model & CoNLL++ & DR & total & hidden & disagr. & true & DR \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
ehp = [v["hidden_part"] for v in es["models"].values()]; eap = [v["adj_part"] for v in es["models"].values()]
mac("cnEnt", f"{es['n_items']:,}".replace(",", "{,}"))
mac("cnEntHiddenMin", f"{100 * min(ehp):+.2f}"); mac("cnEntHiddenMax", f"{100 * max(ehp):+.2f}")
mac("cnEntAdjMin", f"{100 * min(eap):+.2f}"); mac("cnEntAdjMax", f"{100 * max(eap):+.2f}")
mac("cnEntTau", f"{es['kendall_tau_b']:.3f}"); mac("cnEntNinv", str(len(es["inversions"])))
if es["inversions"]:
    a, b = es["inversions"][0]
    mac("cnEntInvA", nice(a)); mac("cnEntInvB", nice(b))
    mac("cnEntInvGap", f"{100 * abs(es['models'][a]['true'] - es['models'][b]['true']):.2f}")
mac("cnTies", str(cn["dr_rank_ties"])); mac("cnPairs", str(len(cn["models"]) * (len(cn["models"]) - 1) // 2))
mac("cnDisagreeChanges", str(cn["reiss_changes"] - cn["agree_both_change"]))

# ------------------------------------------------------------------ labeler sweep
sw = load("sweep")
llm = {k: v for k, v in sw.items() if k.startswith("B:")}
mac("sweepN", str(len(llm)))
mac("sweepErrMin", pp(min(v["label_error_rate"] for v in llm.values()), 1))
mac("sweepErrMax", pp(max(v["label_error_rate"] for v in llm.values()), 1))
best = min(llm, key=lambda k: llm[k]["label_error_rate"])
mac("sweepBestLabeler", short(best[2:])); mac("sweepBestErr", pp(llm[best]["label_error_rate"], 1))
mac("sweepBestMedInfl", pp(llm[best]["median_inflation"], 1)); mac("sweepBestMaxGain", str(llm[best]["max_rank_gain"]))
mac("sweepMedInflMin", pp(min(v["median_inflation"] for v in llm.values()), 1))
mac("sweepMedInflMax", pp(max(v["median_inflation"] for v in llm.values()), 1))
mac("sweepMaxGainMax", str(max(v["max_rank_gain"] for v in llm.values())))
mac("sweepFavMin", str(min(v["n_flaggers_favoured"] for v in llm.values())))
import pandas as _pd
_df = _pd.read_parquet("data/derived/redux_helm.parquet"); _df = _df[_df.ref_status.isin(["ok", "wrong_gt"])]
mac("dupAgree", f"{100 * (_df['google_text-unicorn@001'] == _df['writer_palmyra-x-v3']).mean():.2f}")
mac("dupDiffer", str(int((_df['google_text-unicorn@001'] != _df['writer_palmyra-x-v3']).sum())))

(Path("paper") / "numbers.tex").write_text("% generated by code/make_tables.py; do not edit\n" +
                                           "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
print(len(macros), "macros;", sorted(p.name for p in T.iterdir()))
