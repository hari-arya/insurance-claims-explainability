"""scripts/09_stats.py - builds Table 4.6 (fidelity and consistency) and Table 4.7 (judge ratings with the
Friedman and Holm-corrected Wilcoxon tests), plus generator and prompt effects (Thesis Sections 4.7-4.8, 5.4-5.5).
Run:  python scripts/09_stats.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import config
from src.explanations import ClaimBank, C2
from src.text_eval import fidelity
from src import store
 
bank = ClaimBank()
G = pd.DataFrame(store.read(store.GENERATIONS)); J = pd.DataFrame(store.read(store.JUDGEMENTS))
# ---- automatic fidelity per narrative
F = pd.DataFrame([{**fidelity(bank.claim(g["idx"]), g["text"], bank.names), "idx": g["idx"],
                   "cond": f"C3-{g['prompt']}-{g['gen']}", "run": g["run"], "seconds": g.get("seconds")} for _, g in G.iterrows()])
F = F.merge(J[["idx", "cond", "run", "ucr"]], on=["idx", "cond", "run"], how="left")
cons = pd.read_csv(config.TAB / "consistency_by_claim.csv") if (config.TAB / "consistency_by_claim.csv").exists() else None
t46 = F.groupby("cond").agg(driver_recall=("driver_recall", "mean"), sign_accuracy=("sign_accuracy", "mean"),
                            ucr=("ucr", "mean"), words=("words", "mean"), seconds=("seconds", "mean"))
if cons is not None: t46 = t46.join(cons.groupby("cond")[["bertscore_f1", "driver_jaccard"]].mean())
t46["meets_recall_target"] = t46["driver_recall"] >= config.TARGET_DRIVER_RECALL
t46["meets_ucr_target"] = t46["ucr"] < config.TARGET_UCR
t46.round(3).to_csv(config.TAB / "table_4_6_fidelity_consistency.csv")
# ---- judge ratings (average the 3 runs of each C3 condition per claim)
Jm = J.groupby(["idx", "cond"])[["clarity", "actionability", "ucr"]].mean().reset_index()
out = {"table_4_6": t46.round(3).to_dict(orient="index")}
for m in ["clarity", "actionability"]:
    W = Jm.pivot(index="idx", columns="cond", values=m).dropna()
    chi, p = friedmanchisquare(*[W[c] for c in W.columns])
    best = W.filter(like="C3").mean().idxmax()
    ps = [wilcoxon(W[best], W[b]).pvalue for b in ["C0", "C1", "C2"]]
    out[m] = {"means": W.mean().round(2).to_dict(), "friedman_chi2": round(chi, 2), "friedman_df": W.shape[1] - 1,
              "friedman_p": p, "best_llm_condition": best,
              "holm_p_vs_best": dict(zip(["C0", "C1", "C2"], multipletests(ps, method="holm")[1]))}
t47 = pd.DataFrame({m: out[m]["means"] for m in ["clarity", "actionability"]})
t47.to_csv(config.TAB / "table_4_7_judge_ratings.csv")
# ---- effects of generator and prompt (for Section 5.4)
F["gen"] = F["cond"].str[-2:]; F["prompt"] = F["cond"].str[3:5]
out["by_generator"] = F.groupby("gen")[["driver_recall", "sign_accuracy", "ucr", "seconds"]].mean().round(3).to_dict(orient="index")
out["by_prompt"] = F.groupby("prompt")[["driver_recall", "sign_accuracy", "ucr", "words"]].mean().round(3).to_dict(orient="index")
neg = []  # errors on negative drivers (features whose presence/absence lowers the score)
for _, g in G.iterrows():
    c = bank.claim(g["idx"])
    for d in c["drivers"]:
        from src.text_eval import mentioned, stated_sign
        if mentioned(g["text"], d["feature"]):
            s = stated_sign(g["text"], d["feature"])
            if s != 0: neg.append({"negative": d["shap"] < 0, "correct": s == (1 if d["shap"] > 0 else -1)})
neg = pd.DataFrame(neg)
if len(neg): out["sign_accuracy_positive_vs_negative_drivers"] = neg.groupby("negative")["correct"].mean().round(3).to_dict()
# share of narratives the dashboard would replace with the template (Thesis Sections 3.14 and 4.9)
F["passes_check"] = (F["driver_recall"] >= config.TARGET_DRIVER_RECALL) & (F["sign_accuracy"].isna() | (F["sign_accuracy"] == 1))
out["fallback_rate_by_condition"] = (1 - F.groupby("cond")["passes_check"].mean()).round(3).to_dict()
out["fallback_rate_overall"] = round(float(1 - F["passes_check"].mean()), 3)
json.dump(out, open(config.OUT / "llm_results.json", "w"), indent=1, default=float)
print(t46.round(3)); print(t47.round(2)); print(json.dumps({k: out[k] for k in ["clarity", "actionability"]}, indent=1, default=float))
