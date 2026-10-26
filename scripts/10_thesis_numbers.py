"""scripts/10_thesis_numbers.py - collects every number the thesis needs into one file,
outputs/THESIS_NUMBERS.md, organised by thesis section and table.   Run:  python scripts/10_thesis_numbers.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import config
 
def load(name):
    p = config.OUT / name
    return json.load(open(p)) if p.exists() else None
 
E, T, X, B, L, H = (load("eda_summary.json"), load("train_results.json"), load("explain_results.json"),
                    load("baselines_summary.json"), load("llm_results.json"), load("human_validation.json"))
md = ["# Numbers for Thesis v6 (generated " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + ")\n",
      "Compare each value with the thesis text. Sections 4.1-4.6 should match the reference run; Sections 4.7-4.8 are new.\n"]
if E:
    md += ["## Section 4.1 / Figure 4.1 / Table 4.1", f"- Rows {E['rows']}, fraud counts {E['fraud_counts']}",
           f"- Unknown values: {E['unknown_values']}; 'None' in authorities_contacted: {E['authorities_none']}",
           f"- Fraud rate by severity: {E['fraud_rate_by_severity']}", f"- Top hobby fraud rates: {E['fraud_rate_by_hobby_top4']}",
           f"- Skewness of total claim {E['skew_total_claim']}; Spearman total-vehicle {E['spearman_total_vs_vehicle']}; age-months {E['spearman_age_vs_months']}",
           "- Table 4.1: outputs/tables/table_4_1_associations.csv\n"]
if T:
    md += ["## Section 4.2 / Figure 4.2 / Appendix A", f"- RFECV kept {T['rfecv_n_selected']} of {T['n_encoded']} encoded features; best CV PR-AUC {T['rfecv_best_cv_pr_auc']}; one-SE rule {T['rfecv_k_one_se']}",
           f"- Features: {', '.join(T['selected_features'])}\n", "## Table 3.1 (selected hyperparameters)"]
    md += [f"- {m}: {v['best_params']} (CV PR-AUC {v['cv_pr_auc']})" for m, v in T["tuning"].items()]
    md += ["\n## Table 4.2 (test set, tuned thresholds)", "| Model | Thr | Precision | Recall | F1 | F1 CI | PR-AUC (CI) | ROC-AUC | MCC | Accuracy |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in T["test"].items():
        t = v["tuned"]
        md.append(f"| {m} | {t['threshold']} | {t['precision']} | {t['recall']} | {t['f1']} | {t['f1_ci'][0]}–{t['f1_ci'][1]} | {t['pr_auc']} ({t['pr_auc_ci'][0]}–{t['pr_auc_ci'][1]}) | {t['roc_auc']} | {t['mcc']} | {t['accuracy']} |")
    md += ["", "- F1 at threshold 0.5: " + ", ".join(f"{m} {v['at_0.5']['f1']}" for m, v in T["test"].items()),
           "- Out-of-fold PR-AUC: " + ", ".join(f"{m} {v['oof_pr_auc']}" for m, v in T["test"].items()),
           f"- RF confusion matrix [[TN, FP], [FN, TP]]: {T['test']['RF']['tuned']['confusion_matrix']}  (Figure 4.3)\n",
           "## Table 4.4 (imbalance ablation, RF)"] + [f"- {k}: {v}" for k, v in T["ablation_rf"].items()] + [""]
if X:
    md += ["## Section 4.5 / Table 4.5 / Figures 4.5-4.8", f"- SHAP base value {X['shap_base_value']}",
           "- Global mean |SHAP|: " + "; ".join(f"{a} {b}" for a, b in X["shap_global"]),
           f"- Agreement: {X['agreement']}", f"- Example claim: {X['example_claim']}\n"]
if B:
    md += ["## Section 4.6", f"- Flagged {B['flagged']} ({B['true_positives']} TP, {B['false_positives']} FP); C2 driver recall {B['C2_driver_recall']}, sign accuracy {B['C2_sign_accuracy']}, words {B['C2_words_mean']} (range {B['C2_words_range']})\n"]
if L:
    md += ["## Section 4.7 / Table 4.6 (paste from outputs/tables/table_4_6_fidelity_consistency.csv)"]
    md += [f"- {c}: " + ", ".join(f"{k} {v}" for k, v in r.items()) for c, r in L["table_4_6"].items()]
    md += ["", "## Section 4.8 / Table 4.7"] + [f"- {m}: means {L[m]['means']}; Friedman chi2({L[m]['friedman_df']}) = {L[m]['friedman_chi2']}, p = {L[m]['friedman_p']:.3g}; best LLM {L[m]['best_llm_condition']}; Holm p vs best {L[m]['holm_p_vs_best']}" for m in ["clarity", "actionability"]]
    md += ["", "## Section 5.4 inputs", f"- By generator: {L['by_generator']}", f"- By prompt: {L['by_prompt']}",
           f"- Sign accuracy for positive vs negative drivers: {L.get('sign_accuracy_positive_vs_negative_drivers')}",
           f"- Section 4.9 dashboard fallback rate overall {L.get('fallback_rate_overall')}; by condition {L.get('fallback_rate_by_condition')}\n"]
if H:
    md += ["## Section 4.8 judge validation", f"- {json.dumps(H, default=float)}\n"]
if config.MOCK: md.insert(1, "\n**WARNING: MOCK RUN - these LLM numbers are fake and must not be used in the thesis.**\n")
(config.OUT / "THESIS_NUMBERS.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
