# Numbers for Thesis v6 (generated 2026-10-25 15:36)

Compare each value with the thesis text. Sections 4.1-4.6 should match the reference run; Sections 4.7-4.8 are new.

## Section 4.1 / Figure 4.1 / Table 4.1
- Rows 1000, fraud counts {'N': 753, 'Y': 247}
- Unknown values: {'collision_type': 178, 'property_damage': 360, 'police_report_available': 343}; 'None' in authorities_contacted: 91
- Fraud rate by severity: {'Major Damage': 0.605, 'Minor Damage': 0.107, 'Total Loss': 0.129, 'Trivial Damage': 0.067}
- Top hobby fraud rates: {'chess': 0.826, 'cross-fit': 0.743, 'yachting': 0.302, 'board-games': 0.292}
- Skewness of total claim -0.59; Spearman total-vehicle 0.96; age-months 0.91
- Table 4.1: outputs/tables/table_4_1_associations.csv

## Section 4.2 / Figure 4.2 / Appendix A
- RFECV kept 10 of 123 encoded features; best CV PR-AUC 0.691; one-SE rule 9
- Features: months_as_customer, policy_annual_premium, total_claim_amount, injury_claim, property_claim, vehicle_claim, policy_tenure_days, insured_hobbies_chess, insured_hobbies_cross-fit, incident_severity_Major Damage

## Table 3.1 (selected hyperparameters)
- LR: {'C': 10} (CV PR-AUC 0.69)
- RF: {'max_depth': 6, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 300} (CV PR-AUC 0.648)
- SVM: {'C': 10, 'gamma': 0.1} (CV PR-AUC 0.662)

## Table 4.2 (test set, tuned thresholds)
| Model | Thr | Precision | Recall | F1 | F1 CI | PR-AUC (CI) | ROC-AUC | MCC | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| LR | 0.698 | 0.636 | 0.857 | 0.73 | 0.633–0.804 | 0.549 (0.439–0.699) | 0.851 | 0.639 | 0.845 |
| RF | 0.495 | 0.625 | 0.816 | 0.708 | 0.604–0.789 | 0.543 (0.424–0.696) | 0.821 | 0.606 | 0.835 |
| SVM | 0.303 | 0.636 | 0.857 | 0.73 | 0.633–0.804 | 0.522 (0.407–0.672) | 0.818 | 0.639 | 0.845 |
| Voting | 0.394 | 0.636 | 0.857 | 0.73 | 0.633–0.804 | 0.542 (0.428–0.696) | 0.82 | 0.639 | 0.845 |

- F1 at threshold 0.5: LR 0.73, RF 0.708, SVM 0.704, Voting 0.685
- Out-of-fold PR-AUC: LR 0.677, RF 0.633, SVM 0.636, Voting 0.64
- RF confusion matrix [[TN, FP], [FN, TP]]: [[127, 24], [9, 40]]  (Figure 4.3)

## Table 4.4 (imbalance ablation, RF)
- no_resampling: {'pr_auc': 0.565, 'roc_auc': 0.842, 'f1_at_0.5': 0.647}
- class_weight: {'pr_auc': 0.545, 'roc_auc': 0.84, 'f1_at_0.5': 0.708}
- smote_nc: {'pr_auc': 0.543, 'roc_auc': 0.821, 'f1_at_0.5': 0.708}

## Section 4.5 / Table 4.5 / Figures 4.5-4.8
- SHAP base value 0.5
- Global mean |SHAP|: incident_severity_Major Damage 0.2551; insured_hobbies_chess 0.0509; insured_hobbies_cross-fit 0.0475; policy_tenure_days 0.022; property_claim 0.0144; vehicle_claim 0.0122; months_as_customer 0.0085; policy_annual_premium 0.0077; total_claim_amount 0.0068; injury_claim 0.0057
- Agreement: {'n_true_positives': 40, 'overlap_top3': 0.592, 'chance_top3': 0.3, 'overlap_top5': 0.605, 'overlap_top5_sd': 0.13, 'chance_top5': 0.5, 'jaccard_top5': 0.446, 'spearman_abs': 0.389, 'lime_three_seed_stability': 0.56, 'shap_recompute_max_diff': 0.0}
- Example claim: {'test_row': 121, 'prob': 0.88, 'shap_top5': [['insured_hobbies_cross-fit', 0.477], ['incident_severity_Major Damage', -0.139], ['total_claim_amount', 0.023], ['insured_hobbies_chess', -0.022], ['policy_tenure_days', 0.017]], 'lime_top5': [['incident_severity_Major Damage', 0.199], ['insured_hobbies_cross-fit', 0.054], ['total_claim_amount', 0.019], ['injury_claim', -0.009], ['insured_hobbies_chess', 0.008]]}

## Section 4.6
- Flagged 64 (40 TP, 24 FP); C2 driver recall 1.0, sign accuracy 1.0, words 73.2 (range [70, 76])

## Section 4.7 / Table 4.6 (paste from outputs/tables/table_4_6_fidelity_consistency.csv)
- C3-P0-G1: driver_recall 0.89, sign_accuracy 0.999, ucr 0.091, words 130.615, seconds 8.573, bertscore_f1 0.952, driver_jaccard 0.959, meets_recall_target False, meets_ucr_target False
- C3-P0-G2: driver_recall 0.909, sign_accuracy 1.0, ucr 0.152, words 146.875, seconds 5.642, bertscore_f1 0.949, driver_jaccard 0.906, meets_recall_target True, meets_ucr_target False
- C3-P1-G1: driver_recall 0.899, sign_accuracy 0.996, ucr 0.116, words 146.5, seconds 8.655, bertscore_f1 0.953, driver_jaccard 0.948, meets_recall_target False, meets_ucr_target False
- C3-P1-G2: driver_recall 0.911, sign_accuracy 0.998, ucr 0.124, words 155.276, seconds 5.26, bertscore_f1 0.964, driver_jaccard 0.907, meets_recall_target True, meets_ucr_target False
- C3-P2-G1: driver_recall 0.94, sign_accuracy 0.995, ucr 0.088, words 152.651, seconds 8.141, bertscore_f1 0.944, driver_jaccard 0.956, meets_recall_target True, meets_ucr_target False
- C3-P2-G2: driver_recall 0.895, sign_accuracy 0.993, ucr 0.159, words 162.047, seconds 4.864, bertscore_f1 0.938, driver_jaccard 0.874, meets_recall_target False, meets_ucr_target False
- C3-P3-G1: driver_recall 0.975, sign_accuracy 0.979, ucr 0.06, words 111.438, seconds 6.973, bertscore_f1 0.982, driver_jaccard 0.978, meets_recall_target True, meets_ucr_target False
- C3-P3-G2: driver_recall 0.939, sign_accuracy 0.984, ucr 0.058, words 95.229, seconds 3.482, bertscore_f1 0.97, driver_jaccard 0.947, meets_recall_target True, meets_ucr_target False

## Section 4.8 / Table 4.7
- clarity: means {'C0': 3.3, 'C1': 2.0, 'C2': 3.55, 'C3-P0-G1': 3.78, 'C3-P0-G2': 3.19, 'C3-P1-G1': 3.71, 'C3-P1-G2': 3.15, 'C3-P2-G1': 3.78, 'C3-P2-G2': 3.24, 'C3-P3-G1': 3.93, 'C3-P3-G2': 3.65}; Friedman chi2(10) = 357.56, p = 9.89e-71; best LLM C3-P3-G1; Holm p vs best {'C0': 8.25732094164012e-10, 'C1': 1.1503451871010318e-13, 'C2': 2.5282997053365597e-06}
- actionability: means {'C0': 1.0, 'C1': 1.0, 'C2': 2.72, 'C3-P0-G1': 2.73, 'C3-P0-G2': 2.51, 'C3-P1-G1': 2.65, 'C3-P1-G2': 2.43, 'C3-P2-G1': 2.74, 'C3-P2-G2': 2.47, 'C3-P3-G1': 3.03, 'C3-P3-G2': 2.84}; Friedman chi2(10) = 401.38, p = 4.79e-80; best LLM C3-P3-G1; Holm p vs best {'C0': 6.213608977065098e-14, 'C1': 6.213608977065098e-14, 'C2': 1.1966598263461854e-05}

## Section 5.4 inputs
- By generator: {'G1': {'driver_recall': 0.926, 'sign_accuracy': 0.992, 'ucr': 0.089, 'seconds': 8.085}, 'G2': {'driver_recall': 0.914, 'sign_accuracy': 0.994, 'ucr': 0.123, 'seconds': 4.812}}
- By prompt: {'P0': {'driver_recall': 0.899, 'sign_accuracy': 0.999, 'ucr': 0.122, 'words': 138.745}, 'P1': {'driver_recall': 0.905, 'sign_accuracy': 0.997, 'ucr': 0.12, 'words': 150.888}, 'P2': {'driver_recall': 0.917, 'sign_accuracy': 0.994, 'ucr': 0.123, 'words': 157.349}, 'P3': {'driver_recall': 0.957, 'sign_accuracy': 0.981, 'ucr': 0.059, 'words': 103.333}}
- Sign accuracy for positive vs negative drivers: {'false': 0.99, 'true': 0.994}
- Section 4.9 dashboard fallback rate overall 0.354; by condition {'C3-P0-G1': 0.521, 'C3-P0-G2': 0.333, 'C3-P1-G1': 0.474, 'C3-P1-G2': 0.328, 'C3-P2-G1': 0.307, 'C3-P2-G2': 0.333, 'C3-P3-G1': 0.208, 'C3-P3-G2': 0.323}

## Section 4.8 judge validation
- {"n_items": 343, "clarity": {"kappa_judge_vs_rater1": 0.006193425440686129, "kappa_judge_vs_rater2": 0.02785885651782416, "kappa_judge_vs_mean_human": -0.044739830338654585, "kappa_rater1_vs_rater2": 0.49275362318840576, "meets_threshold": false}, "actionability": {"kappa_judge_vs_rater1": 0.0199218022714579, "kappa_judge_vs_rater2": 0.10073452256033577, "kappa_judge_vs_mean_human": 0.01829908504574773, "kappa_rater1_vs_rater2": 0.1983879977655415, "meets_threshold": false}, "ucr_hand_check": {"n": 0, "agreement_with_judge": null}}
