"""scripts/03_explain.py - SHAP and LIME for the explained model, agreement and stability
(Thesis Section 4.5; Table 4.5; Figures 4.5-4.8; Appendix A).   Run:  python scripts/03_explain.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, joblib, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer
import config
 
A = joblib.load(config.MODELS / "trained.joblib")
pipe = A["models"][config.EXPLAINED_MODEL]
rf = pipe.named_steps["clf"]
to_model_space = lambda Z: pipe[1:3].transform(Z)          # the exact inputs the classifier sees
Xtr, Xte = to_model_space(A["X_train"]), to_model_space(A["X_test"])
names = A["selected"]; P = len(names); K = config.TOP_K
p_te = A["probs"][config.EXPLAINED_MODEL]; thr = A["thresholds"][config.EXPLAINED_MODEL]; y_te = A["y_test"]
 
def fraud_shap(explainer, X):
    v = explainer.shap_values(X)
    return v[1] if isinstance(v, list) else (v[:, :, 1] if np.ndim(v) == 3 else v)
 
explainer = shap.TreeExplainer(rf)
sv_te, sv_tr = fraud_shap(explainer, Xte), fraud_shap(explainer, Xtr)
base = explainer.expected_value; base = float(base[1] if np.ndim(base) else base)
R = {"shap_base_value": round(base, 3)}
mean_abs = np.abs(sv_te).mean(0); order = np.argsort(-mean_abs)
R["shap_global"] = [[names[i], round(float(mean_abs[i]), 4)] for i in order]
 
fig, ax = plt.subplots(figsize=(6.5, 4.2)); top = order[::-1]
ax.barh([names[i] for i in top], mean_abs[top], color="#1f4e79"); ax.set_xlabel("Mean |SHAP value| (fraud probability)")
plt.tight_layout(); plt.savefig(config.FIG / "fig_4_5_shap_global.png", dpi=150); plt.close()
plt.figure(); shap.summary_plot(sv_te, Xte, feature_names=names, show=False, plot_size=(7, 4.5))
plt.tight_layout(); plt.savefig(config.FIG / "fig_4_6_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
 
# --- agreement on correctly flagged fraud claims (Table 4.5)
tp = np.where((p_te >= thr) & (y_te == 1))[0]
lime_main = LimeTabularExplainer(Xtr, feature_names=names, class_names=["N", "Y"], mode="classification",
                                 discretize_continuous=False, random_state=config.LIME_SEEDS[0])
def lime_rank(explainer, x):
    m = dict(explainer.explain_instance(x, rf.predict_proba, num_features=P, num_samples=config.LIME_SAMPLES).as_map()[1])
    return m, sorted(m, key=lambda i: -abs(m[i]))
o5, o3, jac, rho, stab, per_claim = [], [], [], [], [], []
for j in tp:
    s_rank = list(np.argsort(-np.abs(sv_te[j])))
    lm, l_rank = lime_rank(lime_main, Xte[j])
    o5.append(len(set(s_rank[:K]) & set(l_rank[:K])) / K); o3.append(len(set(s_rank[:3]) & set(l_rank[:3])) / 3)
    jac.append(len(set(s_rank[:K]) & set(l_rank[:K])) / len(set(s_rank[:K]) | set(l_rank[:K])))
    rho.append(spearmanr(np.abs(sv_te[j]), [abs(lm.get(i, 0)) for i in range(P)]).correlation)
    other = [set(lime_rank(LimeTabularExplainer(Xtr, feature_names=names, class_names=["N", "Y"], mode="classification",
                                                discretize_continuous=False, random_state=s), Xte[j])[1][:K]) for s in config.LIME_SEEDS[1:]]
    stab.append(len(set(l_rank[:K]).intersection(*other)) / K)
    per_claim.append({"test_row": int(j), "prob": round(float(p_te[j]), 3), "overlap5": o5[-1]})
R["agreement"] = {"n_true_positives": int(len(tp)), "overlap_top3": round(float(np.mean(o3)), 3), "chance_top3": round(3 / P, 3),
                  "overlap_top5": round(float(np.mean(o5)), 3), "overlap_top5_sd": round(float(np.std(o5)), 3), "chance_top5": round(K / P, 3),
                  "jaccard_top5": round(float(np.mean(jac)), 3), "spearman_abs": round(float(np.nanmean(rho)), 3),
                  "lime_three_seed_stability": round(float(np.mean(stab)), 3),
                  "shap_recompute_max_diff": float(np.abs(fraud_shap(explainer, Xte[tp[:5]]) - sv_te[tp[:5]]).max())}
 
# --- example claim: highest-probability true positive (Figures 4.7, 4.8)
j = int(tp[np.argmax(p_te[tp])]); R["example_claim"] = {"test_row": j, "prob": round(float(p_te[j]), 3)}
plt.figure(); shap.plots.waterfall(shap.Explanation(values=sv_te[j], base_values=base, data=Xte[j], feature_names=names),
                                   max_display=10, show=False)
plt.tight_layout(); plt.savefig(config.FIG / "fig_4_7_shap_waterfall.png", dpi=150, bbox_inches="tight"); plt.close()
lw = lime_main.explain_instance(Xte[j], rf.predict_proba, num_features=8, num_samples=config.LIME_SAMPLES).as_list()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.barh([a for a, _ in lw][::-1], [b for _, b in lw][::-1], color=["#c0504d" if b > 0 else "#1f4e79" for _, b in lw][::-1])
ax.set_xlabel("LIME weight (towards fraud)"); plt.tight_layout(); plt.savefig(config.FIG / "fig_4_8_lime.png", dpi=150); plt.close()
R["example_claim"]["shap_top5"] = [[names[i], round(float(sv_te[j][i]), 3)] for i in np.argsort(-np.abs(sv_te[j]))[:K]]
R["example_claim"]["lime_top5"] = [[a, round(float(b), 3)] for a, b in lw[:5]]
 
joblib.dump({"shap_test": sv_te, "shap_train": sv_tr, "base": base, "names": names, "Xte_model": Xte, "Xtr_model": Xtr},
            config.MODELS / "shap.joblib")
json.dump(R, open(config.OUT / "explain_results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
