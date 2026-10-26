"""scripts/02_train.py - split, RFECV, tune four classifiers, choose thresholds, evaluate on the test set,
run the imbalance ablation, and save everything (Thesis Sections 4.2-4.4; Tables 3.1, 4.2, 4.3, 4.4;
Figures 4.2, 4.3, 4.4).   Run:  python scripts/02_train.py      (takes about 5-15 minutes)"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import config
from src.data_loader import load_raw, prepare
from src import modelling as M
 
t0 = time.time()
X, y, cat_cols, num_cols = prepare(load_raw())
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
R = {"split": {"train": len(y_tr), "test": len(y_te), "train_fraud": int(y_tr.sum()), "test_fraud": int(y_te.sum())},
     "n_predictors": X.shape[1]}
print("Split:", R["split"])
 
# --- RFECV (Section 4.2, Figure 4.2)
pre = M.make_preprocessor(X_tr, cat_cols, num_cols).fit(X_tr)
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
Xtr_enc = pre.transform(X_tr); R["n_encoded"] = Xtr_enc.shape[1]
print("Running RFECV on", Xtr_enc.shape[1], "encoded columns ...")
rfecv = M.run_rfecv(Xtr_enc, y_tr)
mask = rfecv.support_; selected = [n for n, s in zip(names, mask) if s]
scores = rfecv.cv_results_["mean_test_score"]; se = rfecv.cv_results_["std_test_score"] / np.sqrt(config.CV_FOLDS)
nfeat = np.arange(5, 5 + len(scores)); k_1se = int(nfeat[np.argmax(scores >= scores.max() - se[scores.argmax()])])
R.update({"rfecv_n_selected": int(mask.sum()), "rfecv_best_cv_pr_auc": round(float(scores.max()), 3),
          "rfecv_k_one_se": k_1se, "selected_features": selected})
print("RFECV kept", mask.sum(), "features:", selected)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(nfeat, scores, color="#1f4e79"); ax.fill_between(nfeat, scores - se, scores + se, alpha=.2, color="#1f4e79")
ax.axvline(mask.sum(), ls="--", color="#c0504d", label=f"max CV PR-AUC (k={mask.sum()})")
ax.axvline(k_1se, ls=":", color="#555", label=f"one-SE rule (k={k_1se})")
ax.set_xlabel("Number of encoded features retained"); ax.set_ylabel("5-fold CV PR-AUC"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(config.FIG / "fig_4_2_rfecv.png", dpi=150); plt.close()
 
# --- tuning (Table 3.1)
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
best, R["tuning"] = {}, {}
for name, (est, grid) in M.base_models().items():
    print("Tuning", name, "...")
    best[name], params, score = M.tune(name, est, grid, pre, cat_idx, mask, X_tr, y_tr)
    R["tuning"][name] = {"best_params": params, "cv_pr_auc": round(score, 3)}
best["Voting"] = M.voting(best, pre, cat_idx, mask)
 
# --- thresholds from out-of-fold predictions, then ONE evaluation on the test set (Table 4.2)
R["test"], probs = {}, {}
for name, model in best.items():
    thr, oof_ap = M.oof_threshold(model, X_tr, y_tr)
    model.fit(X_tr, y_tr); p = model.predict_proba(X_te)[:, 1]; probs[name] = p
    R["test"][name] = {"tuned": M.metrics(y_te, p, thr), "at_0.5": M.metrics(y_te, p, 0.5), "oof_pr_auc": round(oof_ap, 3)}
    print(f"{name}: F1={R['test'][name]['tuned']['f1']}  PR-AUC={R['test'][name]['tuned']['pr_auc']}")
 
# --- imbalance ablation for the RF (Table 4.4)
rf_params = R["tuning"]["RF"]["best_params"]; Xte_enc = pre.transform(X_te)
R["ablation_rf"] = {}
for tag, model, Xa, Xb in [
        ("no_resampling", RandomForestClassifier(**rf_params, random_state=config.SEED, n_jobs=-1), Xtr_enc[:, mask], Xte_enc[:, mask]),
        ("class_weight", RandomForestClassifier(**rf_params, class_weight="balanced_subsample", random_state=config.SEED, n_jobs=-1), Xtr_enc[:, mask], Xte_enc[:, mask])]:
    p = model.fit(Xa, y_tr).predict_proba(Xb)[:, 1]
    R["ablation_rf"][tag] = {"pr_auc": round(average_precision_score(y_te, p), 3), "roc_auc": round(roc_auc_score(y_te, p), 3),
                             "f1_at_0.5": round(f1_score(y_te, (p >= .5).astype(int)), 3)}
p = probs["RF"]
R["ablation_rf"]["smote_nc"] = {"pr_auc": round(average_precision_score(y_te, p), 3), "roc_auc": round(roc_auc_score(y_te, p), 3),
                                "f1_at_0.5": round(f1_score(y_te, (p >= .5).astype(int)), 3)}
 
# --- Figures 4.3 and 4.4
cm = np.array(R["test"][config.EXPLAINED_MODEL]["tuned"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(3.8, 3.2)); ax.imshow(cm, cmap="Blues")
for (a, b), v in np.ndenumerate(cm):
    ax.text(b, a, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=12)
ax.set_xticks([0, 1], ["Pred. N", "Pred. Y"]); ax.set_yticks([0, 1], ["Actual N", "Actual Y"])
plt.tight_layout(); plt.savefig(config.FIG / "fig_4_3_confusion_matrix.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for name, c in zip(["LR", "RF", "SVM", "Voting"], ["#7f7f7f", "#1f4e79", "#c0504d", "#548235"]):
    pr, rc, _ = precision_recall_curve(y_te, probs[name])
    ax.plot(rc, pr, color=c, label=f"{name} (AP={R['test'][name]['tuned']['pr_auc']:.2f})")
ax.axhline(y_te.mean(), ls="--", color="k", lw=.8, label=f"No-skill ({y_te.mean():.2f})")
ax.set_xlabel("Recall (fraud)"); ax.set_ylabel("Precision (fraud)"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
plt.tight_layout(); plt.savefig(config.FIG / "fig_4_4_pr_curves.png", dpi=150); plt.close()
 
# --- Table 4.2 as CSV, and save everything the next scripts need
pd.DataFrame([{"model": n, **{k: v for k, v in R["test"][n]["tuned"].items() if k != "confusion_matrix"}} for n in best]
             ).to_csv(config.TAB / "table_4_2_test_performance.csv", index=False)
joblib.dump({"models": best, "X_train": X_tr, "X_test": X_te, "y_train": y_tr.values, "y_test": y_te.values,
             "selected": selected, "cat_cols": cat_cols, "num_cols": num_cols, "probs": probs,
             "thresholds": {n: R["test"][n]["tuned"]["threshold"] for n in best}}, config.MODELS / "trained.joblib")
R["minutes"] = round((time.time() - t0) / 60, 1)
json.dump(R, open(config.OUT / "train_results.json", "w"), indent=1, default=str)
print("\nSaved: models/trained.joblib, outputs")
