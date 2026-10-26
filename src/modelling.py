"""src/modelling.py - preprocessing, SMOTE-NC, RFECV, the four classifiers, thresholds and metrics
(Thesis Sections 3.4-3.8)."""
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
import config
 
 
def make_preprocessor(X_train, cat_cols, num_cols):
    """Median-impute + standardise numbers; one-hot encode categories.
    Categories are fixed from the TRAINING data so every CV fold produces the same columns."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(categories=[sorted(X_train[c].unique()) for c in cat_cols],
                              handle_unknown="ignore", sparse_output=False), cat_cols)])
 
 
def cv_splitter():
    return StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
 
 
def run_rfecv(Xtr_encoded, y_train):
    """Recursive feature elimination with 5-fold CV, scored on PR-AUC (Thesis 3.6)."""
    rfecv = RFECV(RandomForestClassifier(n_estimators=200, random_state=config.SEED, n_jobs=-1, class_weight="balanced"),
                  step=1, cv=cv_splitter(), scoring="average_precision", min_features_to_select=5, n_jobs=-1)
    rfecv.fit(Xtr_encoded, y_train)
    return rfecv
 
 
class ColumnSelector:
    """Keeps only the RFECV-selected columns. A class (not a lambda) so the pipeline can be saved with joblib."""
    def __init__(self, mask): self.mask = np.asarray(mask)
    def __call__(self, Z): return Z[:, self.mask]
 
 
def smote_pipeline(estimator, preprocessor, cat_idx, mask):
    """SMOTE-NC -> encode/scale -> keep selected columns -> classifier.
    Because SMOTE-NC is inside the pipeline, it only ever sees training folds."""
    return ImbPipeline([
        ("smote", SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=config.SEED)),
        ("pre", clone(preprocessor)),
        ("sel", FunctionTransformer(ColumnSelector(mask))),
        ("clf", estimator)])
 
 
def base_models():
    return {
        "LR": (LogisticRegression(max_iter=5000, random_state=config.SEED), config.LR_GRID),
        "RF": (RandomForestClassifier(random_state=config.SEED, n_jobs=-1), config.RF_GRID),
        "SVM": (SVC(kernel="rbf", probability=True, random_state=config.SEED), config.SVM_GRID),
    }
 
 
def tune(name, estimator, grid, pre, cat_idx, mask, X_train, y_train):
    gs = GridSearchCV(smote_pipeline(estimator, pre, cat_idx, mask), grid, cv=cv_splitter(),
                      scoring="average_precision", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}, gs.best_score_
 
 
def voting(best, pre, cat_idx, mask):
    return smote_pipeline(VotingClassifier([("rf", best["RF"].named_steps["clf"]),
                                            ("svm", best["SVM"].named_steps["clf"])], voting="soft"),
                          pre, cat_idx, mask)
 
 
def oof_threshold(model, X_train, y_train):
    """Threshold that maximises F1 on out-of-fold training predictions (Equation 3.10)."""
    p = cross_val_predict(model, X_train, y_train, cv=cv_splitter(), method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])]), float(average_precision_score(y_train, p))
 
 
def bootstrap_ci(y_true, p, y_hat):
    rng = np.random.default_rng(config.SEED); y_true = np.asarray(y_true); f1s, aps = [], []
    for _ in range(config.N_BOOTSTRAP):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].sum() == 0: continue
        f1s.append(f1_score(y_true[i], y_hat[i])); aps.append(average_precision_score(y_true[i], p[i]))
    return ([round(float(np.percentile(f1s, q)), 3) for q in (2.5, 97.5)],
            [round(float(np.percentile(aps, q)), 3) for q in (2.5, 97.5)])
 
 
def metrics(y_true, p, threshold):
    y_hat = (p >= threshold).astype(int)
    ci_f1, ci_ap = bootstrap_ci(y_true, p, y_hat)
    return {"threshold": round(threshold, 3), "precision": round(precision_score(y_true, y_hat, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_hat), 3), "f1": round(f1_score(y_true, y_hat), 3), "f1_ci": ci_f1,
            "pr_auc": round(average_precision_score(y_true, p), 3), "pr_auc_ci": ci_ap,
            "roc_auc": round(roc_auc_score(y_true, p), 3), "mcc": round(matthews_corrcoef(y_true, y_hat), 3),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_hat), 3),
            "accuracy": round(float((y_hat == np.asarray(y_true)).mean()), 3),
            "confusion_matrix": confusion_matrix(y_true, y_hat).tolist()}
