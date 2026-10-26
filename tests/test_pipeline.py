"""tests/test_pipeline.py - quick checks.  Run:  pytest -q"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pytest
import config
from src.data_loader import load_raw, prepare
 
 
def test_data_shape_and_labels():
    df = load_raw()
    assert df.shape[0] == 1000
    assert (df["fraud_reported"] == "Y").sum() == 247
    assert (df["authorities_contacted"] == "None").sum() == 91     # "None" must not become missing
 
 
def test_no_question_marks_after_prepare():
    X, y, cat, num = prepare(load_raw())
    assert not (X == "?").any().any()
    assert "policy_tenure_days" in X.columns
 
 
@pytest.mark.skipif(not (config.MODELS / "trained.joblib").exists(), reason="run 02_train.py first")
def test_train_and_test_do_not_overlap():
    A = joblib.load(config.MODELS / "trained.joblib")
    assert set(A["X_train"].index).isdisjoint(A["X_test"].index)
    assert len(A["y_test"]) == 200
 
 
@pytest.mark.skipif(not (config.MODELS / "shap.joblib").exists(), reason="run 03_explain.py first")
def test_template_is_fully_faithful():
    from src.explanations import ClaimBank, C2
    from src.text_eval import fidelity
    bank = ClaimBank()
    for i in bank.flagged("test")[:10]:
        c = bank.claim(i); f = fidelity(c, C2(c), bank.names)
        assert f["driver_recall"] == 1.0 and f["sign_accuracy"] == 1.0
 
 
def test_fidelity_detects_wrong_direction():
    from src.text_eval import fidelity
    claim = {"drivers": [{"feature": "incident_severity_Major Damage", "shap": 0.3}]}
    good = fidelity(claim, "Major damage raises the fraud score.", [])
    bad = fidelity(claim, "Major damage lowers the fraud score.", [])
    assert good["sign_accuracy"] == 1.0 and bad["sign_accuracy"] == 0.0