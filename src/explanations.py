"""src/explanations.py - builds the inputs of every explanation condition (Thesis Sections 3.10-3.11):
claim serialiser, top-5 drivers, C0/C1/C2 baselines and prompts P0-P3. No model and no retrieval is used here."""
import numpy as np
import joblib
import config
 
LABEL = {  # encoded feature -> (readable name, how to show this claim's own value)
    "months_as_customer": ("time as a customer", lambda r: f"{int(r['months_as_customer'])} months"),
    "policy_annual_premium": ("annual premium", lambda r: f"${r['policy_annual_premium']:,.2f}"),
    "total_claim_amount": ("total claim amount", lambda r: f"${r['total_claim_amount']:,.0f}"),
    "injury_claim": ("injury component of the claim", lambda r: f"${r['injury_claim']:,.0f}"),
    "property_claim": ("property component of the claim", lambda r: f"${r['property_claim']:,.0f}"),
    "vehicle_claim": ("vehicle component of the claim", lambda r: f"${r['vehicle_claim']:,.0f}"),
    "policy_tenure_days": ("time between policy start and incident", lambda r: f"{int(r['policy_tenure_days']):,} days"),
    "insured_hobbies_chess": ("policyholder hobby recorded as chess", lambda r: "yes" if r["insured_hobbies"] == "chess" else "no"),
    "insured_hobbies_cross-fit": ("policyholder hobby recorded as cross-fit", lambda r: "yes" if r["insured_hobbies"] == "cross-fit" else "no"),
    "incident_severity_Major Damage": ("incident severity recorded as major damage",
                                       lambda r: "yes" if r["incident_severity"] == "Major Damage" else f"no ({r['incident_severity']})"),
}
# If your RFECV run selects different features, add a line above for each new feature name.
 
 
class ClaimBank:
    """Gives easy access to a claim, its probability and its SHAP drivers."""
    def __init__(self):
        T = joblib.load(config.MODELS / "trained.joblib"); S = joblib.load(config.MODELS / "shap.joblib")
        self.names = S["names"]; self.base = S["base"]
        self.thr = T["thresholds"][config.EXPLAINED_MODEL]
        self.sets = {"test": (T["X_test"], S["shap_test"], T["probs"][config.EXPLAINED_MODEL], T["y_test"]),
                     "train": (T["X_train"], S["shap_train"],
                               T["models"][config.EXPLAINED_MODEL].predict_proba(T["X_train"])[:, 1], T["y_train"])}
        missing = [n for n in self.names if n not in LABEL]
        if missing: raise KeyError(f"Add readable labels to LABEL in src/explanations.py for: {missing}")
 
    def flagged(self, split="test"):
        X, sv, p, y = self.sets[split]; return [int(i) for i in np.where(p >= self.thr)[0]]
 
    def claim(self, i, split="test"):
        X, sv, p, y = self.sets[split]; r = X.iloc[i]; order = np.argsort(-np.abs(sv[i]))[:config.TOP_K]
        drivers = [{"feature": self.names[j], "label": LABEL[self.names[j]][0], "value": LABEL[self.names[j]][1](r),
                    "shap": round(float(sv[i][j]), 4), "direction": "raises" if sv[i][j] > 0 else "lowers"} for j in order]
        return {"idx": int(i), "split": split, "prob": float(p[i]), "threshold": float(self.thr), "label": int(y[i]),
                "drivers": drivers, "row": r}
 
 
def serialise(c):
    """Fixed template: the claim's own attributes as four sentences (Thesis Section 4.6 shows an example)."""
    r = c["row"]; unk = lambda v: str(v).replace("UNKNOWN", "not recorded")
    return "\n".join([
        f"Policy: state {r['policy_state']}, combined single limit {r['policy_csl']}, deductible ${r['policy_deductable']:,.0f}, "
        f"annual premium ${r['policy_annual_premium']:,.2f}, umbrella limit ${r['umbrella_limit']:,.0f}, "
        f"customer for {int(r['months_as_customer'])} months, policy in force {int(r['policy_tenure_days']):,} days at the incident date.",
        f"Policyholder: age {int(r['age'])}, {str(r['insured_sex']).lower()}, education {r['insured_education_level']}, "
        f"occupation {r['insured_occupation']}, hobby {r['insured_hobbies']}.",
        f"Incident: {r['incident_type']}, collision type {unk(r['collision_type'])}, severity {r['incident_severity']}, "
        f"{int(r['number_of_vehicles_involved'])} vehicle(s), {int(r['bodily_injuries'])} bodily injur(ies), {int(r['witnesses'])} witness(es), "
        f"authorities contacted: {r['authorities_contacted']}, police report available: {unk(r['police_report_available'])}, "
        f"property damage: {unk(r['property_damage'])}, hour {int(r['incident_hour_of_the_day'])}:00, {r['incident_state']}.",
        f"Claim: total ${r['total_claim_amount']:,.0f} (injury ${r['injury_claim']:,.0f}, property ${r['property_claim']:,.0f}, "
        f"vehicle ${r['vehicle_claim']:,.0f}); vehicle {r['auto_make']} {int(r['auto_year'])}."])
 
 
def drivers_text(c):
    return "\n".join(f"{k + 1}. {d['label']} (this claim: {d['value']}) - {d['direction']} the fraud score (SHAP {d['shap']:+.3f})"
                     for k, d in enumerate(c["drivers"]))
 
 
def user_block(c):
    return (f"Fraud risk score: {c['prob']:.2f} (referral threshold {c['threshold']:.2f}).\n\nCLAIM DETAILS\n{serialise(c)}\n\n"
            f"TOP FIVE MODEL FACTORS (SHAP, largest first)\n{drivers_text(c)}")
 
 
def band(p): return "HIGH" if p >= 0.8 else "ELEVATED" if p >= 0.6 else "MODERATE"
 
# ---------------- non-LLM baselines (Table 3.2)
def C0(c): return f"Fraud risk score {c['prob']:.2f} (threshold {c['threshold']:.2f}); risk band {band(c['prob'])}. Claim referred for review."
def C1(c): return C0(c) + "\nTop factors (SHAP, fraud probability scale):\n" + "\n".join(f"{d['feature']}: {d['shap']:+.3f}" for d in c["drivers"])
def C2(c):
    up = [d for d in c["drivers"] if d["shap"] > 0]; dn = [d for d in c["drivers"] if d["shap"] <= 0]
    s = [f"This claim has a fraud risk score of {c['prob']:.2f}, above the referral threshold of {c['threshold']:.2f}."]
    if up: s.append("The factors raising the score are: " + "; ".join(f"{d['label']} ({d['value']})" for d in up) + ".")
    if dn: s.append("The factors lowering the score are: " + "; ".join(f"{d['label']} ({d['value']})" for d in dn) + ".")
    s.append(f"Recommended check: verify the evidence behind the strongest factor, {c['drivers'][0]['label']}.")
    return " ".join(s)
 
# ---------------- prompts (Table 3.3, Appendix B)
OUTPUT_CONTRACT = (
    f"Write {config.WORD_RANGE[0]}-{config.WORD_RANGE[1]} words in plain English for a claims handler. "
    "Use only the supplied claim details and top-five SHAP factors. Do not add outside facts; treat these inputs as the "
    "complete evidence set. Do not invent "
    "people, events, records, statements, comparisons, results, or missing details. Do not call the claim suspicious or infer "
    "fraud, intent, or wrongdoing. State that the model score is a risk estimate, not a finding. SHAP describes how model "
    "features move the score; it does not prove causation or wrongdoing. Mention each of the five supplied factors and correctly state "
    "whether it raises or lowers the score. Explain the strongest factors first, without implying that a larger contribution "
    "is proof. Do not recommend investigating a person based only on a personal characteristic such as hobby, age, sex, "
    "education, or occupation; if such a feature appears, describe it only as a model input and direct any check to data "
    "coding or model governance. End with exactly one 'Recommended check:' sentence. Give one concrete proposed review tied "
    "to a supplied claim-related driver, name the evidence or record to review and what to compare or confirm, and do not imply the "
    "record exists or a discrepancy was found. If no grounded check is possible, state what information would be needed."
)
SYSTEM_P1 = (
    "You are a careful insurance-claims explanation writer. Explain the model output to a non-specialist claims handler. "
    + OUTPUT_CONTRACT
)
P0_INSTRUCTION = (
    "Explain in plain English why the model flagged this claim. Follow this output contract exactly: "
    + OUTPUT_CONTRACT
)
P2_ADDITION = (
    "\n\nBefore writing, privately verify each supplied factor's value and direction against the input. Do not show this "
    "check, chain of thought, notes, or headings. Return only the final explanation following the output contract, with "
    "the single recommended check as its final sentence."
)
EXEMPLARS = [  # written by the researcher for TRAINING claims only - rewrite in your own words after your pilot
    ("Fraud risk score: 0.95 (referral threshold 0.50).\nTOP FIVE MODEL FACTORS\n1. policyholder hobby recorded as cross-fit (yes) - raises (+0.221)\n"
     "2. incident severity recorded as major damage (yes) - raises (+0.177)\n3. vehicle component of the claim ($38,500) - raises (+0.017)\n"
     "4. annual premium ($1,306.00) - raises (+0.016)\n5. time between policy start and incident (5,491 days) - raises (+0.015)",
     "The model estimates a risk score of 0.95, above the 0.50 referral threshold; this is not a finding of fraud. The recorded "
     "cross-fit hobby and major-damage severity raise the score most. The vehicle component, annual premium and policy tenure "
     "also raise it, with smaller contributions. These SHAP values describe how model inputs move the score, not causes or "
     "evidence of wrongdoing. A hobby is a model input, not evidence about a person's conduct. Recommended check: compare the "
     "major-damage classification with the damage assessment; the input does not establish whether that record exists or what it shows."),
    ("Fraud risk score: 0.88 (referral threshold 0.50).\nTOP FIVE MODEL FACTORS\n1. policyholder hobby recorded as cross-fit (yes) - raises (+0.481)\n"
     "2. incident severity recorded as major damage (no (Total Loss)) - lowers (-0.134)\n3. policyholder hobby recorded as chess (no) - lowers (-0.023)\n"
     "4. time between policy start and incident (6,504 days) - raises (+0.021)\n5. total claim amount ($78,650) - raises (+0.014)",
     "The model estimates a risk score of 0.88, above the 0.50 referral threshold; this is not a finding of fraud. The recorded "
     "cross-fit hobby raises the score most. Total-loss severity and the absence of the recorded chess hobby lower it, while policy "
     "tenure and total claim amount raise it slightly. SHAP values describe model-score movement, not causes or wrongdoing. Hobby "
     "features are model inputs, not evidence about a person's conduct. Recommended check: compare the total-loss classification "
     "with the claim valuation assessment; the input does not establish whether that record exists or what it shows."),
]
 
 
def prompt(c, p):
    """Return (system_message, user_message) for prompt design p."""
    if p == "P0":
        return None, P0_INSTRUCTION + "\n\n" + user_block(c)
    if p == "P1":
        return SYSTEM_P1, user_block(c)
    if p == "P2":
        return SYSTEM_P1, user_block(c) + P2_ADDITION
    if p == "P3":
        shots = "\n\n".join(f"EXAMPLE {k + 1}\n{q}\nNOTE: {a}" for k, (q, a) in enumerate(EXEMPLARS))
        return SYSTEM_P1, shots + "\n\nNOW WRITE THE NOTE FOR THIS CLAIM\n" + user_block(c)
    raise ValueError(p)
 
 
def final_text(p, raw):
    """For P2 only the EXPLANATION section is evaluated."""
    if p == "P2" and "EXPLANATION" in raw:
        return raw.split("EXPLANATION", 1)[1].lstrip(" :*#\n")
    return raw.strip()
