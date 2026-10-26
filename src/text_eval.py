"""src/text_eval.py - automatic fidelity checks and the judge rubric (Thesis Section 3.13)."""
import re, json
import numpy as np
 
# how each driver may be mentioned in a narrative (extend after reading your pilot outputs)
SURFACE = {
    "months_as_customer": ["months as a customer", "time as a customer", "customer for", "customer tenure", "relationship length", "months with"],
    "policy_annual_premium": ["premium"],
    "total_claim_amount": ["total claim", "claim amount", "overall claim", "total amount", "size of the claim", "claimed amount"],
    "injury_claim": ["injury"],
    "property_claim": ["property"],
    "vehicle_claim": ["vehicle claim", "vehicle component", "vehicle damage claim", "vehicle portion", "vehicle amount"],
    "policy_tenure_days": ["policy start", "policy tenure", "policy age", "days between", "since the policy", "time on policy",
                           "policy was taken out", "policy inception", "in force", "bind"],
    "insured_hobbies_chess": ["chess"],
    "insured_hobbies_cross-fit": ["cross-fit", "crossfit", "cross fit"],
    "incident_severity_Major Damage": ["major damage", "severity"],
}
UP = r"(rais|increas|push|add|elevat|contribut\w* to (the )?(higher|fraud)|higher risk|toward(s)? fraud|point(s|ing)? to fraud|strengthen)"
DOWN = r"(lower|reduc|decreas|offset|mitigat|pull\w* (the )?(risk|score) down|away from fraud|counter|weaken|less likely)"
 
 
def mentioned(text, feature):
    return any(s in text.lower() for s in SURFACE[feature])
 
 
def stated_sign(text, feature):
    """+1 if the sentence mentioning the feature says it raises the score, -1 if it lowers it, 0 if unclear."""
    for sentence in re.split(r"(?<=[.;])\s+", text.lower()):
        if any(s in sentence for s in SURFACE[feature]):
            up, down = re.search(UP, sentence), re.search(DOWN, sentence)
            if up and not down: return 1
            if down and not up: return -1
    return 0
 
 
def fidelity(claim, text, all_features):
    """Driver recall (Eq. 3.18) and sign accuracy (Eq. 3.19)."""
    D5 = [d["feature"] for d in claim["drivers"]]
    ment = [f for f in D5 if mentioned(text, f)]
    truth = {d["feature"]: (1 if d["shap"] > 0 else -1) for d in claim["drivers"]}
    judged = [(stated_sign(text, f), truth[f]) for f in ment]; judged = [(s, t) for s, t in judged if s != 0]
    return {"driver_recall": len(ment) / len(D5),
            "sign_accuracy": float(np.mean([s == t for s, t in judged])) if judged else np.nan,
            "signs_stated": len(judged),
            "non_driver_features_mentioned": sum(mentioned(text, f) for f in all_features if f not in D5),
            "words": len(text.split())}
 
 
RUBRIC = """You are evaluating an explanation written for an insurance claims handler about why a fraud model flagged a claim.
You are given the INPUT (the only facts the writer was allowed to use) and the EXPLANATION.
Return JSON only, with keys:
 "claims": list of objects {"text": <one atomic factual statement from the explanation>, "supported": true|false}
            (supported = the statement is entailed by the INPUT; general advice such as the recommended check counts as supported),
 "clarity": integer 1-5  (5 = a non-specialist handler understands it fully without help; 1 = unintelligible or pure jargon),
 "actionability": integer 1-5 (5 = states clearly what to verify next and why; 1 = gives no usable next step),
 "rationale": one sentence.
Score the explanation on its own; do not compare with any other text."""

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["text", "supported"],
                "additionalProperties": False,
            },
        },
        "clarity": {"type": "integer"},
        "actionability": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["claims", "clarity", "actionability", "rationale"],
    "additionalProperties": False,
}


def parse_judge(raw):
    decoder = json.JSONDecoder()
    js = None
    for match in re.finditer(r"\{", raw):
        try:
            candidate, _ = decoder.raw_decode(raw, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            js = candidate
            break
    if js is None:
        raise ValueError("Judge response did not contain a valid JSON object.")
    for key in ("claims", "clarity", "actionability", "rationale"):
        if key not in js:
            raise ValueError(f"Judge response is missing required field: {key}")
    if not isinstance(js["claims"], list) or any(
        not isinstance(claim, dict)
        or not isinstance(claim.get("text"), str)
        or not isinstance(claim.get("supported"), bool)
        for claim in js["claims"]
    ):
        raise ValueError("Judge response 'claims' must be a list of text/supported objects.")
    for key in ("clarity", "actionability"):
        if type(js[key]) is not int or not 1 <= js[key] <= 5:
            raise ValueError(f"Judge response '{key}' must be an integer from 1 to 5.")
    if not isinstance(js["rationale"], str):
        raise ValueError("Judge response 'rationale' must be a string.")
    n = len(js["claims"]); uns = sum(not c["supported"] for c in js["claims"])
    return {"clarity": int(js["clarity"]), "actionability": int(js["actionability"]), "n_claims": n,
            "n_unsupported": uns, "ucr": uns / n if n else 0.0, "claims": js["claims"], "rationale": js.get("rationale", "")}
