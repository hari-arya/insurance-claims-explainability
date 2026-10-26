"""scripts/08_human_rating.py - validate the LLM judge against human raters (Thesis Section 3.13, Eq. 3.24).
  python scripts/08_human_rating.py export   -> outputs/human/rater_1.csv, rater_2.csv, ucr_check.csv (blind)
    (each rater fills clarity and actionability 1-5; fill supported_by_you in ucr_check.csv without viewing its private key)
  python scripts/08_human_rating.py score    -> outputs/human_validation.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
import config
from src.explanations import ClaimBank, C0, C1, C2, user_block
from src import store
 
HUMAN = config.OUT / "human"; HUMAN.mkdir(exist_ok=True)
J = pd.DataFrame(store.read(store.JUDGEMENTS))
if sys.argv[1:] == ["export"]:
    gen = {(g["idx"], f"C3-{g['prompt']}-{g['gen']}", g["run"]): g["text"] for g in store.read(store.GENERATIONS)}
    bank = ClaimBank()
    def text_of(r):
        if r["cond"].startswith("C3"): return gen[(r["idx"], r["cond"], r["run"])]
        return {"C0": C0, "C1": C1, "C2": C2}[r["cond"]](bank.claim(r["idx"]))
    sample = J.groupby("cond").sample(frac=config.HUMAN_SAMPLE_SHARE, random_state=config.SEED)
    sample = sample.sample(frac=1, random_state=config.SEED).reset_index(drop=True)      # shuffle: raters must not see the condition
    sample["item_id"] = range(1, len(sample) + 1)
    sheet = pd.DataFrame({"item_id": sample["item_id"], "input": [user_block(bank.claim(i)) for i in sample["idx"]],
                          "explanation": [text_of(r) for _, r in sample.iterrows()], "clarity": "", "actionability": ""})
    for r in (1, 2): sheet.to_csv(HUMAN / f"rater_{r}.csv", index=False)
    sample[["item_id", "idx", "cond", "run", "clarity", "actionability"]].to_csv(HUMAN / "key_do_not_share.csv", index=False)
    claims = [{"idx": r["idx"], "cond": r["cond"], "run": r["run"], "input": user_block(bank.claim(r["idx"])),
               "statement": c["text"], "judge_supported": c["supported"],
               "supported_by_you": ""} for _, r in J[J["cond"].str.startswith("C3")].iterrows() for c in r["claims"]]
    ucr = pd.DataFrame(claims).sample(n=min(200, len(claims)), random_state=config.SEED).reset_index(drop=True)
    ucr["item_id"] = range(1, len(ucr) + 1)
    ucr[["item_id", "input", "statement", "supported_by_you"]].to_csv(HUMAN / "ucr_check.csv", index=False)
    ucr[["item_id", "judge_supported"]].to_csv(HUMAN / "ucr_key_do_not_share.csv", index=False)
    print(f"Exported {len(sheet)} items for each rater and {min(200, len(claims))} statements for the hand-check to {HUMAN}")
elif sys.argv[1:] == ["score"]:
    key = pd.read_csv(HUMAN / "key_do_not_share.csv"); r1 = pd.read_csv(HUMAN / "rater_1.csv"); r2 = pd.read_csv(HUMAN / "rater_2.csv")
    problems = []
    for name, frame in (("rater_1.csv", r1), ("rater_2.csv", r2), ("key_do_not_share.csv", key)):
        for column in ("clarity", "actionability"):
            scores = pd.to_numeric(frame[column], errors="coerce")
            missing = scores.isna()
            invalid = scores.notna() & ((scores < 1) | (scores > 5) | (scores % 1 != 0))
            if missing.any():
                problems.append(f"{name}: {column} has {int(missing.sum())} blank or non-numeric value(s)")
            if invalid.any():
                problems.append(f"{name}: {column} has {int(invalid.sum())} value(s) outside integer range 1-5")
    if problems:
        raise SystemExit("Cannot score human ratings yet. Complete all 1-5 ratings, then rerun:\n- " + "\n- ".join(problems))

    out = {"n_items": len(key)}
    for m in ["clarity", "actionability"]:
        h1, h2, j = r1[m].astype(int), r2[m].astype(int), key[m].astype(int)
        human = np.rint((h1 + h2) / 2).astype(int)
        out[m] = {"kappa_judge_vs_rater1": cohen_kappa_score(j, h1, weights="quadratic"),
                  "kappa_judge_vs_rater2": cohen_kappa_score(j, h2, weights="quadratic"),
                  "kappa_judge_vs_mean_human": cohen_kappa_score(j, human, weights="quadratic"),
                  "kappa_rater1_vs_rater2": cohen_kappa_score(h1, h2, weights="quadratic")}
        out[m]["meets_threshold"] = out[m]["kappa_judge_vs_mean_human"] >= config.KAPPA_THRESHOLD
    u = pd.read_csv(HUMAN / "ucr_check.csv").merge(
        pd.read_csv(HUMAN / "ucr_key_do_not_share.csv"), on="item_id", validate="one_to_one"
    )
    u = u[u["supported_by_you"].astype(str).str.lower().isin(["true", "false", "yes", "no", "1", "0"])]
    yours = u["supported_by_you"].astype(str).str.lower().isin(["true", "yes", "1"])
    out["ucr_hand_check"] = {"n": int(len(u)), "agreement_with_judge": float((yours == u["judge_supported"].astype(bool)).mean()) if len(u) else None}
    json.dump(out, open(config.OUT / "human_validation.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
else:
    print(__doc__)
