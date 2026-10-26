"""scripts/04_baselines.py - C0, C1, C2 texts and the exact prompts for every flagged test claim
(Thesis Section 4.6). No API needed.   Run:  python scripts/04_baselines.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import config
from src.explanations import ClaimBank, C0, C1, C2, prompt, user_block
from src.text_eval import fidelity
 
bank = ClaimBank(); flagged = bank.flagged("test")
records, fid = [], []
for i in flagged:
    c = bank.claim(i)
    f = fidelity(c, C2(c), bank.names); fid.append(f)
    records.append({"idx": i, "prob": round(c["prob"], 3), "true_label": c["label"], "C0": C0(c), "C1": C1(c), "C2": C2(c),
                    "C2_fidelity": f, "input_block": user_block(c)})
json.dump(records, open(config.LLM_OUT / "baselines.json", "w"), indent=1, default=float)
tp = sum(r["true_label"] for r in records)
summary = {"flagged": len(records), "true_positives": tp, "false_positives": len(records) - tp,
           "C2_driver_recall": float(np.mean([f["driver_recall"] for f in fid])),
           "C2_sign_accuracy": float(np.nanmean([f["sign_accuracy"] for f in fid])),
           "C2_words_mean": round(float(np.mean([f["words"] for f in fid])), 1),
           "C2_words_range": [int(min(f["words"] for f in fid)), int(max(f["words"] for f in fid))]}
json.dump(summary, open(config.OUT / "baselines_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
example = bank.claim(flagged[0])
print("\nExample input block (paste into Thesis Section 4.6):\n" + user_block(example) + "\n\nC2:\n" + C2(example))
