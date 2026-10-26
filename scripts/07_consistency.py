"""scripts/07_consistency.py - BERTScore-F1 and driver Jaccard across the 3 runs (Eq. 3.21-3.23).
The first run downloads a language model for BERTScore (about 1.4 GB). Run: python scripts/07_consistency.py"""
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from bert_score import BERTScorer

import config
from src import store
from src.explanations import ClaimBank
from src.text_eval import mentioned

ROBERTA_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--roberta-large"


def has_local_roberta() -> bool:
    if not ROBERTA_CACHE.exists():
        return False
    return any((ROBERTA_CACHE / "snapshots").glob("**/config.json"))


bank = ClaimBank()
G = pd.DataFrame(store.read(store.GENERATIONS))
rows = []

if not has_local_roberta():
    raise SystemExit(
        "Roberta-large is not available in the local Hugging Face cache. "
        "Download it first or rerun with internet access: "
        "python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('roberta-large')\""
    )

# Force offline use when the model is already cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
scorer = BERTScorer(lang="en")

for (idx, gen, p), grp in G.groupby(["idx", "gen", "prompt"]):
    texts = grp.sort_values("run")["text"].tolist()
    pairs = list(itertools.combinations(range(len(texts)), 2))

    try:
        _, _, F = scorer.score(
            [texts[a] for a, b in pairs],
            [texts[b] for a, b in pairs],
            verbose=False,
        )
    except Exception as e:
        raise SystemExit(
            "BERTScore failed while loading or scoring with the cached roberta-large model.\n"
            + str(e)
        ) from e

    c = bank.claim(idx)
    sets = [{d["feature"] for d in c["drivers"] if mentioned(t, d["feature"])} for t in texts]
    jac = np.mean([len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for a, b in pairs])
    rows.append({
        "idx": idx,
        "cond": f"C3-{p}-{gen}",
        "bertscore_f1": float(F.mean()),
        "driver_jaccard": float(jac),
    })

out = pd.DataFrame(rows)
out.to_csv(config.TAB / "consistency_by_claim.csv", index=False)
print(out.groupby("cond")[["bertscore_f1", "driver_jaccard"]].mean().round(3))
