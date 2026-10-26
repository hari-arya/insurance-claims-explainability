"""scripts/06_judge.py - Claude Sonnet 4.6 rates every explanation (C0, C1, C2 and all C3 narratives) for
clarity, actionability and unsupported statements (Thesis Section 3.13).   Run:  python scripts/06_judge.py"""
import sys
import argparse
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.explanations import ClaimBank, C0, C1, C2, user_block
from src.llm_clients import NonRetryableGenerationError, claude
from src.text_eval import JUDGE_JSON_SCHEMA, RUBRIC, parse_judge
from src import store
 
parser = argparse.ArgumentParser(description="Judge baseline and C3 explanations.")
parser.add_argument("--pilot", action="store_true", help="Judge pilot generations on training claims.")
args = parser.parse_args()

bank = ClaimBank()
generation_path = store.PILOT if args.pilot else store.GENERATIONS
judgement_path = store.PILOT_JUDGEMENTS if args.pilot else store.JUDGEMENTS
generations = [
    record for record in store.read(generation_path)
    if record.get("prompt_version") == config.PROMPT_VERSION
    and record.get("model") == config.GENERATORS.get(record.get("gen"), {}).get("model")
]
split = "train" if args.pilot else "test"
items = []
claim_ids = sorted({g["idx"] for g in generations}) if args.pilot else bank.flagged(split)
for i in claim_ids:
    c = bank.claim(i, split)
    for cond, fn in [("C0", C0), ("C1", C1), ("C2", C2)]:
        items.append((i, cond, 0, user_block(c), fn(c)))
for g in generations:
    items.append((g["idx"], f"C3-{g['prompt']}-{g['gen']}", g["run"],
                  user_block(bank.claim(g["idx"], split)), g["text"]))
done = {
    (record["idx"], record["cond"], record["run"])
    for record in store.read(judgement_path)
    if record.get("prompt_version") == config.PROMPT_VERSION
    and record.get("judge_model") == config.JUDGE_MODEL
}
print(len(items), "explanations to judge;", len(done), "already done", "(pilot)" if args.pilot else "")
for k, (idx, cond, run, inp, text) in enumerate(items, 1):
    if (idx, cond, run) in done: continue
    for attempt in range(3):
        try:
            res = parse_judge(claude(config.JUDGE_MODEL, RUBRIC, f"INPUT\n{inp}\n\nEXPLANATION\n{text}", seed=0,
                                    max_tokens=config.JUDGE_MAX_TOKENS,
                                    json_schema=JUDGE_JSON_SCHEMA)); break
        except NonRetryableGenerationError:
            raise
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise
            if attempt == 2:
                raise
            print("   retrying:", e); res = None
    store.append(judgement_path, {"idx": idx, "cond": cond, "run": run,
                                  "judge_model": config.JUDGE_MODEL,
                                  "seed_requested": 0,
                                  "seed_applied": False,
                                  "temperature": config.TEMPERATURE,
                                  "prompt_version": config.PROMPT_VERSION, **res})
    print(f"[{k}/{len(items)}] {cond} claim {idx}: clarity {res['clarity']}, actionability {res['actionability']}, UCR {res['ucr']:.2f}")

if args.pilot:
    totals = defaultdict(lambda: {"n": 0, "clarity": 0, "actionability": 0, "ucr": 0.0})
    for result in store.read(judgement_path):
        if (
            result.get("prompt_version") != config.PROMPT_VERSION
            or result.get("judge_model") != config.JUDGE_MODEL
        ):
            continue
        if result["cond"] == "C2" or result["cond"].startswith("C3-"):
            row = totals[result["cond"]]
            row["n"] += 1
            row["clarity"] += result["clarity"]
            row["actionability"] += result["actionability"]
            row["ucr"] += result["ucr"]
    print("\nPilot mean ratings (training claims; directional review only):")
    for cond, row in sorted(totals.items()):
        n = row["n"]
        print(f"{cond}: clarity {row['clarity'] / n:.2f}, "
              f"actionability {row['actionability'] / n:.2f}, UCR {row['ucr'] / n:.3f}")
