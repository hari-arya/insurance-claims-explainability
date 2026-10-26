"""scripts/05_generate.py - generate LLM narratives (condition C3).
  python scripts/05_generate.py --pilot          # 10 TRAINING claims, to test prompts and cost (Objective O4)
  python scripts/05_generate.py                  # all flagged TEST claims x 2 generators x 4 prompts x 3 runs
  python scripts/05_generate.py --only G1        # one generator only (e.g. if Ollama is not ready yet)
Safe to stop and restart: finished calls are skipped."""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.explanations import ClaimBank, prompt, final_text
from src.llm_clients import generate
from src import store
 
ap = argparse.ArgumentParser()
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--pilot-claims", type=int, default=10)
ap.add_argument("--only", choices=tuple(config.GENERATORS), default=None)
args = ap.parse_args()
if args.pilot_claims < 1:
    ap.error("--pilot-claims must be at least 1")
bank = ClaimBank()
split = "train" if args.pilot else "test"
claims = bank.flagged(split)[:args.pilot_claims] if args.pilot else bank.flagged(split)
path = store.PILOT if args.pilot else store.GENERATIONS
done = {
    (record["idx"], record["gen"], record["prompt"], record["run"])
    for record in store.read(path)
    if record.get("prompt_version") == config.PROMPT_VERSION
    and record.get("model") == config.GENERATORS.get(record.get("gen"), {}).get("model")
}
gens = [args.only] if args.only else list(config.GENERATORS)
total = len(claims) * len(gens) * len(config.PROMPTS) * config.RUNS
print(f"{split} claims: {len(claims)}  -> {total} generations ({len(done)} already done)")
n = 0
for i in claims:
    c = bank.claim(i, split)
    for g in gens:
        for p in config.PROMPTS:
            system, user = prompt(c, p)
            for run in range(config.RUNS):
                n += 1
                if (i, g, p, run) in done: continue
                t = time.time()
                raw = generate(g, system, user, config.RUN_SEEDS[run])
                store.append(path, {"idx": i, "split": split, "gen": g, "model": config.GENERATORS[g]["model"], "prompt": p,
                                    "run": run, "seed": config.RUN_SEEDS[run],
                                    "seed_applied": config.GENERATORS[g]["provider"] in {"gemini", "ollama"},
                                    "raw": raw, "text": final_text(p, raw), "seconds": round(time.time() - t, 2),
                                    "prompt_version": config.PROMPT_VERSION,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                print(f"[{n}/{total}] claim {i} {g} {p} run {run}  ({time.time() - t:.1f}s)")

                if config.GENERATORS[g]["provider"] == "gemini":
                    time.sleep(4.0)

print("Finished. Output:", path)
