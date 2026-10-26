"""src/store.py - tiny helpers to append and read JSON-lines files (one record per line, safe to resume)."""
import json
import config
 
GENERATIONS = config.LLM_OUT / "generations.jsonl"
PILOT = config.LLM_OUT / "pilot_generations.jsonl"
JUDGEMENTS = config.LLM_OUT / "judgements.jsonl"
PILOT_JUDGEMENTS = config.LLM_OUT / "pilot_judgements.jsonl"
 
 
def read(path):
    if not path.exists(): return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
 
 
def append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
