"""src/llm_clients.py - one small function per language model provider (Thesis Section 3.12)."""
import os, time
import requests
from dotenv import load_dotenv
import config
 
load_dotenv(config.ROOT / ".env")
 

class NonRetryableGenerationError(RuntimeError):
    pass


def gemini(model, system, user, seed):
    from google import genai
    from google.genai import types
    key = os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY is missing from .env")
    client = genai.Client(api_key=key)
    # Relax safety constraints so dataset keywords do not trip a silent block
    safety = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE, # Using Enum prevents payload parsing errors
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
    ]

    cfg = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=min(1200, config.MAX_TOKENS * 3),
        temperature=0.8,
        seed=seed,
        safety_settings=safety,
    )
    
    return client.models.generate_content(model=model, contents=user, config=cfg).text
 
 
def ollama(model, system, user, seed):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    r = requests.post(config.OLLAMA_URL, json={"model": model, "messages": msgs, "stream": False,
                      "options": {"temperature": config.TEMPERATURE, "num_predict": config.MAX_TOKENS, "seed": seed}}, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _claude_response_text(content):
    text = [block.text for block in content if getattr(block, "type", None) == "text"]
    if not text:
        raise NonRetryableGenerationError(
            "Anthropic returned no text block "
            f"(content_types={[getattr(block, 'type', 'unknown') for block in content]}). "
            "Check stop_reason and max_tokens; do not retry the same response."
        )
    return "".join(text)


def claude(model, system, user, seed, max_tokens=None, json_schema=None):
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing from .env")

    looks_placeholder = key.startswith("apikey_") or (
        not key.startswith("sk-ant-") and len(key) < 20 and "anthropic" not in key.lower()
    )
    if looks_placeholder:
        raise RuntimeError(
            "ANTHROPIC_API_KEY looks invalid or incomplete. Paste the real raw Anthropic key from the Anthropic Console into .env, not a placeholder value or 'apikey_' string."
        )

    max_tokens = max_tokens or config.MAX_TOKENS
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        request["system"] = [{"type": "text", "text": system}]
    if model.startswith("claude-haiku-5-5"):
        request["thinking"] = {"type": "disabled"}
        request["output_config"] = {"effort": "medium"}
    elif model.startswith("claude-sonnet-5-5"):
        request["thinking"] = {"type": "between_tools"}
        request["output_config"] = {"effort": "medium"}
    if json_schema:
        request.setdefault("output_config", {})["format"] = {
            "type": "json_schema",
            "schema": json_schema,
        }
    msg = anthropic.Anthropic(api_key=key).messages.create(**request)
    return _claude_response_text(msg.content)
 
 
def _mock_generate(model, system, user, seed):
    """Fake narrative for dry runs: rewrites the factor lines of the prompt. NOT a real result."""
    import random, re
    rnd = random.Random(seed + len(user))
    lines = re.findall(r"\d\. (.+?) \(this claim: (.+?)\) - (raises|lowers)", user)
    keep = [l for l in lines if rnd.random() > 0.15] or lines[:1]
    body = " ".join(f"The {a} ({b}) {c} the score." for a, b, c in keep)
    return f"MOCK OUTPUT. {body} Recommended check: verify the {lines[0][0]}."
 
 
def _mock_judge(system, user, max_tokens=900):
    import random, json
    rnd = random.Random(len(user))
    return json.dumps({"claims": [{"text": "mock statement", "supported": rnd.random() > 0.05} for _ in range(6)],
                       "clarity": rnd.randint(2, 5), "actionability": rnd.randint(1, 5), "rationale": "mock"})
 
 
if config.MOCK:
    gemini = ollama = _mock_generate
    claude = _mock_judge
 
 
def generate(gen_key, system, user, seed, retries=5):
    """Call generator G1 or G2 with automatic retry (waits 1, 2, 4, 8 ... seconds)."""
    g = config.GENERATORS[gen_key]
    provider = g["provider"]
    if provider == "gemini":
        fn = gemini
    elif provider == "claude":
        fn = claude
    elif provider == "ollama":
        fn = ollama
    else:
        raise ValueError(f"Unknown provider: {provider}")

    for attempt in range(retries):
        try:
            return fn(g["model"], system, user, seed)
        except NonRetryableGenerationError:
            raise
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise
            if attempt == retries - 1: raise
            print(f"   retry {attempt + 1} after error: {e}"); time.sleep(2 ** attempt)
