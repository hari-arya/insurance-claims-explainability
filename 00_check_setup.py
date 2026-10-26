"""00_check_setup.py - checks that everything is installed and configured before you start.
Run:  python 00_check_setup.py        Every line should say OK (the API/Ollama lines can wait until Chapter 11)."""
import sys, os, json, importlib, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
problems = []

def line(ok, text):
    print(("OK      " if ok else "PROBLEM ") + text)
    if not ok:
        problems.append(text)
    return ok
 
line(sys.version_info[:2] >= (3, 11), f"Python version {sys.version.split()[0]} (need 3.11 or newer)")
line(sys.prefix != sys.base_prefix, "Virtual environment is active" if sys.prefix != sys.base_prefix else "Virtual environment NOT active - run the activate command first")
for pkg in ["pandas", "numpy", "sklearn", "imblearn", "shap", "lime", "matplotlib", "scipy", "statsmodels",
            "joblib", "dotenv", "anthropic", "requests", "streamlit", "plotly", "bert_score"]:
    try:
        m = importlib.import_module(pkg); line(True, f"{pkg} {getattr(m, '__version__', '')}")
    except Exception as e:
        line(False, f"{pkg} not importable ({e.__class__.__name__}) - run: pip install -r requirements.txt")
import config
line(config.DATA_FILE.exists(), f"Dataset found at {config.DATA_FILE}")
try:
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
    dotenv_available = True
except ImportError:
    dotenv_available = False
    line(False, "python-dotenv unavailable; install project requirements before checking .env")
if dotenv_available:
    for provider in sorted({g["provider"] for g in config.GENERATORS.values()} | {"claude"}):
        if provider == "claude":
            line(bool(os.getenv("ANTHROPIC_API_KEY")), "ANTHROPIC_API_KEY set in .env")
        elif provider == "gemini":
            line(bool(os.getenv("GEMINI_API_KEY")), "GEMINI_API_KEY set in .env")
try:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as response:
        tags = json.load(response)
    names = [m["name"] for m in tags.get("models", [])]
    line(config.GENERATORS["G2"]["model"] in names, f"Ollama running; models installed: {names}")
except Exception:
    line(False, "Ollama not reachable at localhost:11434 - open the Ollama app (needed from Chapter 11)")

if problems:
    raise SystemExit(f"\nSetup check failed: {len(problems)} problem(s) found.")
print("\nSetup check passed.")
