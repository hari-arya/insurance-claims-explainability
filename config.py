"""config.py - every setting of the project in one place.
Change a value here and every script picks it up."""
import os
from pathlib import Path
 
# Dry-run switch: set MOCK_LLM=1 to test the whole pipeline without any API calls or cost.
# Mock results go to outputs_mock/ so they can never be mixed up with real results.
MOCK = os.getenv("MOCK_LLM") == "1"
 
# ---------- folders ----------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "insurance_claims.csv"
DEFAULT_RUN_DIR = ROOT / ("outputs_mock" if MOCK else "outputs")
RUN_DIR = Path(os.getenv("INSURANCE_CLAIMS_RUN_DIR", DEFAULT_RUN_DIR)).expanduser().resolve()
OUT = RUN_DIR
FIG = OUT / "figures"
TAB = OUT / "tables"
LLM_OUT = OUT / "llm"
MODELS = RUN_DIR / "models" if os.getenv("INSURANCE_CLAIMS_RUN_DIR") else ROOT / "models"
for folder in (OUT, FIG, TAB, LLM_OUT, MODELS):  # created automatically if missing
    folder.mkdir(parents=True, exist_ok=True)
 
# ---------- reproducibility ----------
SEED = 42            # used for every random step (split, SMOTE-NC, models, LIME)
TEST_SIZE = 0.20     # 80/20 stratified split -> 800 training and 200 test claims
CV_FOLDS = 5         # stratified cross-validation folds
N_BOOTSTRAP = 1000   # resamples for the 95% confidence intervals
 
# ---------- data preparation (Thesis Section 3.4) ----------
DROP_COLUMNS = ["policy_number", "insured_zip", "incident_location", "auto_model"]
TARGET = "fraud_reported"
 
# ---------- models (Thesis Section 3.7, Table 3.1) ----------
LR_GRID = {"clf__C": [0.01, 0.1, 1, 10]}
RF_GRID = {"clf__n_estimators": [300], "clf__max_depth": [None, 6, 10],
           "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.3]}
SVM_GRID = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}
EXPLAINED_MODEL = "RF"   # the model whose flags are explained (exact TreeSHAP)
 
# ---------- attribution (Thesis Section 3.9) ----------
TOP_K = 5                    # drivers passed to the explanation
LIME_SAMPLES = 2000
LIME_SEEDS = [42, 1, 2]      # first seed = main run, all three = stability check
 
# ---------- language models (Thesis Section 3.12, Table 3.4) ----------
GENERATORS = {
    "G1": {"provider": "ollama", "model": "gemma3:12b"},
    "G2": {"provider": "ollama", "model": "llama3.1:8b-instruct-q4_K_M"},
}

JUDGE_MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "grounded-actionable-v3"
PROMPTS = ["P0", "P1", "P2", "P3"]
TEMPERATURE = 0.2
MAX_TOKENS = 400
JUDGE_MAX_TOKENS = 1200
RUNS = 3
RUN_SEEDS = [11, 22, 33]
WORD_RANGE = (80, 130)
OLLAMA_URL = "http://localhost:11434/api/chat"
 
# ---------- evaluation targets fixed in advance (Thesis Section 1.4) ----------
TARGET_DRIVER_RECALL = 0.90
TARGET_UCR = 0.05
KAPPA_THRESHOLD = 0.40
HUMAN_SAMPLE_SHARE = 0.20