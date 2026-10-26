"""src/data_loader.py - load, check and clean the Mendeley insurance_claims file (Thesis 3.2-3.4)."""
import pandas as pd
import config
 
 
def load_raw() -> pd.DataFrame:
    """Read the CSV exactly as stored.
    keep_default_na=False stops pandas turning the valid category "None"
    (no authority contacted) into a missing value."""
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"Put insurance_claims.csv in {config.DATA_FILE.parent}")
    df = pd.read_csv(config.DATA_FILE, keep_default_na=False, na_values=[""])
    df = df.loc[:, ~df.columns.str.startswith("_c")]      # some copies carry an empty _c39 column
    assert config.TARGET in df.columns, "fraud_reported column missing - wrong file?"
    return df
 
 
def prepare(df: pd.DataFrame):
    """Return (X, y) ready for modelling.
    - y: 1 = fraud ("Y"), 0 = legitimate ("N")
    - dates replaced by policy_tenure_days
    - identifier-like columns removed
    - "?" kept as its own category called UNKNOWN (not imputed)"""
    df = df.copy()
    y = (df[config.TARGET] == "Y").astype(int)
    bind = pd.to_datetime(df["policy_bind_date"])
    incident = pd.to_datetime(df["incident_date"])
    df["policy_tenure_days"] = (incident - bind).dt.days
    X = df.drop(columns=config.DROP_COLUMNS + ["policy_bind_date", "incident_date", config.TARGET])
    X = X.replace("?", "UNKNOWN")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, cat_cols, num_cols
 
 
def summary(df: pd.DataFrame) -> dict:
    """Numbers reported in Thesis Section 4.1."""
    qmark = (df == "?").sum()
    return {
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "fraud_counts": df[config.TARGET].value_counts().to_dict(),
        "unknown_values": {k: int(v) for k, v in qmark[qmark > 0].items()},
        "authorities_none": int((df["authorities_contacted"] == "None").sum()),
        "duplicates": int(df.duplicated().sum()),
    }
