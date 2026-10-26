"""scripts/01_eda.py - exploratory data analysis (Thesis Section 4.1, Figure 4.1, Table 4.1).
Run:  python scripts/01_eda.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu
import config
from src.data_loader import load_raw, summary
 
df = load_raw()
S = summary(df)
y = df[config.TARGET]
 
# fraud rate by level for the two strongest attributes
S["fraud_rate_by_severity"] = df.groupby("incident_severity")[config.TARGET].apply(lambda s: round((s == "Y").mean(), 3)).to_dict()
S["fraud_rate_by_hobby_top4"] = (df.groupby("insured_hobbies")[config.TARGET].apply(lambda s: round((s == "Y").mean(), 3))
                                 .sort_values(ascending=False).head(4).to_dict())
S["skew_total_claim"] = round(float(df["total_claim_amount"].skew()), 2)
S["spearman_total_vs_vehicle"] = round(float(df[["total_claim_amount", "vehicle_claim"]].corr("spearman").iloc[0, 1]), 2)
S["spearman_age_vs_months"] = round(float(df[["age", "months_as_customer"]].corr("spearman").iloc[0, 1]), 2)
 
# Table 4.1 - association tests
rows = []
for col in ["incident_severity", "insured_hobbies", "collision_type", "incident_type", "authorities_contacted",
            "police_report_available", "insured_education_level", "insured_sex"]:
    table = pd.crosstab(df[col], y)
    chi2, p, dof, _ = chi2_contingency(table)
    v = np.sqrt(chi2 / (len(df) * (min(table.shape) - 1)))
    rows.append({"attribute": col, "test": f"chi2({dof}) = {chi2:.1f}", "p_value": p, "effect": f"Cramer's V = {v:.2f}"})
for col in ["total_claim_amount", "age", "months_as_customer", "policy_annual_premium"]:
    a, b = df.loc[y == "Y", col], df.loc[y == "N", col]
    rows.append({"attribute": col, "test": "Mann-Whitney U", "p_value": mannwhitneyu(a, b).pvalue,
                 "effect": f"median fraud {a.median():,.0f} vs legit {b.median():,.0f}"})
pd.DataFrame(rows).to_csv(config.TAB / "table_4_1_associations.csv", index=False)
 
# Figure 4.1
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
counts = (y == "Y").astype(int).value_counts().sort_index()
ax[0].bar(["Legitimate (N)", "Fraud (Y)"], counts.values, color=["#8aa4c8", "#c0504d"])
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 10, f"{v} ({v / len(df):.1%})", ha="center")
ax[0].set_ylabel("Claims"); ax[0].set_title("(a) Class distribution"); ax[0].set_ylim(0, 850)
q = pd.Series(S["unknown_values"]).sort_values() / len(df) * 100
ax[1].barh(q.index, q.values, color="#8aa4c8"); ax[1].set_xlabel("% of records coded '?'")
ax[1].set_title("(b) Unknown values by column"); ax[1].set_xlim(0, 45)
for i, v in enumerate(q.values):
    ax[1].text(v + 0.5, i, f"{v:.1f}%", va="center")
plt.tight_layout(); plt.savefig(config.FIG / "fig_4_1_eda.png", dpi=150); plt.close()
 
json.dump(S, open(config.OUT / "eda_summary.json", "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
print("\nSaved: outputs/eda_summary.json, outputs/tables/table_4_1_associations.csv, outputs/figures/fig_4_1_eda.png")
