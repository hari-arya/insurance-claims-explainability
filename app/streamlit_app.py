"""app/streamlit_app.py - claim explanation dashboard.
Run from the project folder:   streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import config
from src.explanations import ClaimBank, C1, C2, prompt, final_text, serialise
from src.text_eval import fidelity
from src import store
 
st.set_page_config(page_title="Fraud Flag Explainer", layout="wide")
st.title("Insurance claim fraud flag explainer")
st.caption("Research prototype for an MSc thesis. Scores come from a model trained on a public dataset; they are not real risk estimates.")
 
@st.cache_resource
def get_bank():
    return ClaimBank()
 
try:
    bank = get_bank()
except FileNotFoundError:
    st.error("Model files not found. Run scripts/02_train.py and scripts/03_explain.py first."); st.stop()
 
flagged = bank.flagged("test")
choice = st.sidebar.selectbox("Flagged test claim", flagged, format_func=lambda i: f"Claim #{i}  (score {bank.claim(i)['prob']:.2f})")
c = bank.claim(choice)
st.sidebar.markdown(f"**Model:** {config.EXPLAINED_MODEL}  \n**Threshold:** {c['threshold']:.2f}  \n**True label:** {'Fraud' if c['label'] else 'Legitimate'}")
 
tab_claim, tab_model, tab_about = st.tabs(["Explain a claim", "Model performance", "About"])
with tab_claim:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Risk score")
        st.metric("Fraud risk score", f"{c['prob']:.2f}", help="From a model trained on oversampled data, so it overstates true probability (Thesis 5.2).")
        st.subheader("Top five SHAP drivers")
        d = pd.DataFrame(c["drivers"])
        fig = go.Figure(go.Bar(x=d["shap"][::-1], y=(d["label"] + " = " + d["value"])[::-1], orientation="h",
                               marker_color=["#c0504d" if v > 0 else "#1f4e79" for v in d["shap"][::-1]]))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="SHAP value (red raises, blue lowers)")
        st.plotly_chart(fig, width="stretch")
        with st.expander("Claim details (serialised input)"):
            st.text(serialise(c))
    with right:
        st.subheader("Template explanation (C2, no LLM)")
        st.info(C2(c))
        st.subheader("LLM explanation (C3)")
        gen = st.selectbox("Generator", list(config.GENERATORS), format_func=lambda g: f"{g}: {config.GENERATORS[g]['model']}")
        p = st.selectbox("Prompt design", config.PROMPTS, index=1)
        cached = [g for g in store.read(store.GENERATIONS) if g["idx"] == choice and g["gen"] == gen and g["prompt"] == p]
        text = cached[0]["text"] if cached else None
        if st.button("Generate now (calls the model)"):
            from src.llm_clients import generate
            system, user = prompt(c, p)
            with st.spinner("Asking the model ..."):
                text = final_text(p, generate(gen, system, user, config.RUN_SEEDS[0]))
        if text:
            f = fidelity(c, text, bank.names)
            ok = f["driver_recall"] >= config.TARGET_DRIVER_RECALL and (pd.isna(f["sign_accuracy"]) or f["sign_accuracy"] == 1)
            st.write(text)
            st.caption(f"Driver recall {f['driver_recall']:.2f} · sign accuracy {f['sign_accuracy']:.2f} · {f['words']} words" if not pd.isna(f["sign_accuracy"])
                       else f"Driver recall {f['driver_recall']:.2f} · no direction stated · {f['words']} words")
            if not ok:
                st.warning("This narrative failed the automatic fidelity check. In deployment the template (C2) would be shown instead.")
        else:
            st.write("No cached narrative for this claim. Press the button to generate one.")
 
with tab_model:
    import json
    res = config.OUT / "train_results.json"
    if res.exists():
        R = json.load(open(res))
        st.subheader("Test-set performance (Thesis Table 4.2)")
        st.dataframe(pd.DataFrame({m: {k: v for k, v in R["test"][m]["tuned"].items() if k not in ("confusion_matrix", "f1_ci", "pr_auc_ci")}
                                   for m in R["test"]}).T, width="stretch")
        c1, c2 = st.columns(2)
        for col, name, cap in [(c1, "fig_4_4_pr_curves.png", "Precision-recall curves"), (c2, "fig_4_5_shap_global.png", "Global SHAP importance")]:
            if (config.FIG / name).exists(): col.image(str(config.FIG / name), caption=cap)
    else:
        st.info("Run scripts/02_train.py to see model performance here.")
 
with tab_about:
    st.markdown("""**What this is.** A research prototype that explains why a fraud model flagged a motor insurance claim.
It shows the model's risk score, the five SHAP factors behind it, a template explanation (no AI text generation) and,
on request, a narrative written by a language model, which is automatically checked against the SHAP factors.
 
**Limits.** The model was trained on a small public dataset. Two of its strongest factors are policyholder hobbies,
which have no plausible link to fraud, and the scores are inflated by oversampling. Do not use it for real decisions.""")
