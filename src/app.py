"""Streamlit demo for the Round Advisor POC.

Ties the pieces together in one UI:
1. Overview — dataset + cross-validated model metrics.
2. Active learning — the pooled backtest curve (informed vs random).
3. Predict & rank — score/rank pasted sequences under a budget (no API key needed).
4. Agent — natural-language orchestration via Claude (needs ANTHROPIC_API_KEY).

Run: streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src import mock_api
from src.active_learning import run_backtest
from src.data_loader import load_designs
from src.features import load_features
from src.model import evaluate_classifier, evaluate_regressor

st.set_page_config(page_title="Round Advisor", page_icon="🧬", layout="wide")


@st.cache_data(show_spinner=False)
def _designs() -> pd.DataFrame:
    return load_designs()


@st.cache_data(show_spinner="Cross-validating model...")
def _metrics() -> dict:
    df = load_features()
    clf = evaluate_classifier(df)
    reg = evaluate_regressor(df)
    return {"clf": clf, "reg": reg}


@st.cache_data(show_spinner="Running active-learning backtest...")
def _backtest() -> pd.DataFrame:
    res = run_backtest()
    return pd.DataFrame({
        "designs tested": res.tested,
        "informed": res.informed_hits,
        "random": res.random_hits,
    }).set_index("designs tested")


st.title("🧬 Adaptyv Foundry")
st.caption(
    "Active-learning + agent layer over Adaptyv Bio's public EGFR competition data. "
    "POC, model and API schema are illustrative, not production."
)

tab_overview, tab_al, tab_rank, tab_agent = st.tabs(
    ["Overview", "Active learning", "Predict & rank", "Agent"]
)

with tab_overview:
    df = _designs()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Designs", len(df))
    c2.metric("Binders", int(df["binds"].sum()))
    c3.metric("KD-labeled", int(df["kd"].notna().sum()))
    c4.metric("Rounds", df["round"].nunique())

    m = _metrics()
    st.subheader("Cross-validated model performance")
    a, b = st.columns(2)
    with a:
        st.markdown("**Binder classifier** (5-fold)")
        st.metric("ROC-AUC", f"{m['clf'].roc_auc:.3f}")
        st.metric("Precision@20", f"{m['clf'].precision_at_20:.3f}")
        st.metric("Avg precision", f"{m['clf'].avg_precision:.3f}")
    with b:
        st.markdown("**KD regressor** (log10 KD, 5-fold)")
        st.metric("Spearman", f"{m['reg'].spearman:.3f}")
        st.caption(f"n = {m['reg'].n} KD-labeled designs (small; illustrative)")

with tab_al:
    st.subheader("Does informed selection beat random?")
    st.caption(
        "Pooled rounds 1+2. Seed with a small random set, then an ensemble UCB "
        "acquisition picks each next batch. Curves = binders found vs designs tested."
    )
    curve = _backtest()
    st.line_chart(curve)
    total = int(_designs()["binds"].sum())
    at100 = curve.iloc[(curve.index - 100).to_series().abs().values.argmin()]
    st.metric(
        "At ~100 tests: binders found",
        f"{at100['informed']:.0f} informed vs {at100['random']:.0f} random",
        f"{at100['informed'] / max(at100['random'], 1):.1f}x  ({total} total binders)",
    )

with tab_rank:
    st.subheader("Score & rank candidate sequences")
    st.caption("No API key needed, uses the trained model directly.")
    default_seqs = (
        "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGRRGYFDYWGQGTLVTVSS\n"
        "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK\n"
        "GSHMKEIAALKEKIAALKEKIAALKE"
    )
    text = st.text_area("Sequences (one per line)", value=default_seqs, height=140)
    budget = st.number_input("Test budget", min_value=1, max_value=100, value=2)
    if st.button("Rank"):
        seqs = [s.strip() for s in text.splitlines() if s.strip()]
        scored = mock_api.score_sequences(seqs)
        table = pd.DataFrame(scored).sort_values("acquisition_score", ascending=False)
        table.insert(0, "selected", [i < budget for i in range(len(table))])
        st.dataframe(table, use_container_width=True)

with tab_agent:
    st.subheader("Ask the agent")
    if not os.getenv("OPENROUTER_API_KEY"):
        st.warning("Set OPENROUTER_API_KEY (in a .env file) to use the agent.")
    else:
        from src.agent import run_agent

        q = st.text_area(
            "Request",
            value="I have budget for 2 tests. Rank these and explain the trade-off:\n"
            "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGRRGYFDYWGQGTLVTVSS\n"
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK\n"
            "GSHMKEIAALKEKIAALKEKIAALKE",
            height=140,
        )
        if st.button("Run agent"):
            with st.spinner("Agent thinking..."):
                st.markdown(run_agent(q))
