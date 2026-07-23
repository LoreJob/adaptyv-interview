"""JSON API backend for the Round Advisor React UI.

Wraps the model, active-learning backtest, mock lab API, and agent behind a small
set of JSON endpoints the frontend fetches. Heavy computations (CV metrics,
backtest) are cached in memory after first use.

Run: uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache

if __package__ in (None, ""):  # allow `python src/api.py`
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import mock_api
from src.active_learning import run_backtest
from src.data_loader import load_designs
from src.features import load_features
from src.model import evaluate_classifier, evaluate_regressor

load_dotenv()

app = FastAPI(title="Round Advisor API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- cached computations --------------------------------------------------

@lru_cache(maxsize=1)
def _stats() -> dict:
    df = load_designs()
    return {
        "designs": int(len(df)),
        "binders": int(df["binds"].sum()),
        "kd_labeled": int(df["kd"].notna().sum()),
        "rounds": int(df["round"].nunique()),
        "strengths": {k: int(v) for k, v in df["binding_strength"].value_counts().items()},
    }


@lru_cache(maxsize=1)
def _example_binder() -> dict:
    """A real strong binder from the data, for the hero data card."""
    df = load_designs()
    strong = df[(df["binds"]) & df["kd"].notna()].sort_values("kd")
    r = strong.iloc[0]
    return {
        "id": str(r["design_id"])[:24],
        "kd_nM": round(float(r["kd"]) * 1e9, 2),
        "binding_strength": str(r["binding_strength"]).capitalize(),
        "expression": str(r["expression"]).capitalize(),
        "round": int(r["round"]),
    }


@lru_cache(maxsize=1)
def _metrics() -> dict:
    df = load_features()
    clf = evaluate_classifier(df)
    reg = evaluate_regressor(df)
    return {
        "roc_auc": round(clf.roc_auc, 3),
        "avg_precision": round(clf.avg_precision, 3),
        "precision_at_20": round(clf.precision_at_20, 3),
        "precision_at_50": round(clf.precision_at_50, 3),
        "spearman": round(reg.spearman, 3),
        "kd_n": reg.n,
        "n": clf.n,
        "n_pos": clf.n_pos,
    }


@lru_cache(maxsize=1)
def _backtest() -> dict:
    res = run_backtest()
    s = res.summary_at(100)
    return {
        "tested": [int(x) for x in res.tested],
        "informed": [round(float(x), 2) for x in res.informed_hits],
        "random": [round(float(x), 2) for x in res.random_hits],
        "total_hits": res.total_hits,
        "summary_at_100": {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()},
    }


# --- request models -------------------------------------------------------

class RankRequest(BaseModel):
    sequences: list[str] = Field(..., min_length=1)
    budget: int = Field(1, ge=1)


class AgentRequest(BaseModel):
    message: str


# --- endpoints ------------------------------------------------------------

@app.get("/api/stats")
def stats() -> dict:
    return _stats()


@app.get("/api/example-binder")
def example_binder() -> dict:
    return _example_binder()


@app.get("/api/metrics")
def metrics() -> dict:
    return _metrics()


@app.get("/api/backtest")
def backtest() -> dict:
    return _backtest()


@app.post("/api/rank")
def rank(req: RankRequest) -> dict:
    seqs = [s.strip() for s in req.sequences if s.strip()]
    if not seqs:
        raise HTTPException(status_code=400, detail="No sequences provided.")
    scored = mock_api.score_sequences(seqs)
    scored.sort(key=lambda d: d["acquisition_score"], reverse=True)
    for i, d in enumerate(scored):
        d["selected"] = i < req.budget
    return {"budget": req.budget, "ranked": scored}


@app.get("/api/agent/available")
def agent_available() -> dict:
    return {"available": bool(os.getenv("OPENROUTER_API_KEY"))}


@app.post("/api/agent")
def agent(req: AgentRequest) -> dict:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not set.")
    from src.agent import run_agent
    try:
        return {"reply": run_agent(req.message)}
    except Exception as exc:  # surface agent/model errors to the UI
        raise HTTPException(status_code=500, detail=str(exc))
