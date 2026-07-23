"""Mock Adaptyv-style REST API for the Round Advisor POC.

**This schema is a hypothesis, not Adaptyv's real API.** It is a plausible
design-submission flow inspired by the public product narrative
(upload designs -> run experiments -> results -> next round), documented here so
the agent has a concrete surface to orchestrate. Treat it as a starting point
for alignment, not ground truth.

Flow:
* ``POST /designs``            -> submit a batch of sequences, get a batch id.
* ``GET  /designs/{id}/results`` -> retrieve per-sequence experimental results.

Because these are novel sequences with no wet-lab measurement, the "experiment"
is **simulated**: results are the predictive model's own binder-probability and
KD estimate, clearly flagged ``simulated=True``. In a real deployment this
endpoint would return actual BLI/SPR characterization from the Adaptyv lab.

The in-process helpers (:func:`submit_batch`, :func:`get_results`) back both the
FastAPI app and the agent tools, so nothing needs an HTTP server running.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if __package__ in (None, ""):  # allow `python src/mock_api.py` and IDE Run
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.features import load_features, sequences_to_feature_frame
from src.model import BootstrapEnsemble, RANDOM_STATE, _classifier_pipeline, prepare_xy

# --- schema ---------------------------------------------------------------

class DesignSubmission(BaseModel):
    """Request body for POST /designs."""
    sequences: list[str] = Field(..., min_length=1, description="Protein sequences (1-letter AA).")
    round_name: str | None = Field(None, description="Optional label for this submission round.")


class DesignResult(BaseModel):
    """One design's (simulated) characterization result."""
    design_id: str
    sequence: str
    binder_probability: float = Field(..., description="Predicted P(binds), 0-1.")
    binder_probability_std: float = Field(..., description="Ensemble uncertainty on the probability.")
    predicted_log10_kd: float | None = Field(None, description="Predicted log10(KD in M); null if not a likely binder.")
    simulated: bool = Field(True, description="True = model prediction, not a real lab measurement.")


class BatchResponse(BaseModel):
    """Response for POST /designs."""
    batch_id: str
    round_name: str | None
    n_designs: int
    status: str
    submitted_at: str


class ResultsResponse(BaseModel):
    """Response for GET /designs/{batch_id}/results."""
    batch_id: str
    status: str
    results: list[DesignResult]


# --- in-memory store + model ---------------------------------------------

_BATCHES: dict[str, dict] = {}
_MODELS: dict[str, object] = {}  # lazily-fitted predictors, cached


def _get_predictors() -> tuple[BootstrapEnsemble, object]:
    """Fit (once) the binder classifier ensemble and a KD regressor."""
    if "binder" not in _MODELS:
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        df = load_features()
        Xc, yc = prepare_xy(df, "binds")
        _MODELS["binder"] = BootstrapEnsemble(
            _classifier_pipeline(), n_estimators=25, classifier=True
        ).fit(Xc, yc)

        Xr, yr = prepare_xy(df, "log10_kd")
        reg = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        reg.fit(Xr, yr)
        _MODELS["kd"] = reg
    return _MODELS["binder"], _MODELS["kd"]


def _predict(sequences: list[str]) -> list[DesignResult]:
    """Score novel sequences with the fitted models."""
    binder, kd = _get_predictors()
    X = sequences_to_feature_frame(sequences)
    prob_mean, prob_std = binder.predict_mean_std(X)
    log_kd = kd.predict(X)

    results = []
    for seq, pm, ps, lk in zip(sequences, prob_mean, prob_std, log_kd):
        results.append(DesignResult(
            design_id=f"dsn_{uuid.uuid4().hex[:10]}",
            sequence=seq,
            binder_probability=round(float(pm), 4),
            binder_probability_std=round(float(ps), 4),
            # Only surface a KD estimate for likely binders.
            predicted_log10_kd=round(float(lk), 3) if pm >= 0.5 else None,
            simulated=True,
        ))
    return results


def score_sequences(sequences: list[str]) -> list[dict]:
    """Predict binder probability + KD for sequences without submitting a batch.

    Returns plain dicts (design_id omitted) for use by the agent's prediction
    and ranking tools.
    """
    return [
        {
            "sequence": r.sequence,
            "binder_probability": r.binder_probability,
            "binder_probability_std": r.binder_probability_std,
            "predicted_log10_kd": r.predicted_log10_kd,
            # UCB acquisition score = mean + 1 std (explore high-uncertainty).
            "acquisition_score": round(r.binder_probability + r.binder_probability_std, 4),
        }
        for r in _predict(sequences)
    ]


# --- in-process helpers (used by agent + FastAPI) -------------------------

def submit_batch(sequences: list[str], round_name: str | None = None) -> BatchResponse:
    """Submit designs; runs the simulated experiment immediately."""
    batch_id = f"bat_{uuid.uuid4().hex[:10]}"
    results = _predict(sequences)
    _BATCHES[batch_id] = {
        "round_name": round_name,
        "results": results,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    return BatchResponse(
        batch_id=batch_id,
        round_name=round_name,
        n_designs=len(sequences),
        status="completed",
        submitted_at=_BATCHES[batch_id]["submitted_at"],
    )


def get_results(batch_id: str) -> ResultsResponse:
    """Fetch results for a previously submitted batch."""
    if batch_id not in _BATCHES:
        raise KeyError(batch_id)
    b = _BATCHES[batch_id]
    return ResultsResponse(batch_id=batch_id, status="completed", results=b["results"])


# --- FastAPI app ----------------------------------------------------------

app = FastAPI(
    title="Adaptyv Foundry, Mock API",
    description="HYPOTHETICAL design-submission API. Not Adaptyv's real schema.",
    version="0.1.0",
)


@app.post("/designs", response_model=BatchResponse)
def post_designs(submission: DesignSubmission) -> BatchResponse:
    return submit_batch(submission.sequences, submission.round_name)


@app.get("/designs/{batch_id}/results", response_model=ResultsResponse)
def get_designs_results(batch_id: str) -> ResultsResponse:
    try:
        return get_results(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown batch_id: {batch_id}")
