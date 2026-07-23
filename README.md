<div align="center">

# 🧬 Adaptyv Foundry

### Test fewer designs. Find more binders.

An **active-learning + agent** layer over Adaptyv Bio's public EGFR protein-design
competition data, built as a take-home proof of concept.

![status](https://img.shields.io/badge/status-proof--of--concept-2f93e6)
![python](https://img.shields.io/badge/python-3.13-2f93e6)
![stack](https://img.shields.io/badge/stack-FastAPI%20%2B%20React-2f93e6)
![data](https://img.shields.io/badge/data-ODbL-2ea36b)

</div>

---

## TL;DR

On Adaptyv's own published EGFR rounds (604 designs, 63 real binders), an ensemble
**UCB acquisition strategy finds ~2.2× more binders than random selection at a
100-test budget** (24 vs ~11). The value is the *decision layer*: rank designs so
the real binders surface early, and the same wet-lab budget buys more hits.

A small tool-calling **agent** wraps the model in the Adaptyv product loop
(upload designs, run experiments, read results, plan the next round), and a
Vite/React dashboard in Adaptyv's visual style makes it demoable.

> This is a POC, not a production predictor. The point is the strategy, not
> squeezing accuracy out of a few hundred data points.

---

## Why this matters for Adaptyv

Adaptyv runs a cloud lab where every wet-lab test costs time and money. If a model
ranks designs so the real binders come first, the same experimental budget finds
more binders, or the same result costs fewer experiments. This project backtests
exactly that on Adaptyv's own rounds, and shows how it plugs into an API-driven
"submit a round, get results, plan the next round" loop.

## Headline result

Pooled EGFR rounds, 604 designs, 63 real binders, 5-fold cross-validation:

| Model | Metric | Value |
|---|---|---|
| Active learning (UCB vs random) | Binders found @100 tests | **24 vs ~11 (2.2×)** |
| Binder classifier | ROC-AUC | 0.86 |
| Binder classifier | Precision@20 | 0.85 |
| KD regressor (log10 KD, n=63) | Spearman | 0.40 |

The KD regressor is intentionally modest: only 63 designs have a fitted KD, so it
is illustrative. Reporting uses Spearman, not R², given the small, outlier-prone set.

---

## Quickstart

```bash
# 1. Environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # optional: add OPENROUTER_API_KEY for the agent panel

# 2. Backend API (terminal 1)
.venv/bin/uvicorn src.api:app --port 8000 --reload

# 3. Frontend dashboard (terminal 2)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** (the frontend proxies `/api` to the backend on `:8000`).
The dashboard runs fully without a key; only the agent panel needs `OPENROUTER_API_KEY`.

<details>
<summary><b>Other entry points</b></summary>

```bash
# Data + model pipeline (each prints a short report)
.venv/bin/python -m src.data_loader
.venv/bin/python -m src.features
.venv/bin/python -m src.model
.venv/bin/python -m src.active_learning

# Tests
.venv/bin/python -m pytest

# Agent from the CLI (needs OPENROUTER_API_KEY)
.venv/bin/python -m src.agent "I have budget for 20 tests, give me the best selection and explain the trade-off"

# Streamlit demo (quick, single process)
.venv/bin/streamlit run src/app.py
```

</details>

---

## Architecture

```
src/
  data_loader.py      harmonize rounds 1+2 into one design table (data/processed/designs.parquet)
  features.py         Biopython ProtParam physicochemical features + reused AF2/ESM metrics
  model.py            binder classifier + KD regressor, bootstrap ensemble, CV reporting
  active_learning.py  pooled UCB backtest vs random
  mock_api.py         hypothetical Adaptyv REST schema (FastAPI + Pydantic), simulated lab
  agent.py            tool-calling agent (4 tools) over the model + mock API (OpenRouter)
  api.py              JSON API for the React UI (stats, metrics, backtest, rank, agent)
  app.py              Streamlit demo tying it together
frontend/             Vite + React dashboard in Adaptyv's visual style, calls src/api.py
tests/                sanity checks on the backtest
notebooks/            01_eda.ipynb, exploratory data analysis
```

The **mock API schema is a hypothesis** based on the public product narrative, not
Adaptyv's real API. It is a starting point for alignment, flagged as such in code.

---

## Data

Adaptyv Bio EGFR Protein Design Competition, rounds 1 and 2 (public):

- https://github.com/adaptyvbio/egfr_competition_1
- https://github.com/adaptyvbio/egfr_competition_2

Data is licensed **ODbL** (Open Database License); competition code is Apache 2.0.
The modeling data is the processed characterization tables (`results/*.csv`), not
the raw SPR-curve packages. The processed table (`data/processed/designs.parquet`)
ships in this repo so the app runs out of the box; the heavy raw assay bundle is
not included (regenerate it via the links above if you want to rebuild from scratch).

Three data realities shaped the design:

- **KD is sparse.** Round 1 reports a fitted KD for only 8 designs, round 2 for 55.
  Too few for a per-round KD regressor, so binding is framed primarily as
  **classification** (binder vs non-binder, all 604 designs), with KD regression as
  a secondary head on the 63 labeled designs.
- **Schemas differ between rounds.** Round 1 lacks inline binding/expression columns
  (aggregated from its replicate table); round 2 carries them plus ESM/AF2 metrics.
  `data_loader.py` harmonizes both into one schema.
- **Neutralisation is out of scope.** Round 2's neutralisation package wasn't part of
  this build; the POC is binding-affinity only.

---

## Honest limits

- **Small dataset** (hundreds of points, 63 binders): the model is illustrative, not
  a production predictor. Don't oversell accuracy to a team that generates this data.
- **Cold-start features.** Predictions on novel sequences use physicochemical features
  only (the AF2/ESM metrics that help most aren't available pre-fold), so they lean on
  median imputation.
- **Hypothetical API.** The schema is a proposal; the agent's "lab results" are
  simulated model predictions, clearly flagged.
- **Deliberately small agent** (4 tools): a simple agent that works beats an ambitious
  one that breaks in a demo.
