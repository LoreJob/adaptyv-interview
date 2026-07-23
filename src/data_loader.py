"""Data layer for the Round Advisor POC.

Loads and harmonizes Adaptyv Bio's public EGFR protein-design competition data
(rounds 1 and 2) into a single design-level table.

Key realities of the source data (see PLAN.md / reference READMEs):

* The usable modeling data lives in the *processed* tables under
  ``reference/egfr_competition_{1,2}-main/results/`` — NOT the raw SPR-curve zips.
* KD is sparse: round 1 reports a fitted KD for only 8 designs, round 2 for 55.
  Non-binders have no KD (right-censored); we keep ``kd = NaN`` rather than
  imputing a fake value into the regression target.
* Round 1 and round 2 ``result_summary.csv`` have *different* schemas. Round 1
  has no binding/expression columns — those are aggregated up from its
  ``replicate_summary.csv``. Round 2 carries them inline.
* KD is reported in molar (e.g. ``6.6e-9`` == 6.6 nM); lower means stronger.

Public entry point: :func:`load_designs`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- paths ----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
R1_RESULTS = REPO_ROOT / "reference" / "egfr_competition_1-main" / "results"
R2_RESULTS = REPO_ROOT / "reference" / "egfr_competition_2-main" / "results"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "designs.parquet"

# --- unified schema -------------------------------------------------------

# One row per design after harmonization. iptm/esm_pll are round-2 only.
UNIFIED_COLUMNS = [
    "design_id",
    "username",
    "round",
    "sequence",
    "dna",
    "kd",
    "log10_kd",
    "binds",
    "binding_strength",
    "expression",
    "pae_interaction",
    "plddt",
    "iptm",
    "esm_pll",
    "similarity_check",
]

# Canonical binding-strength bins, derived from round 2's native
# KD <-> binding_strength mapping so round 1 can be labeled consistently:
#   strong  KD < 50 nM      (observed strong max ~18 nM, medium min ~52 nM)
#   medium  50 nM <= KD < 1 uM
#   weak    1 uM <= KD <= 10 uM (assay ceiling)
#   none    no KD / not a binder
_STRONG_MAX_M = 50e-9
_MEDIUM_MAX_M = 1e-6

# Ordinal rank for aggregating categorical expression across replicates.
_EXPRESSION_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_RANK_EXPRESSION = {v: k for k, v in _EXPRESSION_RANK.items()}

# Round-1 replicate ``binding`` values that count as a real binding event.
_R1_BINDING_TRUE = {"true", "weak"}


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce to float, mapping ``null``/``''``/``none`` style tokens to NaN."""
    return pd.to_numeric(series, errors="coerce")


def strength_from_kd(kd_molar: float, *, binds: bool) -> str:
    """Map a molar KD to the canonical binding-strength bin.

    ``binds`` covers designs that bind but have no fitted KD (labeled ``weak``).
    """
    if kd_molar is not None and not pd.isna(kd_molar):
        if kd_molar < _STRONG_MAX_M:
            return "strong"
        if kd_molar < _MEDIUM_MAX_M:
            return "medium"
        return "weak"
    return "weak" if binds else "none"


# --- round 1 --------------------------------------------------------------

def _aggregate_round1_replicates() -> pd.DataFrame:
    """Collapse round-1 replicate rows to one row per design.

    Round 1's ``result_summary.csv`` has no binding/expression, so we derive
    design-level ``binds`` and ``expression`` from ``replicate_summary.csv``:

    * ``binds``      — any replicate with ``binding`` in {true, weak}.
    * ``expression`` — best (max-rank) expression seen across replicates.
    """
    rep = pd.read_csv(R1_RESULTS / "replicate_summary.csv")
    rep["binding_norm"] = rep["binding"].astype(str).str.strip().str.lower()
    rep["expression_norm"] = rep["expression"].astype(str).str.strip().str.lower()
    rep["_expr_rank"] = rep["expression_norm"].map(_EXPRESSION_RANK)

    grouped = rep.groupby("name", as_index=False).agg(
        binds=("binding_norm", lambda s: bool(s.isin(_R1_BINDING_TRUE).any())),
        _expr_rank=("_expr_rank", "max"),
    )
    grouped["expression"] = grouped["_expr_rank"].map(_RANK_EXPRESSION)
    return grouped[["name", "binds", "expression"]]


def load_round1() -> pd.DataFrame:
    """Load round 1 into the unified schema."""
    summ = pd.read_csv(R1_RESULTS / "result_summary.csv")
    summ["kd"] = _to_numeric(summ["kd"])

    agg = _aggregate_round1_replicates()
    df = summ.merge(agg, on="name", how="left")

    # Designs absent from replicate_summary: treat as non-binders.
    df["binds"] = df["binds"].fillna(False).astype(bool)
    df["expression"] = df["expression"].fillna("none")

    df["binding_strength"] = [
        strength_from_kd(kd, binds=binds)
        for kd, binds in zip(df["kd"], df["binds"])
    ]

    out = pd.DataFrame({
        "design_id": df["name"],
        "username": df["username"],
        "round": 1,
        "sequence": df["sequence"],
        "dna": df["dna"],
        "kd": df["kd"],
        "binds": df["binds"],
        "binding_strength": df["binding_strength"],
        "expression": df["expression"],
        "pae_interaction": _to_numeric(df["pae_interaction"]),
        "plddt": _to_numeric(df["plddt"]),
        "iptm": np.nan,          # not measured in round 1
        "esm_pll": np.nan,       # not measured in round 1
        "similarity_check": _to_numeric(df["similarity_check"]),
    })
    return out


# --- round 2 --------------------------------------------------------------

def load_round2() -> pd.DataFrame:
    """Load round 2 into the unified schema (binding/expression are inline)."""
    df = pd.read_csv(R2_RESULTS / "result_summary.csv")
    df["kd"] = _to_numeric(df["kd"])

    binding = df["binding"].astype(str).str.strip().str.lower()
    binds = binding.eq("true")

    expression = df["expression"].astype(str).str.strip().str.lower()
    expression = expression.where(expression.isin(_EXPRESSION_RANK), "none")

    strength = df["binding_strength"].astype(str).str.strip().str.lower()
    # Fold round 2's "unknown" strength into the canonical scale via KD/binds.
    known = {"strong", "medium", "weak", "none"}
    strength = pd.Series([
        s if s in known else strength_from_kd(kd, binds=b)
        for s, kd, b in zip(strength, df["kd"], binds)
    ], index=df.index)

    out = pd.DataFrame({
        "design_id": df["name"],
        "username": df["username"],
        "round": 2,
        "sequence": df["sequence"],
        "dna": df["dna"],
        "kd": df["kd"],
        "binds": binds.to_numpy(dtype=bool),
        "binding_strength": strength,
        "expression": expression,
        "pae_interaction": _to_numeric(df["pae_interaction"]),
        "plddt": _to_numeric(df["plddt"]),
        "iptm": _to_numeric(df["iptm"]),
        "esm_pll": _to_numeric(df["esm_pll"]),
        "similarity_check": _to_numeric(df["similarity_check"]),
    })
    return out


# --- orchestrator ---------------------------------------------------------

def harmonize() -> pd.DataFrame:
    """Concatenate both rounds into one design-level table."""
    df = pd.concat([load_round1(), load_round2()], ignore_index=True)

    # Derived + hygiene.
    df["log10_kd"] = np.log10(df["kd"])
    df = df.drop_duplicates(subset=["design_id", "round"]).reset_index(drop=True)

    return df[UNIFIED_COLUMNS]


def load_designs(cache: bool = True) -> pd.DataFrame:
    """Return the harmonized design table, writing a parquet cache.

    Parameters
    ----------
    cache:
        If True, read from ``data/processed/designs.parquet`` when present and
        (re)write it after building.
    """
    if cache and PROCESSED_PATH.exists():
        return pd.read_parquet(PROCESSED_PATH)

    df = harmonize()

    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH, index=False)

    return df


if __name__ == "__main__":
    frame = load_designs(cache=False)
    print(f"designs: {frame.shape}")
    print(f"by round: {frame['round'].value_counts().to_dict()}")
    print(f"binders: {int(frame['binds'].sum())}")
    print(f"kd-labeled: {int(frame['kd'].notna().sum())}")
    print(f"binding_strength: {frame['binding_strength'].value_counts().to_dict()}")
