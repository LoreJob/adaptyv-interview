"""Feature engineering for the Round Advisor POC.

Two feature families:

1. **Sequence physicochemical** features computed from the amino-acid sequence
   with Biopython's ProtParam (``Bio.SeqUtils.ProtParam.ProteinAnalysis``):
   length, molecular weight, aromaticity, instability index, isoelectric point,
   GRAVY hydropathy, secondary-structure fractions, net charge at pH 7, and a
   few residue-group fractions.

2. **Pre-computed structural / model metrics** already present in the Adaptyv
   processed tables and carried through by :mod:`src.data_loader`:
   ``pae_interaction``, ``plddt``, ``iptm``, ``esm_pll``, ``similarity_check``.
   These are strong signals (AF2 interface confidence, ESM likelihood) and are
   reused directly rather than recomputed. Note ``iptm``/``esm_pll`` are round-2
   only and are NaN for round 1.

Public entry point: :func:`load_features`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

if __package__ in (None, ""):  # allow `python src/features.py` and IDE Run
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data_loader import PROCESSED_DIR, load_designs

FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# Metrics already computed by Adaptyv, reused as-is (round-2 only ones are NaN
# for round 1 and get imputed downstream in the model, not here).
PRECOMPUTED_FEATURES = [
    "pae_interaction",
    "plddt",
    "iptm",
    "esm_pll",
    "similarity_check",
]

# Physicochemical feature names produced by :func:`_sequence_features`.
SEQUENCE_FEATURES = [
    "seq_length",
    "molecular_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy",
    "helix_fraction",
    "turn_fraction",
    "sheet_fraction",
    "charge_at_ph7",
    "frac_hydrophobic",
    "frac_positive",
    "frac_negative",
    "frac_aromatic",
    "frac_cysteine",
]

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
_HYDROPHOBIC = set("AVLIMFWC")
_POSITIVE = set("KRH")
_NEGATIVE = set("DE")
_AROMATIC = set("FWY")


def _clean_sequence(seq: str) -> str:
    """Uppercase, strip whitespace, drop any non-standard residues."""
    s = "".join(c for c in str(seq).upper() if c in _STANDARD_AA)
    return s


def _sequence_features(seq: str) -> dict[str, float]:
    """Physicochemical descriptors for one protein sequence."""
    s = _clean_sequence(seq)
    if not s:
        return {k: np.nan for k in SEQUENCE_FEATURES}

    pa = ProteinAnalysis(s)
    helix, turn, sheet = pa.secondary_structure_fraction()
    n = len(s)

    return {
        "seq_length": float(n),
        "molecular_weight": pa.molecular_weight(),
        "aromaticity": pa.aromaticity(),
        "instability_index": pa.instability_index(),
        "isoelectric_point": pa.isoelectric_point(),
        "gravy": pa.gravy(),
        "helix_fraction": helix,
        "turn_fraction": turn,
        "sheet_fraction": sheet,
        "charge_at_ph7": pa.charge_at_pH(7.0),
        "frac_hydrophobic": sum(c in _HYDROPHOBIC for c in s) / n,
        "frac_positive": sum(c in _POSITIVE for c in s) / n,
        "frac_negative": sum(c in _NEGATIVE for c in s) / n,
        "frac_aromatic": sum(c in _AROMATIC for c in s) / n,
        "frac_cysteine": s.count("C") / n,
    }


def build_sequence_features(sequences: pd.Series) -> pd.DataFrame:
    """Compute the physicochemical feature block for a series of sequences."""
    rows = [_sequence_features(s) for s in sequences]
    return pd.DataFrame(rows, index=sequences.index, columns=SEQUENCE_FEATURES)


def sequences_to_feature_frame(sequences: list[str]) -> pd.DataFrame:
    """Build a model-ready feature frame for novel sequences.

    Novel (proposed) sequences have no AlphaFold/ESM metrics, so the
    ``PRECOMPUTED_FEATURES`` columns are filled with NaN and imputed downstream
    by the model pipeline. Only the physicochemical block is real.
    """
    seq = pd.Series(list(sequences))
    block = build_sequence_features(seq)
    for col in PRECOMPUTED_FEATURES:
        block[col] = np.nan
    return block[SEQUENCE_FEATURES + PRECOMPUTED_FEATURES]


def load_features(cache: bool = True) -> pd.DataFrame:
    """Return designs joined with all model features.

    Output = the unified design table (labels + metadata) plus the
    physicochemical feature block. ``SEQUENCE_FEATURES + PRECOMPUTED_FEATURES``
    are the model-ready columns.
    """
    if cache and FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)

    designs = load_designs()
    seq_block = build_sequence_features(designs["sequence"])
    df = pd.concat([designs, seq_block], axis=1)

    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEATURES_PATH, index=False)

    return df


FEATURE_COLUMNS = SEQUENCE_FEATURES + PRECOMPUTED_FEATURES


if __name__ == "__main__":
    df = load_features(cache=False)
    print(f"rows: {len(df)}  feature cols: {len(FEATURE_COLUMNS)}")
    print(df[FEATURE_COLUMNS].describe().T[["mean", "min", "max"]].round(3).to_string())
    print("\nNaN per feature:")
    print(df[FEATURE_COLUMNS].isna().sum().to_string())
