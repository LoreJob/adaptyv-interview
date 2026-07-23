"""Predictive models for the Round Advisor POC.

Two heads over the harmonized feature table (see :mod:`src.features`):

* **Binder classifier** — binder vs non-binder on the full pooled set
  (~604 designs, 63 positives). Reported with ROC-AUC, average precision, and
  precision@k under stratified cross-validation. This is the primary signal the
  active-learning backtest ranks on, because binding labels exist for every
  design while KD does not.
* **KD regressor** — log10(KD) on the ~63 KD-labeled designs only. Reported
  with Spearman correlation under cross-validation (not R^2: small n, outliers).

:class:`BootstrapEnsemble` wraps either head to produce a mean prediction plus a
disagreement-based uncertainty, consumed by :mod:`src.active_learning` for the
UCB acquisition function.

The data is small; models are deliberately simple (random forests) and the
emphasis is honest, cross-validated reporting rather than squeezing accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

if __package__ in (None, ""):  # allow `python src/model.py` and IDE Run
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.features import FEATURE_COLUMNS, load_features

RANDOM_STATE = 42


# --- data prep ------------------------------------------------------------

def prepare_xy(
    df: pd.DataFrame, target: str = "binds"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for a target.

    ``target='binds'`` uses all rows. ``target='log10_kd'`` keeps only the
    KD-labeled rows. Imputation of missing features is left to the model
    pipeline so cross-validation stays leak-free.
    """
    X = df[FEATURE_COLUMNS].copy()
    if target == "log10_kd":
        mask = df["log10_kd"].notna()
        return X.loc[mask], df.loc[mask, "log10_kd"]
    return X, df[target].astype(int)


def _classifier_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def _regressor_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# --- bootstrap ensemble (mean + uncertainty) ------------------------------

class BootstrapEnsemble:
    """Bootstrap-resampled ensemble giving a mean prediction and its spread.

    ``predict_mean_std`` returns (mean, std) across ensemble members; the std is
    the epistemic-uncertainty estimate the UCB acquisition adds to the mean.
    """

    def __init__(self, base_pipeline: Pipeline, n_estimators: int = 25,
                 classifier: bool = True, random_state: int = RANDOM_STATE):
        self.base_pipeline = base_pipeline
        self.n_estimators = n_estimators
        self.classifier = classifier
        self.random_state = random_state
        self.members_: list[Pipeline] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BootstrapEnsemble":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        self.members_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)  # sample with replacement
            # A resample may miss a class; retry a few times for classifiers.
            for _retry in range(5):
                if not self.classifier or len(np.unique(y[idx])) > 1:
                    break
                idx = rng.integers(0, n, size=n)
            member = clone(self.base_pipeline)
            member.fit(X.iloc[idx], y[idx])
            self.members_.append(member)
        return self

    def _member_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self.classifier:
            return np.column_stack([m.predict_proba(X)[:, 1] for m in self.members_])
        return np.column_stack([m.predict(X) for m in self.members_])

    def predict_mean_std(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scores = self._member_scores(X)
        return scores.mean(axis=1), scores.std(axis=1)


# --- evaluation -----------------------------------------------------------

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of true positives among the top-k highest-scored items."""
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[order]))


@dataclass
class ClassifierReport:
    roc_auc: float
    avg_precision: float
    precision_at_20: float
    precision_at_50: float
    n: int
    n_pos: int


def evaluate_classifier(df: pd.DataFrame, n_splits: int = 5) -> ClassifierReport:
    """Cross-validated binder-classifier metrics on out-of-fold predictions."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    X, y = prepare_xy(df, "binds")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(
        _classifier_pipeline(), X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    y = y.to_numpy()
    return ClassifierReport(
        roc_auc=float(roc_auc_score(y, oof)),
        avg_precision=float(average_precision_score(y, oof)),
        precision_at_20=precision_at_k(y, oof, 20),
        precision_at_50=precision_at_k(y, oof, 50),
        n=len(y),
        n_pos=int(y.sum()),
    )


@dataclass
class RegressorReport:
    spearman: float
    spearman_p: float
    n: int


def evaluate_regressor(df: pd.DataFrame, n_splits: int = 5) -> RegressorReport:
    """Cross-validated KD-regressor Spearman on out-of-fold predictions."""
    from sklearn.model_selection import KFold

    X, y = prepare_xy(df, "log10_kd")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(_regressor_pipeline(), X, y, cv=cv, n_jobs=-1)
    rho, p = spearmanr(y, oof)
    return RegressorReport(spearman=float(rho), spearman_p=float(p), n=len(y))


# --- fitted-model factory (for the agent / API) ---------------------------

def fit_binder_ensemble(df: pd.DataFrame, n_estimators: int = 25) -> BootstrapEnsemble:
    X, y = prepare_xy(df, "binds")
    return BootstrapEnsemble(
        _classifier_pipeline(), n_estimators=n_estimators, classifier=True
    ).fit(X, y)


if __name__ == "__main__":
    df = load_features()
    clf = evaluate_classifier(df)
    print("Binder classifier (5-fold CV):")
    print(f"  n={clf.n}  positives={clf.n_pos}")
    print(f"  ROC-AUC        {clf.roc_auc:.3f}")
    print(f"  Avg precision  {clf.avg_precision:.3f}")
    print(f"  Precision@20   {clf.precision_at_20:.3f}")
    print(f"  Precision@50   {clf.precision_at_50:.3f}")

    reg = evaluate_regressor(df)
    print("\nKD regressor (log10 KD, 5-fold CV):")
    print(f"  n={reg.n}")
    print(f"  Spearman       {reg.spearman:.3f}  (p={reg.spearman_p:.2g})")
