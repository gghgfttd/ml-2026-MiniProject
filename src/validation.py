"""Shared validation for all models in the project.

The module knows nothing about specific models or the dataset: it takes any
sklearn-like estimator (fit/predict) and an X DataFrame sorted by time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# random : shuffled k-fold (neighbouring hours end up in train and test)
# blocked: contiguous k-fold, no shuffling (same fold sizes as "random", but each test fold is one continuous block of time).
# chrono : expanding window, test always after train (see DateSplit)
SCHEMES = ("random", "blocked", "chrono")


class DateSplit:
    """Expanding-window chronological splits defined by calendar time.

    The first min_train_frac of the time span is always training data. The
    rest is cut into n_splits test blocks of equal duration. Fold k trains on
    everything before block k and tests on block k. With the default 0.5 and
    two years of data, the first fold trains on the whole first year, so every
    season has been seen before any test block.

    Blocks are equal in time, not in row count (some hours are missing).
    Requires X with a DatetimeIndex sorted in time. The same class is used as
    the inner CV of GridSearchCV, where it applies to the training part only.
    """

    def __init__(self, n_splits=5, min_train_frac=0.5):
        self.n_splits = n_splits
        self.min_train_frac = min_train_frac

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        t = X.index
        if not isinstance(t, pd.DatetimeIndex):
            raise ValueError("DateSplit needs X with a DatetimeIndex")
        start, end = t.min(), t.max()
        first_cut = start + (end - start) * self.min_train_frac
        step = (end - first_cut) / self.n_splits
        for k in range(self.n_splits):
            cut = first_cut + k * step
            upper = first_cut + (k + 1) * step
            is_last = k == self.n_splits - 1
            train = np.flatnonzero(t < cut)
            test = np.flatnonzero((t >= cut) & ((t < upper) | is_last))
            yield train, test


def get_cv(scheme, n_splits=5, seed=SEED):
    """Splitter for the given scheme.

    Used both for the outer evaluation (evaluate) and for the inner
    hyperparameter search, so the two always match:
        GridSearchCV(pipe, grid, cv=get_cv(scheme, seed=seed))
    """
    if scheme == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if scheme == "blocked":
        return KFold(n_splits=n_splits, shuffle=False)
    if scheme == "chrono":
        return DateSplit(n_splits=n_splits)
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def _same_scheme(a, b):
    """True if two splitters implement the same scheme."""
    return type(a) is type(b) and getattr(a, "shuffle", None) == getattr(b, "shuffle", None)


def compute_metrics(y_true, y_pred):
    """RMSE, MAE, R2 and RMSLE (negative predictions are clipped to 0 for RMSLE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmsle = np.sqrt(
        mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))
    )
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmsle": rmsle,
    }


def evaluate(model, X, y, scheme, name, n_splits=5, seed=SEED, save=True):
    """Evaluate a model under a scheme and (by default) save the results.

    model  : pipeline or GridSearchCV; it is cloned and fitted from scratch on
             every fold, so nothing from the test part leaks into training.
             If it is a GridSearchCV, its inner cv must use the same scheme
             (build the model separately for each scheme).
    X, y   : DataFrame/Series in time order (as returned by load_data).
    scheme : "random", "blocked" or "chrono".
    name   : model name used in the output file names.

    Returns (folds, preds):
      folds : per-fold table (sizes, test period, metrics, best_params)
      preds : y_true, y_pred and fold number, indexed by datetime
    Files: results/{name}_{scheme}_folds.csv and results/{name}_{scheme}_preds.csv
    """
    if not X.index.equals(y.index):
        raise ValueError("Indices of X and y do not match")
    if scheme in ("chrono", "blocked") and not X.index.is_monotonic_increasing:
        raise ValueError(f"Scheme {scheme!r} needs X sorted by time")

    cv = get_cv(scheme, n_splits, seed)

    inner_cv = getattr(model, "cv", None)
    if inner_cv is not None and not _same_scheme(inner_cv, cv):
        raise ValueError(
            f"The model's inner cv ({inner_cv!r}) does not match scheme "
            f"{scheme!r}. Build the model with cv=get_cv({scheme!r})."
        )

    fold_rows, pred_parts = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        est = clone(model)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = est.predict(X.iloc[test_idx])
        y_true = y.iloc[test_idx]

        row = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "test_start": X.index[test_idx].min(),
            "test_end": X.index[test_idx].max(),
            **compute_metrics(y_true, y_pred),
        }
        if hasattr(est, "best_params_"):  # GridSearchCV
            row["best_params"] = str(est.best_params_)
        fold_rows.append(row)

        pred_parts.append(
            pd.DataFrame(
                {"fold": fold, "y_true": y_true.to_numpy(), "y_pred": y_pred},
                index=X.index[test_idx],
            )
        )

    folds = pd.DataFrame(fold_rows)
    preds = pd.concat(pred_parts).sort_index()
    preds.index.name = "datetime"

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        folds.to_csv(RESULTS_DIR / f"{name}_{scheme}_folds.csv", index=False)
        preds.to_csv(RESULTS_DIR / f"{name}_{scheme}_preds.csv")

    return folds, preds