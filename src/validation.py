from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SCHEMES = ("random", "chrono")


def get_cv(scheme, n_splits=5, seed=SEED):
    if scheme == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if scheme == "chrono":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError(f"scheme має бути одним із {SCHEMES}, отримано {scheme!r}")


def compute_metrics(y_true, y_pred):
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
    if not X.index.equals(y.index):
        raise ValueError("Індекси X і y не збігаються")
    if scheme == "chrono" and not X.index.is_monotonic_increasing:
        raise ValueError("Для схеми 'chrono' X має бути відсортований за часом")

    cv = get_cv(scheme, n_splits, seed)
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
        if hasattr(est, "best_params_"):  # якщо це GridSearchCV
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