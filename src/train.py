"""Train + evaluate: Logistic Regression baseline vs. Optuna-tuned, calibrated XGBoost.

Split discipline (stratified 60 / 20 / 20):
  * train      -> model fitting, 5-fold CV for hyper-parameter search
  * validation -> probability calibration + choosing decision thresholds
  * test       -> final reported numbers only (never used for any choice)

Usage:  python -m src.train [--trials N]
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import cli_ui
from src.config import load_settings, resolve
from src.data import TARGET, all_feature_names, feature_columns, prepare_dataset

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Splits + pipelines
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    val_frac = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=train_val[TARGET], random_state=seed
    )
    return train, val, test


def make_preprocessor(groups: dict[str, list[str]], scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale_numeric else "passthrough", groups["numeric"]),
            ("bin", "passthrough", groups["binary"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                groups["categorical"],
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_logreg(groups: dict[str, list[str]], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(groups, scale_numeric=True)),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed),
            ),
        ]
    )


def make_xgb(groups: dict[str, list[str]], params: dict[str, Any], seed: int) -> Pipeline:
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    return Pipeline([("prep", make_preprocessor(groups, scale_numeric=False)), ("clf", clf)])


def tune_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    n_trials: int,
    cv_folds: int,
    seed: int,
    callbacks: list | None = None,
) -> tuple[dict[str, Any], float]:
    """Optuna TPE search maximising mean CV PR-AUC on the training split only."""
    neg_pos = float((y == 0).sum() / max((y == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, neg_pos),
        }
        model = make_xgb(groups, params, seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def f1_optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall have one more element than thresholds
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def top_k_lift(y_true: np.ndarray, proba: np.ndarray, frac: float = 0.10) -> float:
    """Churn rate in the top ``frac`` of scores divided by the overall churn rate."""
    k = max(int(round(len(proba) * frac)), 1)
    top = np.argsort(-proba, kind="stable")[:k]
    return float(y_true[top].mean() / y_true.mean())


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "flagged_share": float(pred.mean()),
        "top_decile_lift": top_k_lift(y_true, proba, 0.10),
    }


def paired_bootstrap(
    y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, n_iter: int, seed: int
) -> dict[str, dict[str, float]]:
    """95% CI for (model B - model A) on PR-AUC and ROC-AUC over test-set resamples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs: dict[str, list[float]] = {"pr_auc": [], "roc_auc": []}
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        diffs["pr_auc"].append(
            average_precision_score(yt, proba_b[idx]) - average_precision_score(yt, proba_a[idx])
        )
        diffs["roc_auc"].append(roc_auc_score(yt, proba_b[idx]) - roc_auc_score(yt, proba_a[idx]))
    out = {}
    for k, v in diffs.items():
        arr = np.asarray(v)
        out[k] = {
            "mean_diff": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "p_b_better": float((arr > 0).mean()),
        }
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# Reference categorical palette (light mode), fixed slot order.
_SERIES = {"xgb_cal": "#2a78d6", "logreg": "#eb6834", "xgb_raw": "#1baf7a"}
_SURFACE, _INK, _INK_2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)


def save_figures(
    y_test: np.ndarray, curves: dict[str, tuple[str, np.ndarray]], figures_dir
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    prevalence = float(y_test.mean())

    # PR curve
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=_SURFACE)
    _style_axes(ax)
    for key in ("xgb_cal", "logreg"):
        label, proba = curves[key]
        p, r, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(r, p, color=_SERIES[key], linewidth=2, label=f"{label}  (PR-AUC {ap:.3f})")
    ax.axhline(prevalence, color=_INK_2, linewidth=1, linestyle="--")
    ax.text(
        0.01,
        prevalence + 0.015,
        f"No-skill = churn rate {prevalence:.2f}",
        color=_INK_2,
        fontsize=8,
    )
    ax.set_xlabel("Recall (share of churners caught)")
    ax.set_ylabel("Precision (share of flagged who churn)")
    ax.set_title("Precision-recall on held-out test set", color=_INK, fontsize=11, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=_INK)
    fig.tight_layout()
    path = figures_dir / "pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # ROC curve
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=_SURFACE)
    _style_axes(ax)
    for key in ("xgb_cal", "logreg"):
        label, proba = curves[key]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=_SERIES[key], linewidth=2, label=f"{label}  (ROC-AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=_INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC on held-out test set", color=_INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=_INK)
    fig.tight_layout()
    path = figures_dir / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=_SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=_INK_2, linewidth=1, linestyle="--", label="Perfectly calibrated")
    for key in ("xgb_cal", "xgb_raw", "logreg"):
        label, proba = curves[key]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_test, proba)
        ax.plot(
            mean_pred,
            frac_pos,
            color=_SERIES[key],
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{label}  (Brier {brier:.3f})",
        )
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration on held-out test set", color=_INK, fontsize=11, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=_INK)
    fig.tight_layout()
    path = figures_dir / "calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(n_trials: int | None = None, write_figures: bool = True) -> dict[str, Any]:
    settings = load_settings()
    seed = settings["seed"]
    n_trials = n_trials or settings["model"]["optuna_trials"]

    cli_ui.banner("Train", "logistic-regression baseline vs tuned, calibrated XGBoost")
    with cli_ui.spinner("Loading and cleaning data"):
        df = prepare_dataset(settings)
    source = df.attrs.get("source")
    groups = feature_columns(df, settings)
    features = all_feature_names(groups)
    train, val, test = split_data(
        df, settings["split"]["test_size"], settings["split"]["val_size"], seed
    )
    log.info(
        "source=%s train=%d val=%d test=%d features=%d",
        source,
        len(train),
        len(val),
        len(test),
        len(features),
    )

    X_tr, y_tr = train[features], train[TARGET].to_numpy()
    X_va, y_va = val[features], val[TARGET].to_numpy()
    X_te, y_te = test[features], test[TARGET].to_numpy()

    # Baseline
    with cli_ui.spinner("Fitting logistic-regression baseline"):
        logreg = make_logreg(groups, seed).fit(X_tr, y_tr)
        lr_cv = cross_val_score(
            make_logreg(groups, seed),
            X_tr,
            y_tr,
            cv=StratifiedKFold(settings["model"]["cv_folds"], shuffle=True, random_state=seed),
            scoring="average_precision",
        ).mean()

    # Tuned XGBoost
    with cli_ui.trial_progress(n_trials) as on_trial:
        best_params, xgb_cv = tune_xgb(
            X_tr,
            pd.Series(y_tr),
            groups,
            n_trials,
            settings["model"]["cv_folds"],
            seed,
            callbacks=[on_trial],
        )
    xgb_model = make_xgb(groups, best_params, seed).fit(X_tr, y_tr)

    # Calibrate on the validation split (model weights frozen). The configured
    # method is the final model; the other method is kept as a reference row.
    cal_method = settings["model"].get("calibration_method", "sigmoid")
    alt_method = "isotonic" if cal_method == "sigmoid" else "sigmoid"
    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb_model), method=cal_method)
    calibrated.fit(X_va, y_va)
    calibrated_alt = CalibratedClassifierCV(FrozenEstimator(xgb_model), method=alt_method)
    calibrated_alt.fit(X_va, y_va)

    # Probabilities
    p_va = {
        "logreg": logreg.predict_proba(X_va)[:, 1],
        "xgb_raw": xgb_model.predict_proba(X_va)[:, 1],
        "xgb_cal": calibrated.predict_proba(X_va)[:, 1],
        "xgb_cal_alt": calibrated_alt.predict_proba(X_va)[:, 1],
    }
    p_te = {
        "logreg": logreg.predict_proba(X_te)[:, 1],
        "xgb_raw": xgb_model.predict_proba(X_te)[:, 1],
        "xgb_cal": calibrated.predict_proba(X_te)[:, 1],
        "xgb_cal_alt": calibrated_alt.predict_proba(X_te)[:, 1],
    }
    labels = {
        "logreg": "Logistic Regression (baseline)",
        "xgb_raw": "XGBoost tuned (uncalibrated)",
        "xgb_cal": f"XGBoost tuned + {cal_method} (final)",
        "xgb_cal_alt": f"XGBoost tuned + {alt_method} (reference)",
    }

    models: dict[str, Any] = {}
    for key in ("logreg", "xgb_raw", "xgb_cal", "xgb_cal_alt"):
        thr_f1 = f1_optimal_threshold(y_va, p_va[key])  # chosen on VALIDATION
        models[key] = {
            "label": labels[key],
            "test_at_0.50": evaluate(y_te, p_te[key], 0.50),
            "test_at_val_f1_threshold": evaluate(y_te, p_te[key], thr_f1),
            "validation_pr_auc": float(average_precision_score(y_va, p_va[key])),
        }
    models["logreg"]["cv_pr_auc_train"] = float(lr_cv)
    models["xgb_raw"]["cv_pr_auc_train"] = float(xgb_cv)
    models["xgb_cal"]["cv_pr_auc_train"] = float(xgb_cv)
    models["xgb_cal_alt"]["cv_pr_auc_train"] = float(xgb_cv)

    with cli_ui.spinner("Paired bootstrap (XGBoost vs baseline)"):
        comparison = paired_bootstrap(
            y_te, p_te["logreg"], p_te["xgb_cal"], settings["model"]["bootstrap_iterations"], seed
        )

    # Persist artifacts
    models_dir = resolve(settings["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, models_dir / "logreg.joblib")
    joblib.dump(xgb_model, models_dir / "xgb.joblib")
    joblib.dump(calibrated, models_dir / "xgb_calibrated.joblib")
    joblib.dump(
        {"groups": groups, "features": features, "source": source},
        models_dir / "feature_spec.joblib",
    )

    # Score every customer with the final model; `split` lets the app / ROI
    # step restrict realised-profit numbers to the untouched test set.
    scored = df.copy()
    scored["split"] = "train"
    scored.loc[val.index, "split"] = "validation"
    scored.loc[test.index, "split"] = "test"
    scored["p_churn"] = calibrated.predict_proba(scored[features])[:, 1]
    scored["p_churn_raw"] = xgb_model.predict_proba(scored[features])[:, 1]
    scored["p_churn_logreg"] = logreg.predict_proba(scored[features])[:, 1]
    scored_path = resolve(settings["paths"]["scored_file"])
    scored.to_parquet(scored_path, index=False)

    figures = []
    if write_figures:
        figures = save_figures(
            y_te,
            {k: (labels[k], p_te[k]) for k in ("logreg", "xgb_raw", "xgb_cal")},
            resolve(settings["paths"]["figures_dir"]),
        )

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "rows": {"total": len(df), "train": len(train), "validation": len(val), "test": len(test)},
        "churn_rate": {
            "overall": float(df[TARGET].mean()),
            "test": float(y_te.mean()),
        },
        "n_features": len(features),
        "features": groups,
        "excluded_leakage_columns": [
            c for c in settings["features"]["leakage_blocklist"] if c in df.columns
        ],
        "calibration_method": cal_method,
        "optuna": {"trials": n_trials, "best_cv_pr_auc": xgb_cv, "best_params": best_params},
        "models": models,
        "xgb_cal_minus_logreg_test_bootstrap": comparison,
        "figures": figures,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "optuna": optuna.__version__,
        },
    }
    metrics_path = resolve(settings["paths"]["metrics_file"])
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _print_summary(metrics)
    cli_ui.summary_panel("Training complete", [metrics_path, scored_path, models_dir, *figures])
    return metrics


def _print_summary(metrics: dict[str, Any]) -> None:
    rows = []
    for m in metrics["models"].values():
        a, b = m["test_at_0.50"], m["test_at_val_f1_threshold"]
        rows.append(
            {
                "model": m["label"],
                "PR-AUC": a["pr_auc"],
                "ROC-AUC": a["roc_auc"],
                "Brier": a["brier"],
                "Recall@0.50": a["recall"],
                "F1@0.50": a["f1"],
                "thr(val-F1)": b["threshold"],
                "Recall@thr": b["recall"],
                "F1@thr": b["f1"],
                "Lift@10%": a["top_decile_lift"],
            }
        )
    cli_ui.df_table(pd.DataFrame(rows), "Model comparison (held-out test set)")
    boot = metrics["xgb_cal_minus_logreg_test_bootstrap"]
    cli_ui.df_table(
        pd.DataFrame(
            [
                {
                    "metric": k,
                    "XGB - LR": v["mean_diff"],
                    "95% CI low": v["ci_low"],
                    "95% CI high": v["ci_high"],
                    "P(XGB better)": v["p_b_better"],
                }
                for k, v in boot.items()
            ]
        ),
        "Paired bootstrap: final XGBoost minus baseline",
        {"XGB - LR": "+.4f", "95% CI low": "+.4f", "95% CI high": "+.4f"},
    )


def main() -> None:
    parser = cli_ui.base_parser(__doc__)
    parser.add_argument("--trials", type=int, default=None, help="Optuna trials (default: config)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    cli_ui.setup(args.quiet)
    run(n_trials=args.trials, write_figures=not args.no_figures)


if __name__ == "__main__":
    main()
