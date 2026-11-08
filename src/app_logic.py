"""Non-UI logic for the Streamlit app: loading saved models, building a feature row from
check-in answers, scoring, and evaluating uploaded test data. No retraining happens here."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from .data_utils import ID_COLS, TARGET, coerce_numeric_text, rename_columns

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
HEADLINE_MODEL = "Logistic Regression"          # best model on the held-out test set
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

FRIENDLY = {
    "age_yrs": "Your age", "weight_kg": "Weight", "height_cm": "Height", "bmi": "Body mass index (BMI)",
    "blood_group": "Blood group", "pulse_rate_bpm": "Resting pulse", "resp_rate": "Breathing rate",
    "hb_g_dl": "Haemoglobin", "cycle_ri": "Cycle regularity", "cycle_length_days": "Length of your period",
    "marriage_yrs": "Years married", "pregnant": "Pregnancy status", "n_abortions": "Past miscarriages / abortions",
    "beta_hcg_1": "β-hCG (1st test)", "beta_hcg_2": "β-hCG (2nd test)", "fsh": "FSH level", "lh": "LH level",
    "fsh_lh_ratio": "FSH : LH ratio", "hip_inch": "Hip size", "waist_inch": "Waist size",
    "waist_hip_ratio": "Waist-to-hip ratio", "tsh": "Thyroid (TSH)", "amh": "AMH level", "prl": "Prolactin",
    "vit_d3": "Vitamin D3", "prg": "Progesterone", "rbs": "Blood sugar (random)",
    "weight_gain": "Unexplained weight gain", "hair_growth": "Extra facial / body hair",
    "skin_darkening": "Skin darkening", "hair_loss": "Hair thinning", "pimples": "Acne / pimples",
    "fast_food": "Fast-food habit", "reg_exercise": "Regular exercise", "bp_systolic": "Blood pressure (upper)",
    "bp_diastolic": "Blood pressure (lower)", "follicle_l": "Follicle count – left ovary",
    "follicle_r": "Follicle count – right ovary", "avg_f_size_l": "Follicle size – left",
    "avg_f_size_r": "Follicle size – right", "endometrium": "Endometrial thickness",
}
LAB_KEYS = {"fsh", "lh", "tsh", "amh", "prl", "vit_d3", "prg", "rbs", "beta_hcg_1", "beta_hcg_2"}
SCAN_KEYS = {"follicle_l", "follicle_r", "avg_f_size_l", "avg_f_size_r", "endometrium"}


# ----------------------------------------------------------------------------- loading
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


def load_feature_order() -> list[str]:
    return list(joblib.load(MODEL_DIR / "feature_config.joblib")["feature_order"])


def load_defaults() -> dict:
    return json.loads((MODEL_DIR / "feature_defaults.json").read_text())


# ----------------------------------------------------------------------------- check-in
def build_feature_row(core: dict, optional: dict, defaults: dict, features: list[str]):
    """Merge check-in answers over the typical values. Returns (1-row DataFrame, set of features the user supplied)."""
    vals = dict(defaults)
    provided = set(core)
    vals.update(core)
    for k, v in optional.items():
        if v is not None:
            vals[k] = v
            provided.add(k)

    vals["bmi"] = vals["weight_kg"] / (vals["height_cm"] / 100) ** 2
    provided.add("bmi")
    vals["waist_hip_ratio"] = vals["waist_inch"] / vals["hip_inch"]
    if {"waist_inch", "hip_inch"} & provided:
        provided.add("waist_hip_ratio")
    vals["fsh_lh_ratio"] = vals["fsh"] / max(vals["lh"], 0.01)
    if {"fsh", "lh"} & provided:
        provided.add("fsh_lh_ratio")
    return pd.DataFrame([vals])[features], provided


def _base_feature(col: str) -> str:
    for prefix in ("blood_group_", "cycle_ri_"):
        if col.startswith(prefix):
            return prefix[:-1]
    return col


def factor_contributions(pipe, row: pd.DataFrame, provided: set) -> pd.Series:
    """Per-feature push on the log-odds from the (linear) Logistic Regression, relative to the typical
    woman in the training data. Only features the user actually supplied are returned."""
    xt = pipe.named_steps["prep"].transform(row)
    contrib = pd.Series(xt.iloc[0].to_numpy() * pipe.named_steps["clf"].coef_[0], index=xt.columns)
    grouped = contrib.groupby(_base_feature).sum()
    return grouped[grouped.index.isin(provided)].sort_values()


def assess(models: dict, defaults: dict, features: list[str], core: dict, optional: dict) -> dict:
    row, provided = build_feature_row(core, optional, defaults, features)
    probs = {name: float(m.predict_proba(row)[0, 1]) for name, m in models.items()}
    given_optional = {k for k, v in optional.items() if v is not None}
    if given_optional & {"follicle_l", "follicle_r"}:
        level = "Detailed"
    elif given_optional & LAB_KEYS:
        level = "Good"
    else:
        level = "Basic"
    return {
        "probs": probs,
        "contrib": factor_contributions(models[HEADLINE_MODEL], row, provided),
        "level": level,
        "n_provided": len(provided),
        "n_total": len(features),
        "bmi": float(row["bmi"].iloc[0]),
        "age": float(row["age_yrs"].iloc[0]),
    }


# ----------------------------------------------------------------------------- evaluation
def validate_upload(raw: pd.DataFrame, features: list[str]):
    """Return (clean_df | None, errors, notes). Accepts the cleaned snake_case headers or the original Kaggle ones."""
    errors, notes = [], []
    df = coerce_numeric_text(rename_columns(raw))
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

    missing = [c for c in features if c not in df.columns]
    if missing:
        errors.append(f"Missing {len(missing)} feature column(s): {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}")
    if TARGET not in df.columns:
        errors.append(f"Missing the target column `{TARGET}` (0 = no PCOS, 1 = PCOS).")
    if errors:
        return None, errors, notes

    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < n_before:
        notes.append(f"Ignored {n_before - len(df)} row(s) with an empty target.")
    if not df[TARGET].isin([0, 1]).all():
        errors.append(f"`{TARGET}` must contain only 0 and 1.")
        return None, errors, notes
    if df.empty:
        errors.append("The file has no usable rows.")
        return None, errors, notes

    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    n_blank = int(df[features].isna().sum().sum())
    if n_blank:
        notes.append(f"{n_blank} blank/non-numeric cell(s) were filled with the training medians by the saved pipeline.")
    extra = [c for c in df.columns if c not in features + [TARGET]]
    if extra:
        notes.append(f"Ignored {len(extra)} extra column(s): {', '.join(extra[:5])}{' …' if len(extra) > 5 else ''}")
    df[TARGET] = df[TARGET].astype(int)
    return df, errors, notes


def score_model(model, X: pd.DataFrame, y: pd.Series):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba) if y.nunique() == 2 else float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, pred), "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0), "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred, proba


def compare_models(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({name: score_model(m, X, y)[0] for name, m in models.items()}).T
    table.index.name = "Model"
    return table[METRIC_NAMES]
