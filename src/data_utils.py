"""Shared data-cleaning helpers (used by the notebooks, training script and Streamlit app)."""
import re

import numpy as np
import pandas as pd

TARGET = "pcos"
ID_COLS = ["sl_no", "patient_file_no"]

# normalised raw header (stripped, single spaces) -> clean snake_case name
RENAME_MAP = {
    "Sl. No": "sl_no",
    "Patient File No.": "patient_file_no",
    "PCOS (Y/N)": "pcos",
    "Age (yrs)": "age_yrs",
    "Weight (Kg)": "weight_kg",
    "Height(Cm)": "height_cm",
    "BMI": "bmi",
    "Blood Group": "blood_group",
    "Pulse rate(bpm)": "pulse_rate_bpm",
    "RR (breaths/min)": "resp_rate",
    "Hb(g/dl)": "hb_g_dl",
    "Cycle(R/I)": "cycle_ri",
    "Cycle length(days)": "cycle_length_days",
    "Marraige Status (Yrs)": "marriage_yrs",
    "Pregnant(Y/N)": "pregnant",
    "No. of abortions": "n_abortions",
    "I beta-HCG(mIU/mL)": "beta_hcg_1",
    "II beta-HCG(mIU/mL)": "beta_hcg_2",
    "FSH(mIU/mL)": "fsh",
    "LH(mIU/mL)": "lh",
    "FSH/LH": "fsh_lh_ratio",
    "Hip(inch)": "hip_inch",
    "Waist(inch)": "waist_inch",
    "Waist:Hip Ratio": "waist_hip_ratio",
    "TSH (mIU/L)": "tsh",
    "AMH(ng/mL)": "amh",
    "PRL(ng/mL)": "prl",
    "Vit D3 (ng/mL)": "vit_d3",
    "PRG(ng/mL)": "prg",
    "RBS(mg/dl)": "rbs",
    "Weight gain(Y/N)": "weight_gain",
    "hair growth(Y/N)": "hair_growth",
    "Skin darkening (Y/N)": "skin_darkening",
    "Hair loss(Y/N)": "hair_loss",
    "Pimples(Y/N)": "pimples",
    "Fast food (Y/N)": "fast_food",
    "Reg.Exercise(Y/N)": "reg_exercise",
    "BP _Systolic (mmHg)": "bp_systolic",
    "BP _Diastolic (mmHg)": "bp_diastolic",
    "Follicle No. (L)": "follicle_l",
    "Follicle No. (R)": "follicle_r",
    "Avg. F size (L) (mm)": "avg_f_size_l",
    "Avg. F size (R) (mm)": "avg_f_size_r",
    "Endometrium (mm)": "endometrium",
}

# columns that are numeric but were read as text because of typos in the raw file
TEXT_NUMERIC_COLS = ["beta_hcg_2", "amh"]


def normalise_header(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip/collapse whitespace in headers and map them to clean snake_case names."""
    return df.rename(columns=lambda c: RENAME_MAP.get(normalise_header(c), normalise_header(c)))


def coerce_numeric_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in TEXT_NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
