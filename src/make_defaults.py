"""Write model/feature_defaults.json: the 'typical' value of each feature, taken from the TRAIN set only.

The Streamlit check-in uses these for any optional field a user leaves blank.
Run from the repo root:  python src/make_defaults.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "processed" / "train.csv").drop(columns="pcos")

defaults = {}
for col in train.columns:
    s = train[col]
    is_flag = set(s.unique()) <= {0, 1}
    is_code = col in ("blood_group", "cycle_ri")
    defaults[col] = float(s.mode().iloc[0]) if (is_flag or is_code) else round(float(s.median()), 3)

out = ROOT / "model" / "feature_defaults.json"
out.write_text(json.dumps(defaults, indent=2))
print(f"wrote {out.relative_to(ROOT)} ({len(defaults)} features)")
