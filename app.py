"""Bloom · a gentle PCOS awareness check-in + model evaluation dashboard.

Tab 1  – a private check-in for women (friendly form -> risk estimate with plain-language reasons)
Tab 2  – assignment dashboard: upload test CSV, pick a model, see metrics / confusion matrix / report
Tab 3  – about, limits and data

All five models are loaded from model/*.joblib. Nothing is retrained and nothing the user types is stored.
"""
import io
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.app_logic import (BASE, FRIENDLY, HEADLINE_MODEL, METRIC_NAMES, SCAN_KEYS, LAB_KEYS, assess,
                           compare_models, load_defaults, load_feature_order, load_models, score_model,
                           validate_upload)

st.set_page_config(page_title="Bloom · PCOS check-in", page_icon="🌸", layout="centered",
                   initial_sidebar_state="collapsed")

ROSE, PLUM, BLUSH, SAGE, GOLD = "#C94B7A", "#6B3A5B", "#FDEEF2", "#5E9C8A", "#D9922E"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Nunito:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Nunito', sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 10% -10%, #FFE3EC 0%, rgba(255,249,247,0) 60%),
                     radial-gradient(900px 500px at 100% 0%, #EFE6FA 0%, rgba(255,249,247,0) 55%), #FFF9F7; }
h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; color: #6B3A5B; letter-spacing: -.01em; }
.block-container { padding-top: 2rem; max-width: 880px; }
header[data-testid="stHeader"] { background: transparent; }

.hero { text-align: center; padding: 1.2rem 1rem 1.4rem; }
.hero .logo { font-size: 2.6rem; line-height: 1; }
.hero h1 { font-size: 2.5rem; margin: .2rem 0 .1rem; }
.hero p { color: #8a6b80; font-size: 1.05rem; margin: 0 auto; max-width: 560px; }
.pill { display: inline-block; background: #fff; border: 1px solid #F4CBD8; color: #A04068; border-radius: 999px;
        padding: .22rem .8rem; font-size: .82rem; font-weight: 700; margin: .7rem .2rem 0; }

.card { background: #fff; border: 1px solid #F6DDE5; border-radius: 22px; padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 28px rgba(201,75,122,.07); margin-bottom: 1rem; }
.card h3 { margin: 0 0 .15rem; font-size: 1.25rem; }
.card .sub { color: #9a7d90; font-size: .92rem; margin-bottom: .6rem; }

div[data-testid="stTabs"] button[role="tab"] { font-weight: 700; font-size: 1rem; color: #8a6b80; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #C94B7A; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    border-radius: 999px; border: 0; padding: .65rem 1.6rem; font-weight: 700; color: #fff;
    background: linear-gradient(135deg, #D85A89, #B8407A); box-shadow: 0 6px 18px rgba(201,75,122,.28); }
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.06); color:#fff; }
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 14px !important; }
details { border-radius: 16px !important; border: 1px solid #F6DDE5 !important; background: #fff; }

.ring-wrap { display:flex; flex-direction:column; align-items:center; gap:.6rem; padding:.4rem 0 .2rem; }
.ring { width: 190px; height: 190px; border-radius: 50%; display:flex; align-items:center; justify-content:center; }
.ring .inner { width: 142px; height: 142px; border-radius: 50%; background:#fff; display:flex; flex-direction:column;
               align-items:center; justify-content:center; box-shadow: inset 0 0 0 1px #F6DDE5; }
.ring .pct { font-family:'Fraunces',serif; font-size: 2.5rem; font-weight:600; color:#6B3A5B; line-height:1; }
.ring .lbl { font-size:.78rem; color:#9a7d90; margin-top:.2rem; }
.band { font-family:'Fraunces',serif; font-size:1.45rem; font-weight:600; }
.msg { text-align:center; color:#6f5566; max-width:520px; margin:0 auto; }

.chip { display:inline-block; border-radius:999px; padding:.3rem .8rem; margin:.15rem .1rem; font-size:.88rem; font-weight:700; }
.chip.up { background:#FCE0EA; color:#A83A68; } .chip.down { background:#E0F1EC; color:#3F7B6A; }
.bar-row { display:flex; align-items:center; gap:.7rem; margin:.32rem 0; font-size:.93rem; }
.bar-row .name { width: 46%; color:#4d3a4a; } .bar-row .track { flex:1; height:10px; background:#F7EEF1; border-radius:999px; overflow:hidden; }
.bar-row .fill { height:100%; border-radius:999px; }
.metric { background:#fff; border:1px solid #F6DDE5; border-radius:18px; padding:.8rem .5rem; text-align:center;
          box-shadow: 0 6px 18px rgba(201,75,122,.06); }
.metric .v { font-family:'Fraunces',serif; font-size:1.7rem; color:#B8407A; font-weight:600; line-height:1.1; }
.metric .k { color:#9a7d90; font-size:.8rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
.note { background:#FFF4E5; border:1px solid #F7DDB5; color:#7a5a24; border-radius:14px; padding:.7rem 1rem; font-size:.9rem; }
.soft { background:#F4EEFA; border:1px solid #E3D6F2; color:#5a4a78; border-radius:14px; padding:.7rem 1rem; font-size:.9rem; }
.foot { text-align:center; color:#b09aa8; font-size:.82rem; padding:1.4rem 0 .5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def get_resources():
    return load_models(), load_defaults(), load_feature_order()


try:
    MODELS, DEFAULTS, FEATURES = get_resources()
except Exception as exc:                                         # pragma: no cover
    st.error(f"Couldn't load the saved models from the `model/` folder: {exc}")
    st.stop()


# --------------------------------------------------------------------------- helpers
def band_for(p: float):
    if p < 0.25:
        return "Low likelihood", SAGE, "Your answers look similar to women who were not diagnosed with PCOS."
    if p < 0.50:
        return "Mild likelihood", GOLD, "A few of your answers overlap with the PCOS pattern, but most do not."
    if p < 0.75:
        return "Moderate likelihood", "#D9683F", "Several of your answers overlap with the PCOS pattern. Worth discussing with a doctor."
    return "Higher likelihood", ROSE, "Your answers closely match the PCOS pattern. A check-up with a gynaecologist is a good next step."


def ring_html(p: float, color: str) -> str:
    deg = p * 360
    return (f'<div class="ring" style="background: conic-gradient({color} 0deg {deg:.0f}deg, #F7E6EC {deg:.0f}deg 360deg);">'
            f'<div class="inner"><div class="pct">{p*100:.0f}%</div><div class="lbl">match with<br>PCOS pattern</div></div></div>')


def bmi_label(b: float) -> str:
    return "below the usual range" if b < 18.5 else "within the usual range" if b < 23 else "slightly above the usual range" if b < 27.5 else "above the usual range"


def friendly(col: str) -> str:
    return FRIENDLY.get(col, col.replace("_", " ").title())


def metric_cards(m: dict):
    cols = st.columns(len(METRIC_NAMES))
    for c, k in zip(cols, METRIC_NAMES):
        v = m[k]
        c.markdown(f'<div class="metric"><div class="v">{"–" if pd.isna(v) else f"{v:.3f}"}</div><div class="k">{k}</div></div>',
                   unsafe_allow_html=True)


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cmap = LinearSegmentedColormap.from_list("rose", ["#FFFFFF", "#F4A7C0", "#B8407A"])
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * .55 else PLUM)
    ax.set_xticks([0, 1], ["No PCOS", "PCOS"]); ax.set_yticks([0, 1], ["No PCOS", "PCOS"])
    ax.set_xlabel("Predicted", color=PLUM); ax.set_ylabel("Actual", color=PLUM)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, colors=PLUM)
    fig.patch.set_alpha(0); fig.tight_layout()
    return fig


def plot_roc(y_true, proba, name):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.plot(fpr, tpr, color=ROSE, lw=2.5, label=name); ax.fill_between(fpr, tpr, alpha=.12, color=ROSE)
    ax.plot([0, 1], [0, 1], "--", color="#C9B4C0", lw=1)
    ax.set_xlabel("False positive rate", color=PLUM); ax.set_ylabel("True positive rate", color=PLUM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=PLUM); ax.grid(alpha=.2)
    fig.patch.set_alpha(0); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <div class="logo">🌸</div>
  <h1>Bloom</h1>
  <p>A calm, private space to check in on your cycle and hormone health, and understand what your body may be telling you.</p>
  <span class="pill">🔒 Nothing you enter is saved</span><span class="pill">💗 Takes about 2 minutes</span>
</div>
""", unsafe_allow_html=True)

tab_me, tab_eval, tab_about = st.tabs(["💗  My check-in", "📊  Model evaluation", "ℹ️  About"])

# =========================================================================== TAB 1 · CHECK-IN
with tab_me:
    st.markdown('<div class="soft">🌷 <b>Please note:</b> this is an awareness tool, not a diagnosis. Only a doctor can diagnose PCOS, '
                'usually with blood tests and an ultrasound.</div><br>', unsafe_allow_html=True)

    with st.form("checkin"):
        st.markdown('<div class="card"><h3>About you</h3><div class="sub">Just the basics.</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Your age", 15, 60, 28, help="The model learned from women aged 20–48.")
        height = c2.number_input("Height (cm)", 120, 210, 160)
        weight = c3.number_input("Weight (kg)", 30.0, 180.0, 60.0, step=0.5)

        st.markdown('<div class="card"><h3>Your cycle</h3><div class="sub">Cycle pattern is one of the strongest signals.</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        cycle = c1.radio("How would you describe your periods?", ["Regular – roughly every month", "Irregular – skipped, late or unpredictable"],
                         index=0)
        period_days = c2.slider("How many days does your period usually last?", 0, 12, 5)

        st.markdown('<div class="card"><h3>What you\'ve noticed</h3><div class="sub">Switch on anything that sounds like you, lately.</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        weight_gain = c1.toggle("Weight gain that's hard to explain")
        hair_growth = c1.toggle("Extra hair on face, chest or belly")
        skin_dark = c1.toggle("Dark, velvety patches (neck, underarms)")
        hair_loss = c2.toggle("Thinning hair on the scalp")
        pimples = c2.toggle("Frequent acne or pimples")
        fast_food = c2.toggle("Fast food most weeks")
        exercise = c2.toggle("Regular exercise", value=False)

        with st.expander("🔬  Have recent lab or scan results? Add them for a sharper estimate (optional)"):
            st.caption("Leave anything you don't know blank. We'll use typical values instead.")
            st.markdown("**Body measurements**")
            c1, c2 = st.columns(2)
            waist = c1.number_input("Waist (inches)", 20, 60, value=None, placeholder="e.g. 32")
            hip = c2.number_input("Hip (inches)", 20, 65, value=None, placeholder="e.g. 38")
            st.markdown("**Blood tests**")
            c1, c2, c3, c4 = st.columns(4)
            fsh = c1.number_input("FSH (mIU/mL)", 0.0, 200.0, value=None, placeholder="e.g. 5.0")
            lh = c2.number_input("LH (mIU/mL)", 0.0, 200.0, value=None, placeholder="e.g. 3.0")
            amh = c3.number_input("AMH (ng/mL)", 0.0, 100.0, value=None, placeholder="e.g. 4.0")
            tsh = c4.number_input("TSH (mIU/L)", 0.0, 100.0, value=None, placeholder="e.g. 2.5")
            c1, c2, c3, c4 = st.columns(4)
            prl = c1.number_input("Prolactin (ng/mL)", 0.0, 300.0, value=None, placeholder="e.g. 22")
            vitd = c2.number_input("Vitamin D3 (ng/mL)", 0.0, 300.0, value=None, placeholder="e.g. 26")
            prg = c3.number_input("Progesterone (ng/mL)", 0.0, 100.0, value=None, placeholder="e.g. 0.3")
            rbs = c4.number_input("Blood sugar (mg/dL)", 40.0, 500.0, value=None, placeholder="e.g. 100")
            st.markdown("**Pelvic ultrasound**")
            c1, c2, c3 = st.columns(3)
            fol_l = c1.number_input("Follicles – left ovary", 0, 40, value=None, placeholder="count")
            fol_r = c2.number_input("Follicles – right ovary", 0, 40, value=None, placeholder="count")
            endo = c3.number_input("Endometrium (mm)", 0.0, 30.0, value=None, placeholder="e.g. 8.5")
            c1, c2 = st.columns(2)
            fsz_l = c1.number_input("Avg follicle size – left (mm)", 0.0, 40.0, value=None, placeholder="e.g. 15")
            fsz_r = c2.number_input("Avg follicle size – right (mm)", 0.0, 40.0, value=None, placeholder="e.g. 16")

        go = st.form_submit_button("✨  Show my results")

    if go:
        core = {
            "age_yrs": float(age), "height_cm": float(height), "weight_kg": float(weight),
            "cycle_ri": 2.0 if cycle.startswith("Regular") else 4.0, "cycle_length_days": float(period_days),
            "weight_gain": float(weight_gain), "hair_growth": float(hair_growth), "skin_darkening": float(skin_dark),
            "hair_loss": float(hair_loss), "pimples": float(pimples), "fast_food": float(fast_food),
            "reg_exercise": float(exercise),
        }
        optional = {"waist_inch": waist, "hip_inch": hip, "fsh": fsh, "lh": lh, "amh": amh, "tsh": tsh, "prl": prl,
                    "vit_d3": vitd, "prg": prg, "rbs": rbs, "follicle_l": fol_l, "follicle_r": fol_r, "endometrium": endo,
                    "avg_f_size_l": fsz_l, "avg_f_size_r": fsz_r}
        res = assess(MODELS, DEFAULTS, FEATURES, core, optional)
        p = res["probs"][HEADLINE_MODEL]
        band, color, message = band_for(p)

        st.markdown("---")
        st.markdown(f'<div class="card"><div class="ring-wrap">{ring_html(p, color)}'
                    f'<div class="band" style="color:{color}">{band}</div>'
                    f'<div class="msg">{message}</div></div></div>', unsafe_allow_html=True)

        lvl_color = {"Basic": GOLD, "Good": SAGE, "Detailed": SAGE}[res["level"]]
        st.markdown(f'<div class="soft">📋 <b>How much this estimate knows:</b> '
                    f'<span style="color:{lvl_color};font-weight:800">{res["level"]}</span> — you gave {res["n_provided"]} of '
                    f'{res["n_total"]} details. Lab and scan results make the estimate noticeably more reliable.</div><br>',
                    unsafe_allow_html=True)

        # ---- what shaped it
        contrib = res["contrib"]
        ups = contrib[contrib > 0.05].sort_values(ascending=False).head(5)
        downs = contrib[contrib < -0.05].head(4)
        st.markdown('<div class="card"><h3>What shaped your result</h3>'
                    '<div class="sub">Compared with a typical woman in the data, these answers nudged the estimate.</div>', unsafe_allow_html=True)
        mx = max(ups.abs().max() if len(ups) else 0, downs.abs().max() if len(downs) else 0, 1e-6)
        rows = ""
        for name, v in ups.items():
            rows += (f'<div class="bar-row"><div class="name">▲ {html.escape(friendly(name))}</div><div class="track">'
                     f'<div class="fill" style="width:{abs(v)/mx*100:.0f}%;background:#E58AAB"></div></div></div>')
        for name, v in downs.items():
            rows += (f'<div class="bar-row"><div class="name">▼ {html.escape(friendly(name))}</div><div class="track">'
                     f'<div class="fill" style="width:{abs(v)/mx*100:.0f}%;background:#8CC5B4"></div></div></div>')
        if not rows:
            rows = '<div class="sub">Your answers are very close to the typical pattern, so no single factor stood out.</div>'
        st.markdown(rows + '<div class="sub" style="margin-top:.6rem">▲ raises the match with the PCOS pattern &nbsp; ▼ lowers it</div></div>',
                    unsafe_allow_html=True)

        # ---- body note + agreement
        agree = sum(v >= 0.5 for v in res["probs"].values())
        direction = "a PCOS-like pattern" if p >= 0.5 else "a non-PCOS pattern"
        same = agree if p >= 0.5 else len(res["probs"]) - agree
        st.markdown(f'<div class="card"><h3>A few more notes</h3>'
                    f'<div class="sub">🌿 Your BMI is about <b>{res["bmi"]:.1f}</b>, {bmi_label(res["bmi"])} (South-Asian guidelines).<br>'
                    f'🤝 <b>{same} of {len(res["probs"])}</b> independent models agree your answers fit {direction}.</div></div>',
                    unsafe_allow_html=True)
        with st.expander("See each model's estimate"):
            st.dataframe(pd.DataFrame({"Model": list(res["probs"]), "Match with PCOS pattern": [f"{v*100:.0f}%" for v in res["probs"].values()]}),
                         hide_index=True, use_container_width=True)

        if res["age"] < 20 or res["age"] > 48:
            st.markdown('<div class="note">The data this tool learned from covers women aged 20–48, so the estimate is less certain outside that range.</div><br>',
                        unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Gentle next steps</h3><div class="sub">'
                    '• Note when your last few periods started. A simple log helps any doctor.<br>'
                    '• If periods are irregular, or you have several signs above, book a visit with a gynaecologist or endocrinologist.<br>'
                    '• Common tests: hormone blood panel (LH, FSH, AMH, testosterone), thyroid, blood sugar, and a pelvic ultrasound.<br>'
                    '• Sleep, movement you enjoy, and balanced meals support hormone health whatever the result.</div></div>',
                    unsafe_allow_html=True)

# =========================================================================== TAB 2 · EVALUATION
with tab_eval:
    st.markdown('<div class="card"><h3>Test the models on your own file</h3>'
                '<div class="sub">Upload a CSV of <b>test data</b> (all feature columns plus the <code>pcos</code> target: 0 = no, 1 = yes). '
                'The saved models are loaded as-is. <b>Nothing is retrained.</b></div></div>', unsafe_allow_html=True)

    up = st.file_uploader("Upload test CSV", type=["csv"], label_visibility="collapsed")
    use_sample = st.checkbox("No file handy? Use the bundled sample (test_data.csv)", value=up is None)

    raw = None
    if up is not None:
        try:
            raw = pd.read_csv(up)
            source = up.name
        except Exception as exc:
            st.error(f"That file couldn't be read as a CSV: {exc}")
    elif use_sample and (BASE / "test_data.csv").exists():
        raw, source = pd.read_csv(BASE / "test_data.csv"), "test_data.csv (bundled sample)"

    if raw is None:
        st.info("Upload a CSV, or tick the box above to use the sample, to see results.")
    else:
        data, errors, notes = validate_upload(raw, FEATURES)
        if errors:
            for e in errors:
                st.error(e)
            with st.expander("Columns the app expects"):
                st.code(", ".join(FEATURES + ["pcos"]))
        else:
            for n in notes:
                st.markdown(f'<div class="note">{html.escape(n)}</div>', unsafe_allow_html=True)
            X, y = data[FEATURES], data["pcos"]
            share = y.mean()
            st.caption(f"📄 **{source}**  ·  {len(data)} patients  ·  {share:.0%} with PCOS")
            if y.nunique() < 2:
                st.warning("The target has only one class in this file, so AUC cannot be computed.")

            name = st.selectbox("Choose a model", list(MODELS), index=list(MODELS).index(HEADLINE_MODEL))
            metrics, pred, proba = score_model(MODELS[name], X, y)

            st.markdown(f"### {name}")
            metric_cards(metrics)
            st.write("")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Confusion matrix**")
                st.pyplot(plot_confusion(y, pred), use_container_width=True)
            with c2:
                if y.nunique() == 2:
                    st.markdown("**ROC curve**")
                    st.pyplot(plot_roc(y, proba, name), use_container_width=True)

            st.markdown("**Classification report**")
            rep = pd.DataFrame(classification_report(y, pred, labels=[0, 1], target_names=["No PCOS", "PCOS"],
                                                     output_dict=True, zero_division=0)).T
            rep["support"] = rep["support"].astype(int)
            st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}), use_container_width=True)

            st.markdown("### All five models, side by side")
            table = compare_models(MODELS, X, y)
            st.dataframe(table.style.format("{:.3f}").highlight_max(color="#F9D3DF"), use_container_width=True)
            best = table.mean(axis=1).idxmax()
            st.caption(f"🏆 Best average across the six metrics on this file: **{best}**. The best score in each column is highlighted.")

            out = data.copy()
            out["predicted_pcos"], out["probability_pcos"] = pred, np.round(proba, 4)
            st.download_button("⬇️  Download predictions (CSV)", out.to_csv(index=False).encode(),
                               file_name=f"predictions_{name.lower().replace(' ', '_')}.csv", mime="text/csv")

# =========================================================================== TAB 3 · ABOUT
with tab_about:
    st.markdown("""
<div class="card"><h3>What is PCOS?</h3>
<div class="sub">Polycystic ovary syndrome is a common hormonal condition. It can bring irregular periods, acne, extra hair growth,
weight changes and difficulty conceiving, and it is very treatable once recognised. Many women go years without a diagnosis.</div></div>

<div class="card"><h3>How this tool works</h3>
<div class="sub">Five machine-learning models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) were trained on 432 patient records
and checked on 109 the models had never seen. The estimate on the check-in tab comes from <b>Logistic Regression</b>, which was the
most reliable on that unseen group (about 92% accuracy). The other four are shown for comparison.<br><br>
<b>Dataset:</b> Kaggle “Polycystic Ovary Syndrome (PCOS)” clinical data: 541 patients, 41 features.</div></div>

<div class="card"><h3>Please keep in mind</h3>
<div class="sub">
• It is <b>not a diagnosis</b> and never replaces a doctor.<br>
• The data comes from one clinic, so results may not fit every population.<br>
• Without lab or scan results the tool fills the gaps with typical values, which makes it less precise.<br>
• Ultrasound follicle counts are part of how doctors diagnose PCOS, so those entries carry a lot of weight.<br>
• Nothing you type is stored or sent anywhere.</div></div>
""", unsafe_allow_html=True)

st.markdown('<div class="foot">Made with care 🌸 · For education and awareness only</div>', unsafe_allow_html=True)
