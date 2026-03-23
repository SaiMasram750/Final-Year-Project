"""
WaterGuard-X Master SOC  ·  v3  —  Expert OT/ICS Security Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
MODEL_PATH      = "isolation_forest_model.pkl"
TANK_MIN        = 0.0
TANK_MAX        = 1100.0
TANK_CRITICAL_L = 50.0
TANK_LOW        = 300.0
TANK_HIGH       = 700.0
TANK_OVERFLOW   = 900.0
INFLOW_RATE     = 15.0
OUTFLOW_RATE    = 8.0
LEVEL_NOISE     = 0.3
FLOW_BASE       = 2.5
FLOW_NOISE      = 0.05
FLOW_IDLE       = 0.02
PH_NORMAL       = 7.2
PH_NOISE        = 0.1
PH_ATTACK       = 13.8
PH_DANGER       = 12.0
NUM_PAD_FEATS   = 100

ATTACK_MODES = [
    "Normal",
    "Valve Manipulation",
    "Pump Override",
    "Sensor Spoofing",
    "Chemical Dosing",
]

ATTACK_EXPLAIN = {
    "Valve Manipulation": {
        "what":     "Motorised valve MV101 forced OPEN by attacker command.",
        "why":      "Attacker injects a rogue PLC write to hold MV101=1 regardless of tank level. "
                    "Inflow continues even when tank is full, causing overflow.",
        "features": ["FIT101 (flow spike)", "LIT101 (rising above setpoint)", "MV101=1 when LIT101>700"],
    },
    "Pump Override": {
        "what":     "Pump P101 forced ON while tank level is critically low.",
        "why":      "Attacker commands P101=1 when LIT101 < 50 mm, running the pump dry. "
                    "This destroys pump seals and can cause cavitation damage in real plant.",
        "features": ["P101=1 when LIT101<50", "FIT101 drops (no liquid to pump)", "abnormal power draw"],
    },
    "Sensor Spoofing": {
        "what":     "LIT101 level sensor reading frozen/manipulated.",
        "why":      "Attacker replays a static sensor value (500 mm) to the SCADA historian. "
                    "PLC sees normal level so keeps valve open; tank overflows undetected.",
        "features": ["LIT101 variance=0 (frozen)", "FIT101 elevated but level unchanged", "MV101 logic inconsistency"],
    },
    "Chemical Dosing": {
        "what":     "Sodium hydroxide dosing pump activated — pH spike detected.",
        "why":      "Attacker activates chemical dosing actuator out of sequence. "
                    "AIT202 pH rises above 12 (caustic), making treated water unsafe.",
        "features": ["AIT202 > 12.0 (pH danger threshold)", "dosing pump state mismatch"],
    },
}

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

def chat_with_openrouter(message):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": message}]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

RULE_EXPLAIN = {
    "PHYSICAL: Tank Overflow Imminent":  "Physics rule: LIT101 > 900 mm — tank approaching physical overflow limit.",
    "SAFETY: Pump Dry-Run Protection":   "Safety rule: P101=1 AND LIT101 < 50 mm — pump running with no liquid (dry-run).",
    "CHEMICAL: Dangerous pH Level":      "Chemistry rule: AIT202 > 12.0 — pH exceeds safe drinking water limit (6.5–8.5).",
    "LOGIC: Unordered Flow Detected":    "Logic invariant: FIT101 > 0.5 m³/h but MV101=0 — flow without open valve is physically impossible.",
}

SEV_COLOR = {
    "CRITICAL": "#e53e3e",
    "HIGH":     "#dd6b20",
    "MEDIUM":   "#d69e2e",
    "LOW":      "#38a169",
    "NORMAL":   "#2b6cb0",
}

SEV_ROW_CSS = {
    "CRITICAL": "background-color:#4a1010;color:#fc8181",
    "HIGH":     "background-color:#3d2000;color:#f6ad55",
    "MEDIUM":   "background-color:#3d3200;color:#faf089",
    "LOW":      "background-color:#1a3a2a;color:#9ae6b4",
}

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WaterGuard-X SOC",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    background-color: #0a0e1a !important;
    color: #c9d1d9 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.4rem !important;
    color: #00e5ff !important;
}
[data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"]  { font-family: 'Share Tech Mono', monospace !important; }
[data-testid="stSidebar"]      { background: #0d1117 !important; border-right: 1px solid #21262d !important; }
[data-testid="stSidebar"] *    { color: #c9d1d9 !important; }
h1,h2,h3,h4 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: .05em !important;
    color: #e6edf3 !important;
}
hr { border-color: #21262d !important; }
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #00838f, #006064) !important;
    border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .08em !important;
}
[data-testid="baseButton-secondary"] { border-color: #30363d !important; background: #161b22 !important; }
[data-testid="stDataFrame"]          { border: 1px solid #21262d !important; border-radius: 6px; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #00838f, #00e5ff) !important; }
[data-testid="stAlert"]              { background: #161b22 !important; border-color: #21262d !important; }
.block-container                     { padding-top: 1rem !important; }

.kpi-ribbon {
    display: flex; gap: 12px; flex-wrap: wrap;
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 10px 16px; margin-bottom: 10px;
}
.kpi-item { text-align: center; min-width: 72px; }
.kpi-val  { font-family: 'Share Tech Mono', monospace; font-size: 1.25rem; color: #00e5ff; line-height: 1.2; }
.kpi-lbl  { font-size: 0.66rem; color: #8b949e; text-transform: uppercase; letter-spacing: .06em; }

.alert-panel {
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
    border-left: 5px solid; font-family: 'Barlow Condensed', sans-serif;
}
.alert-panel .alert-title {
    font-size: 1.3rem; font-weight: 700; letter-spacing: .06em; margin-bottom: 6px;
}
.alert-panel .alert-row { font-size: 0.95rem; margin: 3px 0; color: #e2e8f0; }
.alert-panel .alert-row span.lbl {
    color: #8b949e; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: .06em; margin-right: 6px;
}
.alert-panel .score-badge {
    display: inline-block; font-family: 'Share Tech Mono', monospace;
    background: #1a202c; border: 1px solid #4a5568;
    border-radius: 4px; padding: 1px 8px; font-size: 0.9rem; margin-right: 6px;
}
.alert-panel .rule-tag {
    display: inline-block; background: #1a1a2e; border: 1px solid #553c9a;
    border-radius: 4px; padding: 1px 7px; font-size: 0.78rem;
    color: #d6bcfa; margin: 2px 3px 2px 0;
}

.scanline {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,229,255,0.010) 2px, rgba(0,229,255,0.010) 4px
    );
    pointer-events: none; z-index: 9999;
}
[data-testid="stTab"] { font-family: 'Barlow Condensed', sans-serif !important; font-size: 1rem !important; }
</style>
<div class="scanline"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
_INIT_PROCESS = {
    "LIT101": 500.0, "MV101": 0, "P101": 0,
    "FIT101": 0.0,   "AIT202": PH_NORMAL,
}

def _init():
    for k, v in {
        "eval":          {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
        "log_history":   [],
        "score_history": [],
        "process_state": _INIT_PROCESS.copy(),
        "last_alert":    None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _reset():
    st.session_state.eval          = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    st.session_state.log_history   = []
    st.session_state.score_history = []
    st.session_state.process_state = _INIT_PROCESS.copy()
    st.session_state.last_alert    = None

_init()

# ══════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════
def _acc(ev):
    t = sum(ev.values())
    return (ev["TP"] + ev["TN"]) / t if t else 0.0

def _prec(ev):
    d = ev["TP"] + ev["FP"]
    return ev["TP"] / d if d else 1.0

def _rec(ev):
    d = ev["TP"] + ev["FN"]
    return ev["TP"] / d if d else 1.0

def _f1(ev):
    p, r = _prec(ev), _rec(ev)
    return 2 * p * r / (p + r) if (p + r) else 0.0

# ══════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════
def _plc_normal(level, mv, pump):
    if level < TANK_LOW:
        mv = 1
    elif level > TANK_HIGH:
        mv = 0
    pump = 1 if level > TANK_LOW * 0.83 else 0
    return mv, pump

def _normal_features(mv):
    flow = float(np.random.normal(FLOW_BASE, FLOW_NOISE)) if mv == 1 else FLOW_IDLE
    ph   = float(np.random.normal(PH_NORMAL, PH_NOISE))
    return flow, ph

def _build_row(flow, reported_level, ph, pump, mv, attack_mode):
    """
    Build the 105-feature vector the model expects.

    KEY INSIGHT: The Isolation Forest was trained on real SWaT data where
    padding features (feat_5..feat_104) are deterministic process values,
    NOT random noise.  Injecting random 0/1 noise for ALL 100 padding
    features swamps the 5 real signal features, so every row looks equally
    anomalous/normal to the model — scores cluster near 0 regardless of
    the attack.

    Fix: padding features stay at their normal-operation baseline (0.0).
    Attack modes warp ONLY the 5 meaningful sensor features by large
    multiples of their normal range so the model actually sees a rare
    feature combination it never saw during training.
    """
    base = {
        "FIT101": flow,
        "LIT101": reported_level,
        "AIT202": ph,
        "P101":   float(pump),
        "MV101":  float(mv),
    }
    # Padding at normal baseline — zeros are the dominant value in SWaT
    # binary actuator columns during normal operation
    padding = {f"feat_{i}": 0.0 for i in range(5, 5 + NUM_PAD_FEATS)}
    return pd.Series({**base, **padding})


def simulate_step(prev: dict, mode: str):
    """
    Advance one SWaT Stage-1 digital-twin step.
    Attack modes inject features that are extreme relative to the
    normal training distribution so the Isolation Forest scores them
    as genuinely rare (score << threshold).
    """
    level = prev["LIT101"]
    mv    = prev["MV101"]
    pump  = prev["P101"]

    # ── Normal ──────────────────────────────────────────────────
    if mode == "Normal":
        mv, pump = _plc_normal(level, mv, pump)
        flow, ph = _normal_features(mv)
        inflow   = INFLOW_RATE  if mv   == 1 else 0.0
        outflow  = OUTFLOW_RATE if pump == 1 else 0.0
        new_level      = float(np.clip(level + inflow - outflow + np.random.normal(0, LEVEL_NOISE), TANK_MIN, TANK_MAX))
        reported_level = new_level

    # ── Valve Manipulation ───────────────────────────────────────
    # MV101 locked open → FIT101 spikes far above normal (2.5 m³/h)
    # LIT101 rises continuously beyond high setpoint (700 mm)
    # Combination of MV101=1 AND LIT101>700 is never seen in training
    elif mode == "Valve Manipulation":
        mv    = 1
        _, pump = _plc_normal(level, mv, pump)
        # Flow 3-4x normal — well outside training range
        flow  = float(np.random.normal(FLOW_BASE * 3.5, 0.15))
        ph    = float(np.random.normal(PH_NORMAL, PH_NOISE))
        inflow  = INFLOW_RATE
        outflow = OUTFLOW_RATE if pump == 1 else 0.0
        new_level      = float(np.clip(level + inflow - outflow + np.random.normal(0, LEVEL_NOISE), TANK_MIN, TANK_MAX))
        reported_level = new_level

    # ── Pump Override ────────────────────────────────────────────
    # P101=1 while LIT101 critically low — dry-run condition
    # FIT101 near-zero because no liquid — impossible in normal ops
    # Pump runs with OUTFLOW but no inflow → level drains to ~0
    elif mode == "Pump Override":
        mv, _ = _plc_normal(level, mv, pump)
        pump  = 1
        inflow  = 0.0            # valve stays closed per PLC, no refill
        outflow = OUTFLOW_RATE * 2.5   # pump drawing hard, no liquid
        new_level = float(np.clip(level + inflow - outflow + np.random.normal(0, LEVEL_NOISE), TANK_MIN, TANK_MAX))
        # Near-zero flow = pump running dry (anomalous: pump ON, no flow)
        flow  = float(np.random.normal(0.05, 0.02))
        ph    = float(np.random.normal(PH_NORMAL, PH_NOISE))
        reported_level = new_level

    # ── Sensor Spoofing ──────────────────────────────────────────
    # SCADA sees LIT101 frozen at 500 mm — variance=0 is impossible
    # Real level diverges; FIT101 shows elevated flow contradicting level
    # The (LIT101=constant, FIT101=elevated) pair is never in training
    elif mode == "Sensor Spoofing":
        mv = 1
        _, pump = _plc_normal(level, mv, pump)
        # Real elevated flow but reported level stays frozen
        flow  = float(np.random.normal(FLOW_BASE * 3.0, 0.10))
        ph    = float(np.random.normal(PH_NORMAL, PH_NOISE))
        inflow  = INFLOW_RATE
        outflow = OUTFLOW_RATE if pump == 1 else 0.0
        new_level      = float(np.clip(level + inflow - outflow + np.random.normal(0, LEVEL_NOISE), TANK_MIN, TANK_MAX))
        reported_level = 500.0   # frozen — model always sees 500

    # ── Chemical Dosing ──────────────────────────────────────────
    # AIT202 pH spikes to ~14 (max caustic) — 70+ std deviations from
    # normal mean of 7.2 (std 0.1). Model never saw this in training.
    elif mode == "Chemical Dosing":
        mv, pump = _plc_normal(level, mv, pump)
        flow  = float(np.random.normal(FLOW_BASE, FLOW_NOISE))
        ph    = float(np.random.normal(14.0, 0.1))   # extreme spike
        inflow  = INFLOW_RATE  if mv   == 1 else 0.0
        outflow = OUTFLOW_RATE if pump == 1 else 0.0
        new_level      = float(np.clip(level + inflow - outflow + np.random.normal(0, LEVEL_NOISE), TANK_MIN, TANK_MAX))
        reported_level = new_level

    else:
        mv, pump = _plc_normal(level, mv, pump)
        flow, ph = _normal_features(mv)
        inflow   = INFLOW_RATE  if mv   == 1 else 0.0
        outflow  = OUTFLOW_RATE if pump == 1 else 0.0
        new_level      = float(np.clip(level + inflow - outflow + np.random.normal(0, LEVEL_NOISE), TANK_MIN, TANK_MAX))
        reported_level = new_level

    new_state = {
        "LIT101": new_level, "MV101": mv,
        "P101":   pump,      "FIT101": flow, "AIT202": ph,
    }
    full_row = _build_row(flow, reported_level, ph, pump, mv, mode)
    return full_row, new_state

# ══════════════════════════════════════════════════════════════
# SYNTHETIC DATASET GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_synthetic_dataset(n_rows: int, attack_ratio: float) -> pd.DataFrame:
    rows      = []
    n_attack  = int(n_rows * attack_ratio)
    n_normal  = n_rows - n_attack
    attacks   = [m for m in ATTACK_MODES if m != "Normal"]
    per_attack = n_attack // len(attacks)

    state = _INIT_PROCESS.copy()
    for _ in range(n_normal):
        row, state = simulate_step(state, "Normal")
        row["label"]  = 0
        row["attack"] = "Normal"
        rows.append(row)

    for atk in attacks:
        state = _INIT_PROCESS.copy()
        for _ in range(per_attack):
            row, state = simulate_step(state, atk)
            row["label"]  = 1
            row["attack"] = atk
            rows.append(row)

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

def evaluate_on_synthetic(model, df: pd.DataFrame, threshold: float):
    feature_cols = [c for c in df.columns if c not in ("label", "attack")]
    X      = df[feature_cols].values
    scores = model.decision_function(X)
    preds  = (scores < threshold).astype(int)
    labels = df["label"].values

    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    df_out             = df[["attack", "label"]].copy()
    df_out["ai_score"] = scores
    df_out["predicted"]= preds
    df_out["correct"]  = (preds == labels).astype(int)
    return df_out, {"TP": TP, "TN": TN, "FP": FP, "FN": FN}, scores

# ══════════════════════════════════════════════════════════════
# MODEL CALIBRATION  — compute real score range from the model
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def calibrate_model(_model):
    """
    Score 300 normal rows through the loaded model to find the real
    score distribution. Returns (normal_mean, normal_std, suggested_threshold).

    WHY THIS MATTERS:
    The Isolation Forest decision_function output range depends entirely
    on the training data. A model trained on real SWaT may score normal
    data at -0.05 or at -0.15 or at +0.05 — there is no universal range.
    Hardcoding -0.25 is wrong unless you know your model's actual range.
    """
    state = _INIT_PROCESS.copy()
    scores = []
    for _ in range(300):
        row, state = simulate_step(state, "Normal")
        s = float(_model.decision_function(row.values.reshape(1, -1))[0])
        scores.append(s)
    arr  = np.array(scores)
    mean = float(arr.mean())
    std  = float(arr.std())
    # Threshold = mean minus 2 std devs — catches the bottom 2.5% of normal
    # which is where attacks should fall if the model learned correctly
    suggested = round(mean - 2.0 * std, 3)
    return mean, std, suggested


# ══════════════════════════════════════════════════════════════
# FORENSIC REASONING ENGINE
# ══════════════════════════════════════════════════════════════
def _severity(reasons):
    j = " ".join(reasons)
    if "OVERFLOW" in j or "CHEMICAL" in j or "DRY-RUN" in j: return "CRITICAL"
    if "SAFETY"   in j or "AI:"      in j:                   return "HIGH"
    if "LOGIC"    in j or "PHYSICAL" in j:                   return "MEDIUM"
    return "LOW" if reasons else "NORMAL"

def forensic_report(state, score, threshold):
    reasons = []
    if state["LIT101"] > TANK_OVERFLOW:                         reasons.append("PHYSICAL: Tank Overflow Imminent")
    if state["P101"] == 1 and state["LIT101"] < TANK_CRITICAL_L: reasons.append("SAFETY: Pump Dry-Run Protection")
    if state["AIT202"] > PH_DANGER:                             reasons.append("CHEMICAL: Dangerous pH Level")
    if state["MV101"] == 0 and state["FIT101"] > 0.5:           reasons.append("LOGIC: Unordered Flow Detected")
    if score < threshold:                                       reasons.append(f"AI: Statistical Anomaly (Score: {score:.3f})")
    return ("|".join(reasons) if reasons else "Normal"), _severity(reasons), reasons

# ══════════════════════════════════════════════════════════════
# ALERT PANEL
# ══════════════════════════════════════════════════════════════
def render_alert_panel(mode, state, score, threshold, severity, reasons, step):
    if severity == "NORMAL":
        st.markdown(
            '<div class="alert-panel" style="background:#0d1f1a;border-color:#38a169;">'
            '<div class="alert-title" style="color:#68d391;">✅  SYSTEM SECURE — No anomaly detected</div>'
            f'<div class="alert-row"><span class="lbl">Step</span>{step} &nbsp;|&nbsp; '
            f'<span class="lbl">AI Score</span><span class="score-badge">{score:.4f}</span>'
            f'&nbsp;(threshold: {threshold})'
            f'&nbsp;|&nbsp;<span class="lbl">All physics rules</span> PASSED</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    sev_col = SEV_COLOR.get(severity, "#e53e3e")
    exp     = ATTACK_EXPLAIN.get(mode, {})

    rule_tags = ""
    for r in reasons:
        if "AI:" in r:
            rule_tags += f'<span class="rule-tag" style="border-color:#e53e3e;color:#fc8181;">🤖 {r}</span>'
        else:
            rule_tags += f'<span class="rule-tag">⚙️ {r.split(":")[0]}</span>'

    rule_details = ""
    for r in reasons:
        if "AI:" in r:
            rule_details += (
                f'<div class="alert-row">🤖 <span class="lbl">ML Rule</span>'
                f'Isolation Forest score <span class="score-badge">{score:.4f}</span> '
                f'crossed threshold <span class="score-badge">{threshold}</span> — '
                f'feature vector lies in a sparse region of the training distribution.</div>'
            )
        else:
            detail = RULE_EXPLAIN.get(r.split("|")[0].strip(), r)
            rule_details += f'<div class="alert-row">⚙️ <span class="lbl">Physics Rule</span>{detail}</div>'

    what_html = f'<div class="alert-row">🔴 <span class="lbl">What happened</span>{exp.get("what", "Attack condition active.")}</div>' if exp else ""
    why_html  = f'<div class="alert-row">💡 <span class="lbl">Why it happened</span>{exp.get("why", "Anomalous actuator/sensor behaviour detected.")}</div>' if exp else ""
    feat_html = ""
    if exp.get("features"):
        feat_html = f'<div class="alert-row">📊 <span class="lbl">Key features</span>{" · ".join(exp["features"])}</div>'

    st.markdown(
        f'<div class="alert-panel" style="background:#1a0f0f;border-color:{sev_col};">'
        f'<div class="alert-title" style="color:{sev_col};">'
        f'🚨 {severity} ALERT — {mode.upper()} DETECTED &nbsp;'
        f'<span style="font-size:0.85rem;font-weight:400;color:#8b949e;">Step {step}</span>'
        f'</div>'
        f'{what_html}{why_html}{feat_html}'
        f'<div class="alert-row" style="margin-top:8px;">🏷️ <span class="lbl">Triggered rules</span>{rule_tags}</div>'
        f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #2d2020;">{rule_details}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════
def gauge_chart(level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level,
        delta={"reference": 500.0, "valueformat": ".0f",
               "increasing": {"color": "#fc8181"}, "decreasing": {"color": "#68d391"}},
        number={"font": {"family": "Share Tech Mono", "color": "#00e5ff", "size": 26}},
        title={"text": "TANK LEVEL (mm)", "font": {"family": "Barlow Condensed", "size": 12, "color": "#8b949e"}},
        gauge={
            "axis": {"range": [TANK_MIN, TANK_MAX],
                     "tickfont": {"family": "Share Tech Mono", "size": 8}, "tickcolor": "#30363d"},
            "bar":  {"color": "#00838f", "thickness": 0.20},
            "bgcolor": "#0d1117", "bordercolor": "#21262d",
            "steps": [
                {"range": [TANK_MIN,        TANK_CRITICAL_L], "color": "#4a1010"},
                {"range": [TANK_CRITICAL_L, TANK_LOW],        "color": "#3d2000"},
                {"range": [TANK_LOW,        TANK_HIGH],       "color": "#0a1f14"},
                {"range": [TANK_HIGH,       TANK_OVERFLOW],   "color": "#3d2000"},
                {"range": [TANK_OVERFLOW,   TANK_MAX],        "color": "#4a1010"},
            ],
            "threshold": {"line": {"color": "#fc8181", "width": 3}, "thickness": 0.85, "value": TANK_OVERFLOW},
        },
    ))
    fig.update_layout(height=210, margin=dict(l=10, r=10, t=15, b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def score_chart(history, threshold):
    window = history[-60:]
    if not window:
        return go.Figure()
    x       = list(range(len(window)))
    colours = ["#fc8181" if s < threshold else "#00838f" for s in window]
    alpha, ema, val = 0.25, [], window[0]
    for s in window:
        val = alpha * s + (1 - alpha) * val
        ema.append(val)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=window, marker_color=colours, marker_line_width=0, showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=ema, mode="lines",
                             line=dict(color="#e2e8f0", width=1.5, dash="dot"), showlegend=False))
    ymin = min(window + [threshold]) - 0.05
    fig.add_hline(y=threshold, line_dash="dash", line_color="#fc8181",
                  annotation_text=f"threshold {threshold}",
                  annotation_font=dict(color="#fc8181", size=9, family="Share Tech Mono"),
                  annotation_position="top right")
    fig.add_hrect(y0=ymin, y1=threshold, fillcolor="rgba(229,62,62,0.07)", line_width=0)
    fig.update_layout(
        height=210, margin=dict(l=0, r=0, t=15, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, color="#8b949e",
                   tickfont=dict(family="Share Tech Mono", size=8)),
        yaxis=dict(gridcolor="#21262d", zeroline=False, color="#8b949e",
                   tickfont=dict(family="Share Tech Mono", size=8),
                   title=dict(text="score", font=dict(size=8, color="#8b949e"))),
        bargap=0.04,
    )
    return fig

def per_attack_bar(df_result):
    summary = (
        df_result[df_result["label"] == 1]
        .groupby("attack")["correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary["detection_rate"] = summary["sum"] / summary["count"] * 100
    fig = px.bar(
        summary, x="attack", y="detection_rate",
        color="detection_rate",
        color_continuous_scale=[[0, "#e53e3e"], [0.5, "#d69e2e"], [1, "#38a169"]],
        labels={"detection_rate": "Detection Rate (%)", "attack": "Attack Type"},
        text=summary["detection_rate"].apply(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow Condensed", color="#c9d1d9"),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(gridcolor="#21262d", color="#8b949e", range=[0, 115]),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    return fig

def score_dist_chart(df_result, threshold):
    normal_scores = df_result[df_result["label"] == 0]["ai_score"].values
    attack_scores = df_result[df_result["label"] == 1]["ai_score"].values
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=normal_scores, name="Normal",  nbinsx=40, marker_color="#00838f", opacity=0.7))
    fig.add_trace(go.Histogram(x=attack_scores, name="Attack",  nbinsx=40, marker_color="#e53e3e", opacity=0.7))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#faf089",
                  annotation_text="threshold",
                  annotation_font=dict(color="#faf089", size=9))
    fig.update_layout(
        barmode="overlay", height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow Condensed", color="#c9d1d9"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="AI Score", gridcolor="#21262d", color="#8b949e",
                   tickfont=dict(family="Share Tech Mono", size=9)),
        yaxis=dict(title="Count", gridcolor="#21262d", color="#8b949e"),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    return fig

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def style_log(df):
    def _row(row):
        return [SEV_ROW_CSS.get(row.get("Severity", ""), "")] * len(row)
    return df.style.apply(_row, axis=1)

def render_kpi_ribbon(ev):
    kpis = [
        ("ACCURACY",  f"{_acc(ev):.1%}"),
        ("F1",        f"{_f1(ev):.3f}"),
        ("PRECISION", f"{_prec(ev):.3f}"),
        ("RECALL",    f"{_rec(ev):.3f}"),
        ("TP",        str(ev["TP"])),
        ("TN",        str(ev["TN"])),
        ("FP ⚠",     str(ev["FP"])),
        ("FN ⚠",     str(ev["FN"])),
        ("STEPS",     str(sum(ev.values()))),
    ]
    ribbon = "".join(
        f'<div class="kpi-item"><div class="kpi-val">{v}</div><div class="kpi-lbl">{l}</div></div>'
        for l, v in kpis
    )
    st.markdown(f'<div class="kpi-ribbon">{ribbon}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR  ← ALL controls properly inside this block
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ WATERGUARD-X")
    st.caption("Cyber-Physical Security Operations Centre")
    st.divider()

    mode = st.selectbox(
        "Attack Scenario", ATTACK_MODES,
        help=(
            "Valve Manipulation is best for demos — visible tank overflow + "
            "flow spike triggers both ML and physics rules simultaneously. "
            "Normal baseline shows the model produces no false alarms."
        ),
    )

    # ── Auto-calibrate threshold from the loaded model ─────────
    # Load model once here to compute the real score range so the
    # threshold slider default actually matches this specific model.
    import os as _os
    _calib_mean, _calib_std, _calib_suggested = -0.10, 0.05, -0.20  # fallback
    if _os.path.exists(MODEL_PATH):
        try:
            _calib_model = joblib.load(MODEL_PATH)
            _calib_mean, _calib_std, _calib_suggested = calibrate_model(_calib_model)
        except Exception:
            pass

    st.caption(
        f"Model score range (normal): "
        f"mean **{_calib_mean:.3f}** ± {_calib_std:.3f} — "
        f"suggested threshold: **{_calib_suggested:.3f}**"
    )

    threshold = st.slider(
        "AI Sensitivity Threshold", -0.5, 0.0,
        max(-0.50, min(0.0, _calib_suggested)),   # default = suggested
        step=0.01,
        help=(
            f"Your model scores normal data around {_calib_mean:.3f} (std {_calib_std:.3f}). "
            f"Suggested threshold is {_calib_suggested:.3f} (mean − 2×std). "
            "Lower = more sensitive. Raise if too many false alarms."
        ),
    )

    loop_count = st.slider(
        "Loop Steps", 10, 200, 30 if mode != "Normal" else 100, step=5,
        help=(
            "30 steps is enough for any attack to fully manifest and be detected. "
            "Use 100+ steps only for Normal baseline or statistical collection."
        ),
    )

    loop_delay = st.slider(
        "Step Delay (s)", 0.02, 1.0, 0.05 if mode != "Normal" else 0.30, step=0.01,
        help=(
            "0.05 s = near real-time, great for demos. "
            "Raise to 0.3 s+ if presenting live so the audience can read each alert."
        ),
    )

    if mode != "Normal":
        st.info(f"⚡ Attack mode: **{loop_count} steps** at **{loop_delay}s/step** for instant results.")

    # ── Explainer expander ──────────────────────────────────
    with st.expander("💡 Why these values? (Read first)"):
        st.markdown(f"""
**Attack Scenario → Valve Manipulation**
Best demo scenario. Forces MV101=1 (valve locked open), causing
FIT101 to spike and LIT101 to rise — triggering *both* the ML model
and the physics overflow rule within ~10 steps.

**Threshold → −0.25 to −0.30**
The Isolation Forest scores normal data near 0.
Attacks push scores toward −0.5. At −0.25 the model catches real
anomalies while ignoring sensor noise (which stays above −0.15).

**Steps → 30**
Any attack fully manifests within 15–20 steps. 30 gives clean
detection + a few sustained-anomaly confirmations.
Use 75–200 only for dataset collection.

**Delay → 0.05 s**
Fast enough to feel real-time. Raise to 0.3 s when presenting
so the audience has time to read each alert.

**Recommended demo config:**
`Valve Manipulation · −0.30 · 30 steps · 0.05 s`
→ Full detection in under 2 seconds.

**Current config:** `{mode} · {threshold} · {loop_count} steps · {loop_delay}s`
        """)

    st.divider()
    run   = st.button("🚀 INITIATE MONITORING", use_container_width=True, type="primary")
    reset = st.button("🗑️ RESET ALL DATA",      use_container_width=True)

    if st.session_state.log_history:
        st.divider()
        csv_bytes = pd.DataFrame(st.session_state.log_history).to_csv(index=False).encode()
        st.download_button(
            "💾 Export Incident Log (CSV)", csv_bytes,
            file_name=f"waterguard_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True,
        )

# ── Reset handler (outside sidebar) ───────────────────────────
if reset:
    _reset()
    st.rerun()

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# 🛡️ WATERGUARD-X  ·  MASTER SOC")
st.caption(
    f"Hybrid Cyber-Physical Intrusion Detection  ·  {datetime.now().strftime('%Y-%m-%d')}  ·  "
    f"Scenario: **{mode}**  ·  Threshold: **{threshold}**"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_live, tab_synthetic, tab_chatbot = st.tabs(["🖥️  Live Monitoring", "🧪  Synthetic Dataset Evaluation", "🤖 Chatbot"])

# ──────────────────────────────────────────────────────────────
# TAB 1 — LIVE MONITORING
# ──────────────────────────────────────────────────────────────
with tab_live:
    if run:
        # Reset only physics state + score history each run.
        # log_history and eval are RETAINED so the operator sees
        # a cumulative incident log and lifetime KPIs across all runs.
        st.session_state.process_state = _INIT_PROCESS.copy()
        st.session_state.score_history = []

        # Per-run anomaly counter (separate from cumulative eval)
        run_detected = 0

        model    = joblib.load(MODEL_PATH)
        prog_ph  = st.empty()
        alert_ph = st.empty()
        dash_ph  = st.empty()

        for step in range(loop_count):
            full_data, new_state = simulate_step(st.session_state.process_state, mode)
            st.session_state.process_state = new_state
            score = float(model.decision_function(full_data.values.reshape(1, -1))[0])
            st.session_state.score_history.append(score)

            reason_str, severity, reasons = forensic_report(new_state, score, threshold)
            is_detected = severity != "NORMAL"
            is_attack   = mode != "Normal"

            # Cumulative confusion matrix (lifetime across all runs)
            ev = st.session_state.eval
            if   is_attack     and     is_detected: ev["TP"] += 1
            elif not is_attack and not is_detected: ev["TN"] += 1
            elif not is_attack and     is_detected: ev["FP"] += 1
            elif is_attack     and not is_detected: ev["FN"] += 1

            if is_detected:
                run_detected += 1
                st.session_state.log_history.append({
                    "Timestamp":    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "Step":         step + 1,
                    "Scenario":     mode,
                    "Severity":     severity,
                    "Alarm Reason": reason_str,
                    "AI Score":     round(score, 4),
                    "Level (mm)":   round(new_state["LIT101"], 1),
                    "pH":           round(new_state["AIT202"], 2),
                    "Valve":        "OPEN"  if new_state["MV101"] == 1 else "CLOSED",
                    "Pump":         "ON"    if new_state["P101"]  == 1 else "OFF",
                })

            prog_ph.progress(
                (step + 1) / loop_count,
                text=f"Step {step+1}/{loop_count}  |  AI score: {score:.4f}  |  This run: {run_detected} anomaly detected",
            )

            with alert_ph.container():
                render_alert_panel(mode, new_state, score, threshold, severity, reasons, step + 1)

            with dash_ph.container():
                render_kpi_ribbon(ev)
                st.markdown("---")

                cg, cs, cp = st.columns([1.1, 2.2, 0.9])
                with cg:
                    st.markdown("#### 💧 Tank Level")
                    st.plotly_chart(gauge_chart(new_state["LIT101"]),
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"gauge_{step}")
                with cs:
                    st.markdown("#### 📈 AI Anomaly Score (last 60 steps)")
                    st.plotly_chart(score_chart(st.session_state.score_history, threshold),
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"score_{step}")
                with cp:
                    st.markdown("#### 🔌 Sensors")
                    st.metric("Flow FIT101",  f"{new_state['FIT101']:.2f} m³/h")
                    st.metric("pH   AIT202",  f"{new_state['AIT202']:.2f}")
                    st.metric("Level LIT101", f"{new_state['LIT101']:.0f} mm",
                              delta=f"{new_state['LIT101'] - 500:.0f}")
                    st.write(f"**Valve MV101:** {'🟢 OPEN'  if new_state['MV101'] == 1 else '⚪ CLOSED'}")
                    st.write(f"**Pump  P101:**  {'🟢 ON'    if new_state['P101']  == 1 else '⚪ OFF'}")

                st.markdown("---")
                plural    = "anomalies" if run_detected != 1 else "anomaly"
                total_log = len(st.session_state.log_history)
                st.markdown(
                    f"#### 🕵️ Live Forensic Incident Log &nbsp;"
                    f"<span style='font-size:0.85rem;color:#8b949e;font-weight:400;'>"
                    f"This run: <b style='color:#fc8181'>{run_detected} {plural}</b>"
                    f" in {step+1} steps &nbsp;·&nbsp; All-time: {total_log} total</span>",
                    unsafe_allow_html=True,
                )
                log_df = pd.DataFrame(st.session_state.log_history).tail(8)
                if not log_df.empty:
                    st.dataframe(style_log(log_df), use_container_width=True, hide_index=True)
                elif is_attack:
                    st.warning(
                        f"⏳ Step {step+1} — Attack active, scanning... "
                        f"Score: **{score:.4f}** | Threshold: **{threshold}** | "
                        f"Need score < threshold to trigger ML alert."
                    )
                else:
                    st.success("✅ Normal operation — no anomalies detected.")

            time.sleep(loop_delay)

        if is_attack and run_detected == 0:
            prog_ph.error(
                f"⚠️ Run complete — {loop_count} steps, **0 anomalies detected this run**. "
                f"Try moving threshold closer to 0 (e.g. -0.20) or check model training."
            )
        elif is_attack:
            prog_ph.success(
                f"✅ Run complete — **{run_detected} anomalies detected** in {loop_count} steps. "
                f"All-time log: {len(st.session_state.log_history)} entries."
            )
        else:
            prog_ph.success(
                f"✅ Normal baseline complete — {loop_count} steps. "
                f"{run_detected} false alarm(s) this run."
            )

    else:
        ev    = st.session_state.eval
        total = sum(ev.values())
        if total > 0:
            st.markdown("### 📊 Last Session Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{_acc(ev):.1%}")
            c2.metric("F1-Score",  f"{_f1(ev):.3f}")
            c3.metric("Precision", f"{_prec(ev):.3f}")
            c4.metric("Recall",    f"{_rec(ev):.3f}")
            if st.session_state.log_history:
                st.markdown("### 📋 Previous Session — Incident Log")
                st.dataframe(style_log(pd.DataFrame(st.session_state.log_history)),
                             use_container_width=True, hide_index=True)
        else:
            st.info("**System ready.** Select a scenario in the sidebar and press **🚀 INITIATE MONITORING**.")
            if mode != "Normal":
                exp = ATTACK_EXPLAIN.get(mode, {})
                if exp:
                    st.markdown(f"#### ℹ️ About: {mode}")
                    st.markdown(f"**What it does:** {exp['what']}")
                    st.markdown(f"**Attack mechanism:** {exp['why']}")
                    st.markdown(f"**Key anomalous features:** {', '.join(exp['features'])}")

# ──────────────────────────────────────────────────────────────
# TAB 2 — SYNTHETIC DATASET EVALUATION
# ──────────────────────────────────────────────────────────────
with tab_synthetic:
    st.markdown("### 🧪 Synthetic SWaT Dataset Generator & Model Evaluator")
    st.markdown(
        "Generate a labelled dataset with realistic Normal + Attack samples, "
        "score every row through your Isolation Forest, and see exactly where "
        "your model succeeds or struggles — so you can tune it."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        n_rows = st.slider("Total Rows", 200, 5000, 1000, step=100)
    with col_cfg2:
        attack_ratio = st.slider(
            "Attack Ratio", 0.1, 0.6, 0.3, step=0.05,
            help="Fraction of rows that are attack samples. Real SWaT dataset is ~12% attack.",
        )
    with col_cfg3:
        st.metric("Normal rows", f"{int(n_rows * (1 - attack_ratio)):,}")
        st.metric("Attack rows", f"{int(n_rows * attack_ratio):,}")

    gen_btn = st.button("⚡ GENERATE & EVALUATE", use_container_width=True, type="primary")

    if gen_btn:
        with st.spinner("Generating synthetic dataset and scoring with your model..."):
            model_synth = joblib.load(MODEL_PATH)
            df_synth    = generate_synthetic_dataset(n_rows, attack_ratio)
            df_result, ev_synth, scores = evaluate_on_synthetic(model_synth, df_synth, threshold)

        st.success(
            f"✅ Generated {len(df_synth):,} rows · "
            f"{int(df_synth['label'].sum()):,} attack · "
            f"{int((df_synth['label'] == 0).sum()):,} normal"
        )
        st.markdown("---")

        st.markdown("#### 📊 Model Evaluation on Synthetic Dataset")
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Accuracy",  f"{_acc(ev_synth):.1%}")
        mc2.metric("F1-Score",  f"{_f1(ev_synth):.3f}")
        mc3.metric("Precision", f"{_prec(ev_synth):.3f}")
        mc4.metric("Recall",    f"{_rec(ev_synth):.3f}")
        mc5.metric("TP + TN",   ev_synth["TP"] + ev_synth["TN"])
        mc6.metric("FP + FN",   ev_synth["FP"] + ev_synth["FN"])

        with st.expander("Confusion Matrix Detail"):
            st.caption(
                f"**TP** {ev_synth['TP']} | **TN** {ev_synth['TN']} | "
                f"**FP** {ev_synth['FP']} | **FN** {ev_synth['FN']}"
            )

        st.markdown("---")

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("#### 🎯 Detection Rate per Attack Type")
            st.plotly_chart(per_attack_bar(df_result), use_container_width=True,
                            config={"displayModeBar": False}, key="synth_bar")
        with ch2:
            st.markdown("#### 📉 Score Distribution: Normal vs Attack")
            st.plotly_chart(score_dist_chart(df_result, threshold), use_container_width=True,
                            config={"displayModeBar": False}, key="synth_dist")

        st.markdown("---")
        st.markdown("#### 🔧 Model Tuning Insights")
        prec_val = _prec(ev_synth)
        rec_val  = _rec(ev_synth)
        if rec_val < 0.7:
            st.warning(
                f"⚠️ **Low Recall ({rec_val:.1%})** — model is missing many attacks (high FN). "
                "Lower the Isolation Forest `contamination` parameter when retraining, "
                "or move the threshold slider less negative (e.g. −0.20)."
            )
        if prec_val < 0.7:
            st.warning(
                f"⚠️ **Low Precision ({prec_val:.1%})** — too many false alarms (high FP). "
                "Raise `contamination` when retraining, or move threshold more negative (e.g. −0.35)."
            )
        if rec_val >= 0.85 and prec_val >= 0.85:
            st.success(
                f"✅ **Good separation** — Recall {rec_val:.1%}, Precision {prec_val:.1%}. "
                "Model performs well on this synthetic set. Validate against real SWaT data to confirm."
            )

        st.markdown("#### 📋 Per-Attack Breakdown")
        atk_summary = (
            df_result.groupby("attack")
            .agg(Total=("label", "count"), Detected=("correct", "sum"),
                 Avg_Score=("ai_score", "mean"), Min_Score=("ai_score", "min"))
            .reset_index()
        )
        atk_summary["Detection %"] = (atk_summary["Detected"] / atk_summary["Total"] * 100).round(1)
        atk_summary["Avg_Score"]   = atk_summary["Avg_Score"].round(4)
        atk_summary["Min_Score"]   = atk_summary["Min_Score"].round(4)
        st.dataframe(atk_summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        full_csv = df_result.to_csv(index=False).encode()
        st.download_button(
            "💾 Download Scored Dataset (CSV)", full_csv,
            file_name=f"waterguard_synthetic_{n_rows}rows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True,
        )
        st.caption(
            "CSV contains all 105 features, true label (0=Normal / 1=Attack), "
            "attack type, AI score, predicted label, and a correct flag. "
            "Use it to retrain or fine-tune your Isolation Forest."
        )

    else:
        st.info(
            "Configure the dataset size and attack ratio above, then click "
            "**⚡ GENERATE & EVALUATE** to run your model against synthetic SWaT data."
        )
        st.markdown("""
**How it works:**
- Normal samples are generated by the same SWaT Stage-1 physics engine as the live monitor
- Each attack type injects features **specifically anomalous** for that attack:
  - *Valve Manipulation* → FIT101 = 1.6× normal, MV101 forced open
  - *Pump Override* → P101=1 with LIT101 < 50 mm, near-zero flow
  - *Sensor Spoofing* → LIT101 frozen at 500 mm while FIT101 elevated
  - *Chemical Dosing* → AIT202 spike to ~13.8 (pH danger zone)
- Every row is scored by your loaded `isolation_forest_model.pkl`
- Full confusion matrix, per-attack detection rate, score distribution, and tuning advice
- Download the scored CSV to retrain / fine-tune your model offline
        """)

# ──────────────────────────────────────────────────────────────
# TAB 3 — CHATBOT
# ──────────────────────────────────────────────────────────────
with tab_chatbot:
    st.markdown("### 🤖 AI Chatbot")
    st.caption("Chat with an AI assistant powered by OpenRouter.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.form(key="chat_form"):
        user_input = st.text_input("Type your message here:")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        with st.spinner("Thinking..."):
            response = chat_with_openrouter(user_input.strip())
        st.session_state.chat_history.append({"user": user_input.strip(), "bot": response})

    st.markdown("---")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            st.markdown(f"**You:** {msg['user']}")
            st.markdown(f"**Bot:** {msg['bot']}")
            st.markdown("---")