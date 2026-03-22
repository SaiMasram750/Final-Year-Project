"""
WaterGuard-X Master SOC  ·  Improved Edition
=============================================
Improvements over original
──────────────────────────
UI / UX & Visual Design
  · Industrial-terminal aesthetic: deep navy/slate, amber + cyan palette,
    Share Tech Mono + Barlow Condensed typography, subtle CRT scan-line overlay
  · Persistent KPI ribbon (Accuracy, F1, Precision, Recall, TP/TN/FP/FN)
    rendered as custom HTML — never collapses or wraps awkwardly
  · Severity-coloured status banner: NORMAL/LOW/MEDIUM/HIGH/CRITICAL each
    get a distinct colour, not just red/green
  · Tank level as a Plotly Indicator gauge with colour-banded safety zones
    (dry-run zone, safe zone, overflow zone) and a delta from set-point
  · Anomaly score chart redesigned: bars coloured cyan/red relative to
    threshold; overlaid EMA trend line; danger zone shading beneath threshold
  · Incident log rows colour-coded by severity via Pandas Styler
  · Idle screen shows last-session KPIs and full incident log

Features & Functionality
  · Five alert severity tiers: NORMAL → LOW → MEDIUM → HIGH → CRITICAL
  · "Pump Override" attack now correctly forces pump=1 while PLC still
    governs the valve — previous code clamped level to 40 unconditionally
  · "Chemical Dosing" runs normal PLC logic for valve and pump in parallel
  · Sensor Spoofing attack: valve forced open, but reported level to ML
    model stays static at 500 mm (ground-truth level still rises)
  · Incident log captures: timestamp, step, scenario, severity, all alarm
    reasons, AI score, tank level, pH, valve state, pump state
  · CSV export of the complete incident log (sidebar download button)
  · Configurable loop iterations (50–600) and step delay via sidebar sliders
  · Progress bar shows step N/total and last score while loop runs

Performance & Reliability
  · Model loaded exactly once before the loop (joblib.load is ~100 ms)
  · simulate_step() is a pure function — no session state mutation inside it
  · score_history list appended once per step; no re-creation of list objects
  · st.empty() placeholder rendered once; updated in-place every step
  · All magic numbers replaced with named constants
  · Confusion matrix stored as a flat dict; metrics computed on demand
  · EMA smoothing for the trend line done in O(n) with a single pass
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
MODEL_PATH      = "isolation_forest_model.pkl"
TANK_MIN        = 0.0
TANK_MAX        = 1100.0
TANK_CRITICAL_L = 50.0    # dry-run danger below this
TANK_LOW        = 300.0   # PLC opens valve below this
TANK_HIGH       = 700.0   # PLC closes valve above this
TANK_OVERFLOW   = 900.0   # physical overflow warning
INFLOW_RATE     = 15.0    # mm / step when valve open
OUTFLOW_RATE    = 8.0     # mm / step when pump running
LEVEL_NOISE     = 0.3
FLOW_BASE       = 2.5
FLOW_NOISE      = 0.05
FLOW_IDLE       = 0.02
PH_NORMAL       = 7.2
PH_NOISE        = 0.1
PH_ATTACK       = 13.8
PH_DANGER       = 12.0
NUM_PAD_FEATS   = 100     # pad to 105-feature model input

ATTACK_MODES = [
    "Normal",
    "Valve Manipulation",
    "Pump Override",
    "Sensor Spoofing",
    "Chemical Dosing",
]

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

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WaterGuard-X SOC",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS  — industrial terminal aesthetic
# ─────────────────────────────────────────────────────────
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
    font-size: 1.45rem !important;
    color: #00e5ff !important;
}
[data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"]  { font-family: 'Share Tech Mono', monospace !important; }
[data-testid="stSidebar"]      { background: #0d1117 !important; border-right: 1px solid #21262d !important; }
[data-testid="stSidebar"] *    { color: #c9d1d9 !important; }
h1, h2, h3, h4                 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0.05em !important; color: #e6edf3 !important; }
hr                             { border-color: #21262d !important; }
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #00838f, #006064) !important;
    border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.08em !important;
}
[data-testid="baseButton-secondary"] { border-color: #30363d !important; background: #161b22 !important; }
[data-testid="stDataFrame"]          { border: 1px solid #21262d !important; border-radius: 6px; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #00838f, #00e5ff) !important; }
[data-testid="stAlert"]              { background: #161b22 !important; border-color: #21262d !important; }
.block-container                     { padding-top: 1.2rem !important; }

/* KPI ribbon */
.kpi-ribbon {
    display: flex; gap: 14px; flex-wrap: wrap;
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 11px 18px; margin-bottom: 12px;
}
.kpi-item { text-align: center; min-width: 76px; }
.kpi-val  { font-family: 'Share Tech Mono', monospace; font-size: 1.3rem; color: #00e5ff; line-height: 1.2; }
.kpi-lbl  { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: .06em; }

/* Status banner */
.status-banner {
    padding: 12px 20px; border-radius: 8px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.2rem; font-weight: 700; letter-spacing: .08em;
    color: #fff; text-align: center; margin-bottom: 8px;
}

/* Subtle CRT scan-line overlay */
.scanline {
    position: fixed; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,229,255,0.012) 2px, rgba(0,229,255,0.012) 4px
    );
    pointer-events: none; z-index: 9999;
}
</style>
<div class="scanline"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
_INIT_PROCESS = {
    "LIT101": 500.0, "MV101": 0, "P101": 0,
    "FIT101": 0.0,   "AIT202": PH_NORMAL,
}

def _init_state():
    defaults = {
        "eval":          {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
        "log_history":   [],
        "score_history": [],
        "process_state": _INIT_PROCESS.copy(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _reset_state():
    st.session_state.eval          = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    st.session_state.log_history   = []
    st.session_state.score_history = []
    st.session_state.process_state = _INIT_PROCESS.copy()

_init_state()

# ─────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
# PHYSICS ENGINE  (pure function)
# ─────────────────────────────────────────────────────────
def _plc_normal(level, mv, pump):
    """Standard PLC ladder logic for Stage 1."""
    if level < TANK_LOW:
        mv = 1
    elif level > TANK_HIGH:
        mv = 0
    pump = 1 if level > TANK_LOW * 0.83 else 0
    return mv, pump


def simulate_step(prev: dict, mode: str):
    """
    Advance the SWaT Stage-1 digital twin by one step.
    Returns (full_row: pd.Series, new_state: dict).
    """
    level = prev["LIT101"]
    mv    = prev["MV101"]
    pump  = prev["P101"]

    if mode == "Normal":
        mv, pump = _plc_normal(level, mv, pump)

    elif mode == "Valve Manipulation":
        mv = 1
        _, pump = _plc_normal(level, mv, pump)

    elif mode == "Pump Override":
        mv, pump = _plc_normal(level, mv, pump)
        pump = 1                         # attacker forces pump ON

    elif mode == "Sensor Spoofing":
        mv = 1                           # attacker holds valve open
        _, pump = _plc_normal(level, mv, pump)

    elif mode == "Chemical Dosing":
        mv, pump = _plc_normal(level, mv, pump)

    inflow    = INFLOW_RATE  if mv   == 1 else 0.0
    outflow   = OUTFLOW_RATE if pump == 1 else 0.0
    new_level = float(np.clip(
        level + inflow - outflow + np.random.normal(0, LEVEL_NOISE),
        TANK_MIN, TANK_MAX,
    ))

    ph   = PH_ATTACK if mode == "Chemical Dosing" \
           else float(np.random.normal(PH_NORMAL, PH_NOISE))
    flow = float(np.random.normal(FLOW_BASE, FLOW_NOISE)) if mv == 1 else FLOW_IDLE

    # Spoofing: model receives frozen level, ground truth still rises
    reported_level = 500.0 if mode == "Sensor Spoofing" else new_level

    new_state = {
        "LIT101": new_level, "MV101": mv,
        "P101": pump, "FIT101": flow, "AIT202": ph,
    }
    base    = {"FIT101": flow, "LIT101": reported_level,
               "AIT202": ph,  "P101":   float(pump), "MV101": float(mv)}
    padding = {f"feat_{i}": float(np.random.randint(0, 2))
               for i in range(5, 5 + NUM_PAD_FEATS)}
    return pd.Series({**base, **padding}), new_state

# ─────────────────────────────────────────────────────────
# FORENSIC REASONING ENGINE
# ─────────────────────────────────────────────────────────
def _severity(reasons):
    joined = " ".join(reasons)
    if "OVERFLOW" in joined or "CHEMICAL" in joined or "DRY-RUN" in joined:
        return "CRITICAL"
    if "SAFETY" in joined or "AI:" in joined:
        return "HIGH"
    if "LOGIC" in joined or "PHYSICAL" in joined:
        return "MEDIUM"
    return "LOW" if reasons else "NORMAL"


def forensic_report(state: dict, score: float, threshold: float):
    reasons = []
    if state["LIT101"] > TANK_OVERFLOW:
        reasons.append("PHYSICAL: Tank Overflow Imminent")
    if state["P101"] == 1 and state["LIT101"] < TANK_CRITICAL_L:
        reasons.append("SAFETY: Pump Dry-Run Protection")
    if state["AIT202"] > PH_DANGER:
        reasons.append("CHEMICAL: Dangerous pH Level")
    if state["MV101"] == 0 and state["FIT101"] > 0.5:
        reasons.append("LOGIC: Unordered Flow Detected")
    if score < threshold:
        reasons.append(f"AI: Statistical Anomaly (Score: {score:.3f})")
    sev = _severity(reasons)
    return (" | ".join(reasons) if reasons else "Normal"), sev

# ─────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────
def gauge_chart(level: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level,
        delta={"reference": 500.0, "valueformat": ".0f",
               "increasing": {"color": "#fc8181"},
               "decreasing": {"color": "#68d391"}},
        number={"font": {"family": "Share Tech Mono", "color": "#00e5ff", "size": 28}},
        title={"text": "TANK LEVEL (mm)",
               "font": {"family": "Barlow Condensed", "size": 12, "color": "#8b949e"}},
        gauge={
            "axis": {"range": [TANK_MIN, TANK_MAX],
                     "tickfont": {"family": "Share Tech Mono", "size": 8},
                     "tickcolor": "#30363d"},
            "bar":  {"color": "#00838f", "thickness": 0.22},
            "bgcolor": "#0d1117", "bordercolor": "#21262d",
            "steps": [
                {"range": [TANK_MIN,        TANK_CRITICAL_L], "color": "#4a1010"},
                {"range": [TANK_CRITICAL_L, TANK_LOW],        "color": "#3d2000"},
                {"range": [TANK_LOW,        TANK_HIGH],       "color": "#0a1f14"},
                {"range": [TANK_HIGH,       TANK_OVERFLOW],   "color": "#3d2000"},
                {"range": [TANK_OVERFLOW,   TANK_MAX],        "color": "#4a1010"},
            ],
            "threshold": {
                "line": {"color": "#fc8181", "width": 3},
                "thickness": 0.85, "value": TANK_OVERFLOW,
            },
        },
    ))
    fig.update_layout(
        height=225, margin=dict(l=10, r=10, t=15, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def score_chart(history: list, threshold: float) -> go.Figure:
    window  = history[-60:]
    if not window:
        return go.Figure()
    x       = list(range(len(window)))
    colours = ["#fc8181" if s < threshold else "#00838f" for s in window]

    # EMA trend
    alpha, ema, val = 0.25, [], window[0]
    for s in window:
        val = alpha * s + (1 - alpha) * val
        ema.append(val)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=window, marker_color=colours, marker_line_width=0,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=ema, mode="lines",
        line=dict(color="#e2e8f0", width=1.5, dash="dot"),
        showlegend=False,
    ))
    ymin = min(window + [threshold]) - 0.05
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#fc8181",
        annotation_text=f"threshold {threshold}",
        annotation_font=dict(color="#fc8181", size=9, family="Share Tech Mono"),
        annotation_position="top right",
    )
    fig.add_hrect(y0=ymin, y1=threshold,
                  fillcolor="rgba(229,62,62,0.07)", line_width=0)
    fig.update_layout(
        height=225, margin=dict(l=0, r=0, t=15, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, color="#8b949e",
                   tickfont=dict(family="Share Tech Mono", size=8)),
        yaxis=dict(gridcolor="#21262d", zeroline=False, color="#8b949e",
                   tickfont=dict(family="Share Tech Mono", size=8),
                   title=dict(text="score", font=dict(size=8, color="#8b949e"))),
        bargap=0.04,
    )
    return fig

# ─────────────────────────────────────────────────────────
# LOG STYLER
# ─────────────────────────────────────────────────────────
def style_log(df: pd.DataFrame):
    def _row(row):
        css = SEV_ROW_CSS.get(row.get("Severity", ""), "")
        return [css] * len(row)
    return df.style.apply(_row, axis=1)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ WATERGUARD-X")
    st.caption("Cyber-Physical Security Operations Centre")
    st.divider()

    mode       = st.selectbox("Attack Scenario", ATTACK_MODES)
    threshold  = st.slider("AI Sensitivity Threshold", -0.5, 0.0, -0.25, step=0.01,
                           help="Scores below this line trigger an AI alert.")
    loop_count = st.slider("Loop Iterations", 50, 600, 300, step=50)
    loop_delay = st.slider("Step Delay (s)", 0.05, 1.0, 0.35, step=0.05)
    st.divider()

    run   = st.button("🚀 INITIATE MONITORING", use_container_width=True, type="primary")
    reset = st.button("🗑️ RESET ALL DATA",      use_container_width=True)

    if st.session_state.log_history:
        st.divider()
        csv_bytes = (
            pd.DataFrame(st.session_state.log_history)
            .to_csv(index=False).encode()
        )
        st.download_button(
            "💾 Export Incident Log (CSV)",
            csv_bytes,
            file_name=f"waterguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

if reset:
    _reset_state()
    st.rerun()

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("# 🛡️ WATERGUARD-X  ·  MASTER SOC")
st.caption(
    f"Hybrid Cyber-Physical Intrusion Detection  ·  "
    f"{datetime.now().strftime('%Y-%m-%d')}  ·  "
    f"Scenario: **{mode}**  ·  Threshold: **{threshold}**"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────
# MONITORING LOOP
# ─────────────────────────────────────────────────────────
if run:
    model = joblib.load(MODEL_PATH)   # loaded once

    prog_ph = st.empty()
    dash_ph = st.empty()

    for step in range(loop_count):
        # Physics + ML scoring
        full_data, new_state = simulate_step(st.session_state.process_state, mode)
        st.session_state.process_state = new_state
        score = float(model.decision_function(full_data.values.reshape(1, -1))[0])
        st.session_state.score_history.append(score)

        # Forensics
        reason, severity = forensic_report(new_state, score, threshold)
        is_detected = severity != "NORMAL"
        is_attack   = mode != "Normal"

        # Confusion matrix update
        ev = st.session_state.eval
        if   is_attack and     is_detected: ev["TP"] += 1
        elif not is_attack and not is_detected: ev["TN"] += 1
        elif not is_attack and     is_detected: ev["FP"] += 1
        elif is_attack and not is_detected: ev["FN"] += 1

        # Incident log
        if is_detected:
            st.session_state.log_history.append({
                "Timestamp":    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "Step":         step + 1,
                "Scenario":     mode,
                "Severity":     severity,
                "Alarm Reason": reason,
                "AI Score":     round(score, 4),
                "Level (mm)":   round(new_state["LIT101"], 1),
                "pH":           round(new_state["AIT202"], 2),
                "Valve":        "OPEN"  if new_state["MV101"] == 1 else "CLOSED",
                "Pump":         "ON"    if new_state["P101"]  == 1 else "OFF",
            })

        # Progress bar (separate placeholder — doesn't flicker)
        prog_ph.progress(
            (step + 1) / loop_count,
            text=f"Step {step+1}/{loop_count}  ·  score: {score:.4f}",
        )

        # ── DASHBOARD RENDER ──────────────────────────
        with dash_ph.container():

            # Status banner
            col = SEV_COLOR.get(severity, SEV_COLOR["NORMAL"])
            lbl = f"🚨 {severity}  —  ATTACK DETECTED" if is_detected else "✅  SYSTEM SECURE"
            st.markdown(
                f'<div class="status-banner" style="background:{col};">{lbl}</div>',
                unsafe_allow_html=True,
            )

            # KPI ribbon
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
                f'<div class="kpi-item">'
                f'<div class="kpi-val">{v}</div>'
                f'<div class="kpi-lbl">{l}</div>'
                f'</div>'
                for l, v in kpis
            )
            st.markdown(f'<div class="kpi-ribbon">{ribbon}</div>', unsafe_allow_html=True)
            st.markdown("---")

            # Gauge | Score chart | Sensor readings
            cg, cs, cp = st.columns([1.1, 2.2, 0.9])

            with cg:
                st.markdown("#### 💧 Tank Level")
                st.plotly_chart(gauge_chart(new_state["LIT101"]),
                                use_container_width=True,
                                config={"displayModeBar": False})

            with cs:
                st.markdown("#### 📈 AI Anomaly Score  (last 60 steps)")
                st.plotly_chart(
                    score_chart(st.session_state.score_history, threshold),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            with cp:
                st.markdown("#### 🔌 Sensors")
                st.metric("Flow FIT101",  f"{new_state['FIT101']:.2f} m³/h")
                st.metric("pH   AIT202",  f"{new_state['AIT202']:.2f}")
                st.metric("Level LIT101", f"{new_state['LIT101']:.0f} mm",
                          delta=f"{new_state['LIT101'] - 500:.0f}")
                st.write(f"**Valve MV101:** {'🟢 OPEN' if new_state['MV101']==1 else '⚪ CLOSED'}")
                st.write(f"**Pump  P101:**  {'🟢 ON'   if new_state['P101'] ==1 else '⚪ OFF'}")

            st.markdown("---")

            # Forensic incident log
            st.markdown("#### 🕵️ Live Forensic Incident Log")
            log_df = pd.DataFrame(st.session_state.log_history).tail(8)
            if not log_df.empty:
                st.dataframe(style_log(log_df), use_container_width=True, hide_index=True)
            else:
                st.success("✅  Monitoring active — no anomalies detected yet.")

        time.sleep(loop_delay)

    prog_ph.success(f"✅  Monitoring complete — {loop_count} steps processed.")

# ─────────────────────────────────────────────────────────
# IDLE STATE
# ─────────────────────────────────────────────────────────
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
            full_log = pd.DataFrame(st.session_state.log_history)
            st.dataframe(style_log(full_log), use_container_width=True, hide_index=True)
    else:
        st.info(
            "**System ready.**  "
            "Select a scenario in the sidebar and press **🚀 INITIATE MONITORING**."
        )