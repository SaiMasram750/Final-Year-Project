import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & SESSION STATE ---
st.set_page_config(page_title="WaterGuard-X Master SOC", layout="wide")
MODEL_PATH = "isolation_forest_model.pkl"

# Initialize all metrics and histories
if 'eval' not in st.session_state:
    st.session_state.eval = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
if 'log_history' not in st.session_state:
    st.session_state.log_history = []
if 'score_history' not in st.session_state:
    st.session_state.score_history = []
if 'process_state' not in st.session_state:
    # Starting state for the Digital Twin [cite: 22, 134]
    st.session_state.process_state = {'LIT101': 500, 'MV101': 0, 'P101': 0, 'FIT101': 0, 'AIT202': 7.2}

# --- 2. DYNAMIC PHYSICS ENGINE (The Digital Twin) ---
def simulate_swat_step(prev_state, mode):
    """Simulates SWaT Stage 1 physics and mass balance[cite: 115, 121]."""
    level = prev_state.get('LIT101', 500)
    valve = prev_state.get('MV101', 0)
    pump = prev_state.get('P101', 0)

    # PLC Logic Simulation (Normal Operation) [cite: 35]
    if mode == "Normal":
        if level < 300: valve = 1  # Low-level trigger
        if level > 700: valve = 0  # High-level trigger
        pump = 1 if level > 250 else 0
    
    # Attack Logic Overrides [cite: 13, 201]
    elif mode == "Valve Manipulation": valve = 1 
    elif mode == "Pump Override": pump = 1; level = 40 
    elif mode == "Sensor Spoofing": valve = 1 # Real level rises, but reported level stays static
    
    # Physics: Mass Balance Equation [cite: 115, 151]
    inflow = 15.0 if valve == 1 else 0
    outflow = 8.0 if pump == 1 else 0
    new_level = level + (inflow - outflow) + np.random.normal(0, 0.3)
    new_level = max(0, min(1100, new_level))

    row = {
        'FIT101': np.random.normal(2.5, 0.05) if valve == 1 else 0.02,
        'LIT101': 500 if mode == "Sensor Spoofing" else new_level,
        'AIT202': 13.8 if mode == "Chemical Dosing" else np.random.normal(7.2, 0.1),
        'P101': pump, 
        'MV101': valve
    }
    # Pad to 105 features to match SWaT model requirements [cite: 23, 143]
    full_row = {**row, **{f'feat_{i}': np.random.choice([0,1]) for i in range(5, 105)}}
    return pd.Series(full_row), row

# --- 3. FORENSIC REASONING ENGINE ---
def get_forensic_report(state, score, sensitivity):
    """Validates anomalies against physical laws to reduce false alarms[cite: 25, 114]."""
    reasons = []
    # Physical/Safety Constraints [cite: 151]
    if state['LIT101'] > 900: reasons.append("PHYSICAL: Tank Overflow Imminent")
    if state['P101'] == 1 and state['LIT101'] < 50: reasons.append("SAFETY: Pump Dry-Run Protection")
    if state['AIT202'] > 12: reasons.append("CHEMICAL: Dangerous pH Level")
    # Logic Invariants [cite: 116]
    if state['MV101'] == 0 and state['FIT101'] > 0.5: reasons.append("LOGIC: Unordered Flow Detected")
    # AI Statistical Anomaly [cite: 111, 112]
    if score < sensitivity: reasons.append(f"AI: Statistical Anomaly (Score: {score:.3f})")
    
    return " | ".join(reasons) if reasons else "Normal"

# --- 4. THEME & UI ---
st.markdown("""
    <style>
    .status-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; color: white; margin-bottom: 10px; }
    .metric-row { background: #1d2129; padding: 10px; border-radius: 8px; border: 1px solid #3d4450; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🛡️ WaterGuard-X")
    st.caption("Hybrid Cyber-Physical Security [cite: 17]")
    mode = st.selectbox("Attack Scenario", ["Normal", "Valve Manipulation", "Pump Override", "Sensor Spoofing", "Chemical Dosing"])
    sensitivity = st.slider("AI Sensitivity Threshold", -0.5, 0.0, -0.25)
    run = st.button("🚀 INITIATE MONITORING", use_container_width=True)
    if st.button("🗑️ RESET ALL DATA"):
        st.session_state.eval = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        st.session_state.log_history = []
        st.session_state.score_history = []
        st.rerun()

# --- 6. LIVE MONITORING LOOP ---
if run:
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    placeholder = st.empty()

    for _ in range(300):
        # Data & Inference [cite: 100-113]
        full_data, state = simulate_swat_step(st.session_state.process_state, mode)
        st.session_state.process_state = state
        score = model.decision_function(full_data.values.reshape(1, -1))[0] if model else 0
        st.session_state.score_history.append(score)
        
        # Validation & Reasoning [cite: 114, 151]
        reason = get_forensic_report(state, score, sensitivity)
        is_detected = (reason != "Normal")
        is_actual_attack = (mode != "Normal")

        # Update Confusion Matrix [cite: 148, 157]
        if is_actual_attack and is_detected: st.session_state.eval["TP"] += 1
        elif not is_actual_attack and not is_detected: st.session_state.eval["TN"] += 1
        elif not is_actual_attack and is_detected: st.session_state.eval["FP"] += 1
        elif is_actual_attack and not is_detected: st.session_state.eval["FN"] += 1

        with placeholder.container():
            # --- ROW 1: FORENSIC LOGS (TOP PRIORITY) [cite: 128, 158] ---
            st.subheader("🕵️ Live Forensic Incident Logs")
            if is_detected:
                st.session_state.log_history.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "Source": mode,
                    "Reason for Alarm": reason,
                    "AI Score": round(score, 3)
                })
            
            log_df = pd.DataFrame(st.session_state.log_history).tail(5)
            if not log_df.empty:
                st.table(log_df)
            else:
                st.success("Monitoring stream active. No anomalies found.")

            st.divider()

            # --- ROW 2: STATUS & EVALUATION SUITE [cite: 148, 167] ---
            c_status, c_metrics = st.columns([1, 2.5])
            with c_status:
                label, color = ("🚨 ATTACK DETECTED", "#d9534f") if is_detected else ("✅ SYSTEM SECURE", "#28a745")
                st.markdown(f'<div class="status-box" style="background-color:{color};">{label}</div>', unsafe_allow_html=True)
                st.write(f"**Mode:** {mode}")
            
            with c_metrics:
                ev = st.session_state.eval
                total = sum(ev.values())
                acc = (ev["TP"] + ev["TN"]) / total if total > 0 else 0
                prec = ev["TP"] / (ev["TP"] + ev["FP"]) if (ev["TP"] + ev["FP"]) > 0 else 1.0
                rec = ev["TP"] / (ev["TP"] + ev["FN"]) if (ev["TP"] + ev["FN"]) > 0 else 1.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{acc:.1%}")
                m2.metric("TP (Hits)", ev["TP"])
                m3.metric("TN (Safe)", ev["TN"])
                m4.metric("FP (Alarms)", ev["FP"])
                m5.metric("FN (Misses)", ev["FN"])
                st.caption(f"**Precision:** {prec:.2f} | **Recall:** {rec:.2f} | **F1-Score:** {f1:.2f}")

            st.divider()

            # --- ROW 3: PROCESS STATE & ANOMALY GRAPH [cite: 128, 136] ---
            v1, v2 = st.columns([1, 2])
            with v1:
                st.subheader("💧 Sensors")
                st.metric("Tank Level", f"{state['LIT101']:.1f} mm", delta=f"{state['LIT101']-500:.1f}")
                st.write(f"**Valve:** {'🟢 OPEN' if state['MV101']==1 else '⚪ CLOSED'}")
                st.write(f"**Pump:** {'🟢 ON' if state['P101']==1 else '⚪ OFF'}")
                st.write(f"**pH:** {state['AIT202']:.2f}")

            with v2:
                st.subheader("📈 AI Anomaly Score Timeline")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=st.session_state.score_history[-50:], mode='lines', line=dict(color='#00d1ff', width=2.5)))
                fig.add_hline(y=sensitivity, line_dash="dash", line_color="#ff4b4b", annotation_text="DANGER")
                fig.update_layout(height=230, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        time.sleep(0.4) # Control loop frequency [cite: 127]
else:
    st.info("System Ready. Use the sidebar to initiate the WaterGuard-X simulation loop[cite: 154].")