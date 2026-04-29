import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Attribute QC Chart Generator", layout="wide")
st.title("📊 Attribute Control Chart Generator (Plotly)")
st.markdown("Interactive application for attribute control charts (**p, np, c, u Chart**) with Phase 1 (Base) and Phase 2 (Monitoring) capabilities.")

# --- SIDEBAR (INPUT PARAMETERS) ---
st.sidebar.header("⚙️ Chart Parameters")

chart_options = st.sidebar.multiselect(
    "Select Charts to Generate:",
    ["p-Chart", "np-Chart", "c-Chart", "u-Chart"],
    default=["p-Chart", "c-Chart"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Phase 1 Settings (Base Data)")
num_samples_p1 = st.sidebar.number_input("Phase 1: Number of Samples (m)", min_value=1, value=20, step=1)
sample_size_p1 = st.sidebar.number_input("Phase 1: Sample Size (n)", min_value=1, value=50, step=1)
st.sidebar.info("💡 **Note:** For this application, the sample size (n) is assumed constant across all observation points.")

# --- ALARM LOGIC (Full Western Electric Rules) ---
def check_alarms(data, mean_val, ucl, lcl):
    alarms = []
    violation_indices = set()
    n = len(data)
    valid_mask = ~data.isna()
    
    # 0. Out of bounds (>UCL or <LCL)
    for i in range(n):
        if not valid_mask.iloc[i]: continue
        if data.iloc[i] > ucl or data.iloc[i] < lcl:
            disp_idx = str(data.index[i]).split('-', 1)[-1] if '-' in str(data.index[i]) else data.index[i]
            alarms.append(f"**[Type 0]** Point **{disp_idx}** is out of Control Limits (>UCL or <LCL).")
            violation_indices.add(data.index[i])

    sigma = (ucl - mean_val) / 3
    signs = np.sign(data - mean_val)
    
    # 1. 7 points in a row on one side
    for i in range(n - 6):
        if not valid_mask.iloc[i:i+7].all(): continue
        if np.all(signs.iloc[i:i+7] == 1) or np.all(signs.iloc[i:i+7] == -1):
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+6]).split('-', 1)[-1]
            alarms.append(f"**[Type 1]** **{d_start} to {d_end}**: 7 consecutive points on one side.")
            violation_indices.update(data.index[i:i+7])
            
    # 2. 10 out of 11 points on one side
    for i in range(n - 10):
        if not valid_mask.iloc[i:i+11].all(): continue
        window = signs.iloc[i:i+11]
        if np.sum(window == 1) >= 10 or np.sum(window == -1) >= 10:
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+10]).split('-', 1)[-1]
            alarms.append(f"**[Type 2]** **{d_start} to {d_end}**: 10 out of 11 points on one side.")
            violation_indices.update(data.index[i:i+11])
            
    # 3. 12 out of 14 points on one side
    for i in range(n - 13):
        if not valid_mask.iloc[i:i+14].all(): continue
        window = signs.iloc[i:i+14]
        if np.sum(window == 1) >= 12 or np.sum(window == -1) >= 12:
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+13]).split('-', 1)[-1]
            alarms.append(f"**[Type 3]** **{d_start} to {d_end}**: 12 out of 14 points on one side.")
            violation_indices.update(data.index[i:i+14])
            
    # 4. 6 points steadily increasing/decreasing
    for i in range(n - 5):
        if not valid_mask.iloc[i:i+6].all(): continue
        diffs = np.diff(data.iloc[i:i+6])
        if np.all(diffs > 0) or np.all(diffs < 0):
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+5]).split('-', 1)[-1]
            alarms.append(f"**[Type 4]** **{d_start} to {d_end}**: 6 consecutive points steadily increasing/decreasing.")
            violation_indices.update(data.index[i:i+6])
            
    # 5. 2 out of 3 in Zone A (same side)
    for i in range(n - 2):
        if not valid_mask.iloc[i:i+3].all(): continue
        window = data.iloc[i:i+3]
        if np.sum(window > mean_val + 2*sigma) >= 2 or np.sum(window < mean_val - 2*sigma) >= 2:
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+2]).split('-', 1)[-1]
            alarms.append(f"**[Type 5]** **{d_start} to {d_end}**: 2 out of 3 points in Zone A.")
            violation_indices.update(data.index[i:i+3])
            
    # 6. 4 out of 5 in Zone B or beyond (same side)
    for i in range(n - 4):
        if not valid_mask.iloc[i:i+5].all(): continue
        window = data.iloc[i:i+5]
        if np.sum(window > mean_val + sigma) >= 4 or np.sum(window < mean_val - sigma) >= 4:
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+4]).split('-', 1)[-1]
            alarms.append(f"**[Type 6]** **{d_start} to {d_end}**: 4 out of 5 points in Zone B or beyond.")
            violation_indices.update(data.index[i:i+5])

    if not alarms:
        alarms.append("✅ Process is in control.")
        
    return list(dict.fromkeys(alarms)), list(violation_indices)

# --- MAIN GUI ---
st.subheader("📝 Phase 1: Base Observation Data (Limits Calculation)")
index_p1 = [f"Sample {i+1}" for i in range(num_samples_p1)]
df_phase1 = st.data_editor(pd.DataFrame(0, index=index_p1, columns=["Count (D or c)"]), use_container_width=True, key="p1_editor")

st.markdown("### ✂️ Data Cleaning (Phase 1)")
st.info("Select samples to exclude (e.g., outliers) so they are not included in the Center Line and UCL/LCL calculations.")
excluded_samples = st.multiselect("Exclude Phase 1 Samples:", options=index_p1)

st.divider()

st.subheader("📝 Phase 2: Monitoring Data (Optional)")
use_phase2 = st.checkbox("Enable Phase 2 (Add new observations to evaluate against Phase 1 limits)")

df_phase2 = pd.DataFrame()
if use_phase2:
    col_p2_1, col_p2_2 = st.columns(2)
    with col_p2_1:
        num_samples_p2 = st.number_input("Phase 2: Number of Samples (m)", min_value=1, value=10, step=1)
    with col_p2_2:
        st.write(f"Phase 2 Sample Size (n) is fixed to **{sample_size_p1}** to match the Phase 1 structure.")
        
    index_p2 = [f"Sample {i+1}" for i in range(num_samples_p2)]
    df_phase2 = st.data_editor(pd.DataFrame(0, index=index_p2, columns=["Count (D or c)"]), use_container_width=True, key="p2_editor")

if st.button("Generate Combined Charts & Analysis", type="primary"):
    if not chart_options:
        st.warning("Please select at least 1 chart from the left sidebar.")
    
    # --- PLOTLY RENDER FUNCTION (Handles Phase 1 & 2 Combined) ---
    def render_combined_row(data_p1, data_p2, mean_val, ucl, lcl_calc, title, y_label):
        col1, col2 = st.columns([3, 1])
        
        # LCL for Attribute Charts cannot mathematically fall below 0
        lcl = max(0.0, lcl_calc) 
        
        internal_p1 = data_p1.copy()
        internal_p1.index = [f"P1-{x}" for x in data_p1.index]
        
        internal_p2 = pd.Series(dtype=float)
        if data_p2 is not None and not data_p2.empty:
            internal_p2 = data_p2.copy()
            internal_p2.index = [f"P2-{x}" for x in data_p2.index]
        
        combined_internal = pd.concat([internal_p1, internal_p2])
        valid_combined = combined_internal.dropna()
        
        alarms, violation_indices = check_alarms(valid_combined, mean_val, ucl, lcl)
        
        with col1:
            fig = go.Figure()
            x_vals = list(range(len(valid_combined)))
            display_labels = [str(x).split('-', 1)[1].replace('Sample ', 'S') for x in valid_combined.index]
            
            # Sigma calculated from UCL to prevent distortion when LCL is bounded at 0
            one_sigma = (ucl - mean_val) / 3
            
            # --- ZONES BACKGROUND (Dynamic Bounding with A, B, C Labels) ---
            # Zone C (1 Sigma) - Upper
            fig.add_hrect(y0=mean_val, y1=mean_val + one_sigma, fillcolor="#a8e6cf", opacity=0.3, layer="below", line_width=0, annotation_text="<b>C</b>", annotation_position="right")
            # Zone C (1 Sigma) - Lower
            y_c_low = max(lcl, mean_val - one_sigma)
            if y_c_low < mean_val:
                fig.add_hrect(y0=y_c_low, y1=mean_val, fillcolor="#a8e6cf", opacity=0.3, layer="below", line_width=0, annotation_text="<b>C</b>", annotation_position="right")

            # Zone B (2 Sigma) - Upper
            fig.add_hrect(y0=mean_val + one_sigma, y1=mean_val + 2*one_sigma, fillcolor="#ffd3b6", opacity=0.3, layer="below", line_width=0, annotation_text="<b>B</b>", annotation_position="right")
            # Zone B (2 Sigma) - Lower
            y_b_low = max(lcl, mean_val - 2*one_sigma)
            if y_b_low < y_c_low:
                fig.add_hrect(y0=y_b_low, y1=y_c_low, fillcolor="#ffd3b6", opacity=0.3, layer="below", line_width=0, annotation_text="<b>B</b>", annotation_position="right")

            # Zone A (3 Sigma) - Upper
            fig.add_hrect(y0=mean_val + 2*one_sigma, y1=ucl, fillcolor="#ffaaa5", opacity=0.3, layer="below", line_width=0, annotation_text="<b>A</b>", annotation_position="right")
            # Zone A (3 Sigma) - Lower
            y_a_low = max(lcl, mean_val - 3*one_sigma)
            if y_a_low < y_b_low:
                fig.add_hrect(y0=y_a_low, y1=y_b_low, fillcolor="#ffaaa5", opacity=0.3, layer="below", line_width=0, annotation_text="<b>A</b>", annotation_position="right")

            # --- BOUNDARY LINES ---
            fig.add_hline(y=ucl, line_color="red", line_width=1.5, annotation_text=f"UCL: {ucl:.3f}")
            fig.add_hline(y=mean_val, line_color="blue", line_width=1.5, annotation_text=f"Center: {mean_val:.3f}")
            fig.add_hline(y=lcl, line_color="red", line_width=1.5, annotation_text=f"LCL: {lcl:.3f}")
            
            for multiplier in [-2, -1, 1, 2]:
                line_y = mean_val + multiplier*one_sigma
                if line_y >= lcl:  # Do not draw separator lines if they fall below LCL=0
                    fig.add_hline(y=line_y, line_color="gray", line_width=1, line_dash="dash", opacity=0.4)

            # --- PLOT OBSERVATIONS (Combined connection line) ---
            fig.add_trace(go.Scatter(x=x_vals, y=valid_combined.values, mode='lines', line=dict(color='#888888', width=1.5), showlegend=False, hoverinfo='skip'))
            
            # --- PHASE 1 MARKERS ---
            p1_len = len(internal_p1.dropna())
            fig.add_trace(go.Scatter(x=x_vals[:p1_len], y=internal_p1.dropna().values, mode='markers', marker=dict(color='#1f77b4', size=8, symbol='circle'), name='Phase 1 Data', hovertemplate="Sample: %{x}<br>Value: %{y:.3f}<extra></extra>"))
            
            # --- PHASE 2 MARKERS & TRANSITION LINE ---
            if not internal_p2.empty:
                p2_len = len(internal_p2.dropna())
                fig.add_trace(go.Scatter(x=x_vals[p1_len:], y=internal_p2.dropna().values, mode='markers', marker=dict(color='#ff7f0e', size=9, symbol='square'), name='Phase 2 Data', hovertemplate="Sample: %{x}<br>Value: %{y:.3f}<extra></extra>"))
                fig.add_vline(x=p1_len - 0.5, line_color="black", line_width=2, line_dash="dashdot", annotation_text="Phase Transition", annotation_position="top right")

            # --- HIGHLIGHT ALARM POINTS ---
            if violation_indices:
                valid_violations = [v for v in violation_indices if v in valid_combined.index]
                if valid_violations:
                    v_x = [valid_combined.index.get_loc(v) for v in valid_violations]
                    v_y = valid_combined.loc[valid_violations].values
                    fig.add_trace(go.Scatter(x=v_x, y=v_y, mode='markers', marker=dict(color='red', size=12, line=dict(color='black', width=1.5)), name='Alarm Marker', hoverinfo='skip'))

            # --- FORMAT LAYOUT ---
            fig.update_layout(
                title=dict(text=title, font=dict(size=18, color="black"), x=0.05),
                yaxis_title=y_label,
                xaxis=dict(tickmode='array', tickvals=x_vals, ticktext=display_labels, tickangle=-45),
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40),
                plot_bgcolor="white"
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)', rangemode='tozero')
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### 🚨 Alarm Analysis")
            for a in alarms:
                if "✅" in a: st.success(a)
                else: st.error(a)

    # ==========================================
    # DATA EXTRACTION
    # ==========================================
    counts_p1 = df_phase1["Count (D or c)"].drop(index=excluded_samples)
    
    counts_p2 = pd.Series(dtype=float)
    if use_phase2 and not df_phase2.empty:
        counts_p2 = df_phase2["Count (D or c)"]

    # ==========================================
    # 1. p-CHART LOGIC
    # ==========================================
    if "p-Chart" in chart_options:
        st.divider()
        st.subheader("📉 p-Chart (Fraction Nonconforming)")
        
        p_data_p1 = counts_p1 / sample_size_p1
        p_bar = counts_p1.sum() / (len(counts_p1) * sample_size_p1)
        sigma_p = np.sqrt(p_bar * (1 - p_bar) / sample_size_p1) if p_bar > 0 else 0
        
        ucl_p = p_bar + 3 * sigma_p
        lcl_p = p_bar - 3 * sigma_p
        
        p_data_p2 = counts_p2 / sample_size_p1 if not counts_p2.empty else None
        
        render_combined_row(p_data_p1, p_data_p2, p_bar, ucl_p, lcl_p, "p-Chart", "Fraction (p)")

    # ==========================================
    # 2. np-CHART LOGIC
    # ==========================================
    if "np-Chart" in chart_options:
        st.divider()
        st.subheader("📉 np-Chart (Number of Defective Items)")
        
        np_bar = counts_p1.mean()
        p_bar_est = np_bar / sample_size_p1
        sigma_np = np.sqrt(sample_size_p1 * p_bar_est * (1 - p_bar_est)) if p_bar_est > 0 else 0
        
        ucl_np = np_bar + 3 * sigma_np
        lcl_np = np_bar - 3 * sigma_np
        
        render_combined_row(counts_p1, counts_p2, np_bar, ucl_np, lcl_np, "np-Chart", "Count (np)")

    # ==========================================
    # 3. c-CHART LOGIC
    # ==========================================
    if "c-Chart" in chart_options:
        st.divider()
        st.subheader("📉 c-Chart (Number of Defects)")
        st.info("💡 **Note:** The c-Chart evaluates the total number of defects in a constant-sized inspection unit.")
        
        c_bar = counts_p1.mean()
        sigma_c = np.sqrt(c_bar) if c_bar > 0 else 0
        
        ucl_c = c_bar + 3 * sigma_c
        lcl_c = c_bar - 3 * sigma_c
        
        render_combined_row(counts_p1, counts_p2, c_bar, ucl_c, lcl_c, "c-Chart", "Defects (c)")

    # ==========================================
    # 4. u-CHART LOGIC
    # ==========================================
    if "u-Chart" in chart_options:
        st.divider()
        st.subheader("📉 u-Chart (Defects per Unit)")
        
        u_data_p1 = counts_p1 / sample_size_p1
        u_bar = counts_p1.sum() / (len(counts_p1) * sample_size_p1)
        sigma_u = np.sqrt(u_bar / sample_size_p1) if u_bar > 0 else 0
        
        ucl_u = u_bar + 3 * sigma_u
        lcl_u = u_bar - 3 * sigma_u
        
        u_data_p2 = counts_p2 / sample_size_p1 if not counts_p2.empty else None
        
        render_combined_row(u_data_p1, u_data_p2, u_bar, ucl_u, lcl_u, "u-Chart", "Defects per Unit (u)")