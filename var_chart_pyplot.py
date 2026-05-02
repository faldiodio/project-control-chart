import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Chart Generator", layout="wide")
st.title("📊 Quality Control Chart Generator (Interactive via Plotly)")
st.markdown("Application to generate interactive X-Bar R, X-Bar S, X-Bar MR, and CUSUM Charts with Phase 1 (Base) and Phase 2 (Monitoring) capabilities.")

# --- SIDEBAR (INPUT PARAMETERS) ---
st.sidebar.header("⚙️ Control Chart Parameters")

chart_options = st.sidebar.multiselect(
    "Select Charts to Generate:",
    ["X-Bar R", "X-Bar S", "X-Bar MR", "CUSUM"],
    default=[]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Phase 1 Settings (Base Data)")
num_groups_p1 = st.sidebar.number_input("Phase 1: Number of Groups (k)", min_value=1, value=15, step=1)
sample_size_p1 = st.sidebar.number_input("Phase 1: Sample per Group (n)", min_value=1, value=5, step=1)

if "CUSUM" in chart_options:
    st.sidebar.markdown("---")
    st.sidebar.subheader("CUSUM Parameters")
    cusum_target = st.sidebar.number_input("Target Mean (μ0)", value=10.0, step=0.1)
    cusum_sigma = st.sidebar.number_input("Process StDev (σ)", value=1.0, step=0.1)
    cusum_k = st.sidebar.number_input("Slack Value (k)", value=0.5, step=0.1)
    cusum_h = st.sidebar.number_input("Decision Interval (h)", value=5.0, step=0.5)

# --- CONSTANTS ---
def get_constants(n):
    constants = {
        2:  {'A2': 1.880, 'A3': 2.659, 'd2': 1.128, 'D3': 0,     'D4': 3.267, 'B3': 0,     'B4': 3.267},
        3:  {'A2': 1.023, 'A3': 1.954, 'd2': 1.693, 'D3': 0,     'D4': 2.574, 'B3': 0,     'B4': 2.568},
        4:  {'A2': 0.729, 'A3': 1.628, 'd2': 2.059, 'D3': 0,     'D4': 2.282, 'B3': 0,     'B4': 2.266},
        5:  {'A2': 0.577, 'A3': 1.427, 'd2': 2.326, 'D3': 0,     'D4': 2.114, 'B3': 0,     'B4': 2.089},
        6:  {'A2': 0.483, 'A3': 1.287, 'd2': 2.534, 'D3': 0,     'D4': 2.004, 'B3': 0.030, 'B4': 1.970},
        7:  {'A2': 0.419, 'A3': 1.182, 'd2': 2.704, 'D3': 0.076, 'D4': 1.924, 'B3': 0.118, 'B4': 1.882},
        8:  {'A2': 0.373, 'A3': 1.099, 'd2': 2.847, 'D3': 0.136, 'D4': 1.864, 'B3': 0.185, 'B4': 1.815},
        9:  {'A2': 0.337, 'A3': 1.032, 'd2': 2.970, 'D3': 0.184, 'D4': 1.816, 'B3': 0.239, 'B4': 1.761},
        10: {'A2': 0.308, 'A3': 0.975, 'd2': 3.078, 'D3': 0.223, 'D4': 1.777, 'B3': 0.284, 'B4': 1.716},
        11: {'A2': 0.285, 'A3': 0.927, 'd2': 3.173, 'D3': 0.256, 'D4': 1.744, 'B3': 0.321, 'B4': 1.679},
        12: {'A2': 0.266, 'A3': 0.886, 'd2': 3.258, 'D3': 0.283, 'D4': 1.717, 'B3': 0.354, 'B4': 1.646},
        13: {'A2': 0.249, 'A3': 0.850, 'd2': 3.336, 'D3': 0.307, 'D4': 1.693, 'B3': 0.382, 'B4': 1.618},
        14: {'A2': 0.235, 'A3': 0.817, 'd2': 3.407, 'D3': 0.328, 'D4': 1.672, 'B3': 0.406, 'B4': 1.594},
        15: {'A2': 0.223, 'A3': 0.789, 'd2': 3.472, 'D3': 0.347, 'D4': 1.653, 'B3': 0.428, 'B4': 1.572},
        16: {'A2': 0.212, 'A3': 0.763, 'd2': 3.532, 'D3': 0.363, 'D4': 1.637, 'B3': 0.448, 'B4': 1.552},
        17: {'A2': 0.203, 'A3': 0.739, 'd2': 3.588, 'D3': 0.378, 'D4': 1.622, 'B3': 0.466, 'B4': 1.534},
        18: {'A2': 0.194, 'A3': 0.718, 'd2': 3.640, 'D3': 0.391, 'D4': 1.608, 'B3': 0.482, 'B4': 1.518},
        19: {'A2': 0.187, 'A3': 0.698, 'd2': 3.689, 'D3': 0.403, 'D4': 1.597, 'B3': 0.497, 'B4': 1.503},
        20: {'A2': 0.180, 'A3': 0.680, 'd2': 3.735, 'D3': 0.415, 'D4': 1.585, 'B3': 0.510, 'B4': 1.490},
        21: {'A2': 0.173, 'A3': 0.663, 'd2': 3.778, 'D3': 0.425, 'D4': 1.575, 'B3': 0.523, 'B4': 1.477},
        22: {'A2': 0.167, 'A3': 0.647, 'd2': 3.819, 'D3': 0.434, 'D4': 1.566, 'B3': 0.534, 'B4': 1.466},
        23: {'A2': 0.162, 'A3': 0.633, 'd2': 3.858, 'D3': 0.443, 'D4': 1.557, 'B3': 0.545, 'B4': 1.455},
        24: {'A2': 0.157, 'A3': 0.619, 'd2': 3.895, 'D3': 0.451, 'D4': 1.548, 'B3': 0.555, 'B4': 1.445},
        25: {'A2': 0.153, 'A3': 0.606, 'd2': 3.931, 'D3': 0.459, 'D4': 1.541, 'B3': 0.565, 'B4': 1.435}
    }
    return constants.get(n, None)

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
            disp_idx = str(data.index[i])
            alarms.append(f"**[Type 0]** Point **{disp_idx}** is out of Control Limits (>UCL or <LCL).")
            violation_indices.add(data.index[i])

    sigma = (ucl - mean_val) / 3
    signs = np.sign(data - mean_val)
    
    # 1. 7 points in a row on one side
    for i in range(n - 6):
        if not valid_mask.iloc[i:i+7].all(): continue
        if np.all(signs.iloc[i:i+7] == 1) or np.all(signs.iloc[i:i+7] == -1):
            d_start = str(data.index[i])
            d_end = str(data.index[i+6])
            alarms.append(f"**[Type 1]** **{d_start} to {d_end}**: 7 consecutive points on one side.")
            violation_indices.update(data.index[i:i+7])
            
    # 2. 10 out of 11 points on one side
    for i in range(n - 10):
        if not valid_mask.iloc[i:i+11].all(): continue
        window = signs.iloc[i:i+11]
        if np.sum(window == 1) >= 10 or np.sum(window == -1) >= 10:
            d_start = str(data.index[i])
            d_end = str(data.index[i+10])
            alarms.append(f"**[Type 2]** **{d_start} to {d_end}**: 10 out of 11 points on one side.")
            violation_indices.update(data.index[i:i+11])
            
    # 3. 12 out of 14 points on one side
    for i in range(n - 13):
        if not valid_mask.iloc[i:i+14].all(): continue
        window = signs.iloc[i:i+14]
        if np.sum(window == 1) >= 12 or np.sum(window == -1) >= 12:
            d_start = str(data.index[i])
            d_end = str(data.index[i+13])
            alarms.append(f"**[Type 3]** **{d_start} to {d_end}**: 12 out of 14 points on one side.")
            violation_indices.update(data.index[i:i+14])
            
    # 4. 6 points steadily increasing/decreasing
    for i in range(n - 5):
        if not valid_mask.iloc[i:i+6].all(): continue
        diffs = np.diff(data.iloc[i:i+6])
        if np.all(diffs > 0) or np.all(diffs < 0):
            d_start = str(data.index[i])
            d_end = str(data.index[i+5])
            alarms.append(f"**[Type 4]** **{d_start} to {d_end}**: 6 consecutive points steadily increasing/decreasing.")
            violation_indices.update(data.index[i:i+6])
            
    # 5. 2 out of 3 in Zone A (same side)
    for i in range(n - 2):
        if not valid_mask.iloc[i:i+3].all(): continue
        window = data.iloc[i:i+3]
        if np.sum(window > mean_val + 2*sigma) >= 2 or np.sum(window < mean_val - 2*sigma) >= 2:
            d_start = str(data.index[i])
            d_end = str(data.index[i+2])
            alarms.append(f"**[Type 5]** **{d_start} to {d_end}**: 2 out of 3 points in Zone A.")
            violation_indices.update(data.index[i:i+3])
            
    # 6. 4 out of 5 in Zone B or beyond (same side)
    for i in range(n - 4):
        if not valid_mask.iloc[i:i+5].all(): continue
        window = data.iloc[i:i+5]
        if np.sum(window > mean_val + sigma) >= 4 or np.sum(window < mean_val - sigma) >= 4:
            d_start = str(data.index[i])
            d_end = str(data.index[i+4])
            alarms.append(f"**[Type 6]** **{d_start} to {d_end}**: 4 out of 5 points in Zone B or beyond.")
            violation_indices.update(data.index[i:i+5])

    if not alarms:
        alarms.append("✅ Process is in control.")
        
    return list(dict.fromkeys(alarms)), list(violation_indices)

# --- MAIN GUI ---
st.subheader("📝 Phase 1: Base Observation Data (Limits Calculation)")
columns_p1 = [f"Group {i+1}" for i in range(num_groups_p1)]
index_p1 = [f"Sample {i+1}" for i in range(sample_size_p1)]
df_phase1 = st.data_editor(pd.DataFrame(0.0, index=index_p1, columns=columns_p1), use_container_width=True, key="p1_editor")

st.markdown("### ✂️ Data Cleaning (Phase 1)")
st.info("Select data points/groups to exclude from the Phase 1 base calculations (e.g., removing outliers).")
col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    excluded_samples = st.multiselect("Exclude Phase 1 Samples (For MR & CUSUM Vertical Data):", options=index_p1)
with col_ex2:
    excluded_groups = st.multiselect("Exclude Phase 1 Groups (For X-Bar R & S Horizontal Data):", options=columns_p1)

st.divider()

st.subheader("📝 Phase 2: Monitoring Data (Optional)")
use_phase2 = st.checkbox("Enable Phase 2 (Add new observations)")

is_recalc = False
df_phase2 = pd.DataFrame()

if use_phase2:
    phase2_mode = st.radio(
        "Phase 2 Analysis Mode:",
        options=[
            "📊 1. Monitoring Option (Evaluate Phase 2 data using Phase 1 Limits without altering them)",
            "🔄 2. Recalculate Option (Calculate new UCL, CL, LCL for Phase 2, display joined chart)"
        ]
    )
    is_recalc = "Recalculate Option" in phase2_mode

    col_p2_1, col_p2_2 = st.columns(2)
    with col_p2_1:
        num_groups_p2 = st.number_input("Phase 2: Number of Groups (k)", min_value=1, value=5, step=1)
    with col_p2_2:
        sample_size_p2 = st.number_input("Phase 2: Sample per Group (n)", min_value=1, value=5, step=1)
        
    columns_p2 = [f"Group {i+1}" for i in range(num_groups_p2)]
    index_p2 = [f"Sample {i+1}" for i in range(sample_size_p2)]
    df_phase2 = st.data_editor(pd.DataFrame(0.0, index=index_p2, columns=columns_p2), use_container_width=True, key="p2_editor")


if st.button("Generate Combined Charts & Analysis", type="primary"):
    if not chart_options:
        st.warning("Please select at least 1 chart from the left sidebar.")
    
    # --- PLOTLY RENDER FUNCTION (Handles Phase 1 & 2 Combined + Recalculate Splitting) ---
    def render_combined_row(data_p1, data_p2, mean_val, ucl, lcl, title, y_label, mean_val_p2=None, ucl_p2=None, lcl_p2=None):
        col1, col2 = st.columns([3, 1])
        
        internal_p1 = data_p1.copy()
        internal_p1.index = [f"P1-{x}" for x in data_p1.index]
        
        internal_p2 = pd.Series(dtype=float)
        has_p2 = data_p2 is not None and not data_p2.empty
        if has_p2:
            internal_p2 = data_p2.copy()
            internal_p2.index = [f"P2-{x}" for x in data_p2.index]
        
        combined_internal = pd.concat([internal_p1, internal_p2])
        valid_combined = combined_internal.dropna()
        
        has_split_limits = has_p2 and mean_val_p2 is not None
        
        # Split Evaluation if Recalculating
        if has_split_limits:
            alarms_p1, v_idx_p1 = check_alarms(internal_p1.dropna(), mean_val, ucl, lcl)
            alarms_p2, v_idx_p2 = check_alarms(internal_p2.dropna(), mean_val_p2, ucl_p2, lcl_p2)
            
            alarms = [f"**[Phase 1]** {a}" for a in alarms_p1 if "✅" not in a]
            alarms += [f"**[Phase 2]** {a}" for a in alarms_p2 if "✅" not in a]
            if not alarms: alarms.append("✅ Process is in control across both phases.")
            
            violation_indices = list(v_idx_p1) + list(v_idx_p2)
        else:
            alarms, violation_indices = check_alarms(valid_combined, mean_val, ucl, lcl)
        
        with col1:
            fig = go.Figure()
            
            x_vals = list(range(len(valid_combined)))
            display_labels = [str(x).split('-', 1)[1].replace('Group ', 'G').replace('Sample ', 'S') for x in valid_combined.index]
            total_len = len(valid_combined)
            p1_len = len(internal_p1.dropna())
            
            def add_limit_zones(fig_obj, x_start, x_end, mean_v, ucl_v, lcl_v):
                one_sig = (ucl_v - mean_v) / 3
                
                # Zone Rectangles
                fig_obj.add_shape(type="rect", x0=x_start, x1=x_end, y0=mean_v, y1=mean_v + one_sig, fillcolor="#a8e6cf", opacity=0.3, layer="below", line_width=0)
                fig_obj.add_shape(type="rect", x0=x_start, x1=x_end, y0=mean_v - one_sig, y1=mean_v, fillcolor="#a8e6cf", opacity=0.3, layer="below", line_width=0)
                fig_obj.add_shape(type="rect", x0=x_start, x1=x_end, y0=mean_v + one_sig, y1=mean_v + 2*one_sig, fillcolor="#ffd3b6", opacity=0.3, layer="below", line_width=0)
                fig_obj.add_shape(type="rect", x0=x_start, x1=x_end, y0=mean_v - 2*one_sig, y1=mean_v - one_sig, fillcolor="#ffd3b6", opacity=0.3, layer="below", line_width=0)
                fig_obj.add_shape(type="rect", x0=x_start, x1=x_end, y0=mean_v + 2*one_sig, y1=ucl_v, fillcolor="#ffaaa5", opacity=0.3, layer="below", line_width=0)
                fig_obj.add_shape(type="rect", x0=x_start, x1=x_end, y0=lcl_v, y1=mean_v - 2*one_sig, fillcolor="#ffaaa5", opacity=0.3, layer="below", line_width=0)

                # Boundary Lines
                fig_obj.add_shape(type="line", x0=x_start, x1=x_end, y0=ucl_v, y1=ucl_v, line=dict(color="red", width=1.5))
                fig_obj.add_shape(type="line", x0=x_start, x1=x_end, y0=mean_v, y1=mean_v, line=dict(color="blue", width=1.5))
                fig_obj.add_shape(type="line", x0=x_start, x1=x_end, y0=lcl_v, y1=lcl_v, line=dict(color="red", width=1.5))
                
                # Annotations
                fig_obj.add_annotation(x=x_end, y=ucl_v, text=f"UCL: {ucl_v:.2f}", showarrow=False, xanchor="left", xshift=5)
                fig_obj.add_annotation(x=x_end, y=mean_v, text=f"CL: {mean_v:.2f}", showarrow=False, xanchor="left", xshift=5)
                fig_obj.add_annotation(x=x_end, y=lcl_v, text=f"LCL: {lcl_v:.2f}", showarrow=False, xanchor="left", xshift=5)

            # Draw Zones
            if has_split_limits:
                add_limit_zones(fig, 0, p1_len - 0.5, mean_val, ucl, lcl)
                add_limit_zones(fig, p1_len - 0.5, max(total_len - 1, 0), mean_val_p2, ucl_p2, lcl_p2)
            else:
                add_limit_zones(fig, 0, max(total_len - 1, 0), mean_val, ucl, lcl)

            # --- PLOT OBSERVATIONS (Combined connection line) ---
            fig.add_trace(go.Scatter(x=x_vals, y=valid_combined.values, mode='lines', line=dict(color='#888888', width=1.5), showlegend=False, hoverinfo='skip'))
            
            # --- PHASE 1 MARKERS ---
            fig.add_trace(go.Scatter(x=x_vals[:p1_len], y=internal_p1.dropna().values, mode='markers', marker=dict(color='#1f77b4', size=8, symbol='circle'), name='Phase 1 Data', hovertemplate="Sample: %{x}<br>Value: %{y:.2f}<extra></extra>"))
            
            # --- PHASE 2 MARKERS & TRANSITION LINE ---
            if has_p2:
                p2_len = len(internal_p2.dropna())
                fig.add_trace(go.Scatter(x=x_vals[p1_len:], y=internal_p2.dropna().values, mode='markers', marker=dict(color='#ff7f0e', size=9, symbol='square'), name='Phase 2 Data', hovertemplate="Sample: %{x}<br>Value: %{y:.2f}<extra></extra>"))
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
                margin=dict(l=40, r=60, t=60, b=40),
                plot_bgcolor="white"
            )
            # Add grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### 🚨 Alarm Analysis")
            for a in alarms:
                if "✅" in a: st.success(a)
                else: st.error(a)

    # ==========================================
    # 1. CUSUM CHART LOGIC (Plotly)
    # ==========================================
    if "CUSUM" in chart_options:
        st.divider()
        st.subheader("📈 CUSUM Tabular Analysis")
        st.info("💡 CUSUM uses individual observations. Data is taken vertically from the first column.")
        
        cusum_p1 = df_phase1.iloc[:, 0].drop(index=excluded_samples)
        cusum_p1.index = [f"P1-{x}" for x in cusum_p1.index]
        
        cusum_p2 = pd.Series(dtype=float)
        if use_phase2 and not df_phase2.empty:
            cusum_p2 = df_phase2.iloc[:, 0]
            cusum_p2.index = [f"P2-{x}" for x in cusum_p2.index]
            
        combined_cusum_data = pd.concat([cusum_p1, cusum_p2])
        
        K_p1 = cusum_k * cusum_sigma
        H_p1 = cusum_h * cusum_sigma
        target_p1 = cusum_target
        
        target_p2 = target_p1
        K_p2 = K_p1
        H_p2 = H_p1
        
        if is_recalc and not cusum_p2.empty:
            target_p2 = cusum_p2.mean()
            sigma_p2 = cusum_p2.std(ddof=1)
            if pd.isna(sigma_p2) or sigma_p2 == 0:
                sigma_p2 = cusum_sigma # Fallback
            K_p2 = cusum_k * sigma_p2
            H_p2 = cusum_h * sigma_p2
            
        cusum_steps = []
        cp_prev, cm_prev = 0, 0
        
        for i, xi in enumerate(combined_cusum_data):
            # Apply recalculated target and K if in Phase 2
            current_target = target_p2 if (is_recalc and i >= len(cusum_p1)) else target_p1
            current_K = K_p2 if (is_recalc and i >= len(cusum_p1)) else K_p1
            current_H = H_p2 if (is_recalc and i >= len(cusum_p1)) else H_p1
            
            dev_plus = xi - (current_target + current_K)
            cp = max(0, dev_plus + cp_prev)
            
            dev_minus = (current_target - current_K) - xi
            cm = max(0, dev_minus + cm_prev)
            
            cusum_steps.append({
                "ID": combined_cusum_data.index[i],
                "xi": xi,
                "Ci+": cp,
                "Ci-": cm,
                "H_Active": current_H
            })
            cp_prev, cm_prev = cp, cm
            
        df_cusum = pd.DataFrame(cusum_steps).set_index("ID")
        
        # --- CHART RENDERING (Plotly) ---
        col_chart, col_alarm = st.columns([3, 1])
        with col_chart:
            fig_cusum = go.Figure()
            
            x_vals = list(range(len(df_cusum)))
            display_labels = [str(x).split('-', 1)[1].replace('Sample ', 'S') for x in df_cusum.index]
            p1_len = len(cusum_p1)
            total_len = len(df_cusum)
            
            # Plot connecting lines
            fig_cusum.add_trace(go.Scatter(x=x_vals, y=df_cusum["Ci+"], mode='lines', line=dict(color='#888888', width=1), showlegend=False, hoverinfo='skip'))
            fig_cusum.add_trace(go.Scatter(x=x_vals, y=-df_cusum["Ci-"], mode='lines', line=dict(color='#888888', width=1), showlegend=False, hoverinfo='skip'))
            
            # Phase 1 markers
            fig_cusum.add_trace(go.Scatter(x=x_vals[:p1_len], y=df_cusum.iloc[:p1_len]["Ci+"], mode='markers', marker=dict(color='#1f77b4', size=8, symbol='circle'), name='P1: C+ (Upper)'))
            fig_cusum.add_trace(go.Scatter(x=x_vals[:p1_len], y=-df_cusum.iloc[:p1_len]["Ci-"], mode='markers', marker=dict(color='#a6cee3', size=8, symbol='circle'), name='P1: C- (Lower)'))
            
            # Phase 2 markers
            if use_phase2 and not cusum_p2.empty:
                p2_len = len(cusum_p2)
                fig_cusum.add_trace(go.Scatter(x=x_vals[p1_len:], y=df_cusum.iloc[p1_len:]["Ci+"], mode='markers', marker=dict(color='#ff7f0e', size=9, symbol='square'), name='P2: C+ (Upper)'))
                fig_cusum.add_trace(go.Scatter(x=x_vals[p1_len:], y=-df_cusum.iloc[p1_len:]["Ci-"], mode='markers', marker=dict(color='#ffbb78', size=9, symbol='square'), name='P2: C- (Lower)'))
                fig_cusum.add_vline(x=p1_len - 0.5, line_color="black", line_width=2, line_dash="dashdot", annotation_text="Phase Transition", annotation_position="top right")

            # Decision Interval lines
            if is_recalc and not cusum_p2.empty:
                # Phase 1 Lines
                fig_cusum.add_shape(type="line", x0=0, x1=p1_len-0.5, y0=H_p1, y1=H_p1, line=dict(color="red", width=2, dash="dash"))
                fig_cusum.add_shape(type="line", x0=0, x1=p1_len-0.5, y0=-H_p1, y1=-H_p1, line=dict(color="red", width=2, dash="dash"))
                # Phase 2 Lines
                fig_cusum.add_shape(type="line", x0=p1_len-0.5, x1=max(total_len-1, 0), y0=H_p2, y1=H_p2, line=dict(color="red", width=2, dash="dash"))
                fig_cusum.add_shape(type="line", x0=p1_len-0.5, x1=max(total_len-1, 0), y0=-H_p2, y1=-H_p2, line=dict(color="red", width=2, dash="dash"))
                fig_cusum.add_annotation(x=total_len-1, y=H_p2, text=f"+H (P2) = {H_p2:.2f}", showarrow=False, xanchor="left", xshift=5)
                fig_cusum.add_annotation(x=total_len-1, y=-H_p2, text=f"-H (P2) = {-H_p2:.2f}", showarrow=False, xanchor="left", xshift=5)
            else:
                fig_cusum.add_hline(y=H_p1, line_color="red", line_dash="dash", line_width=2, annotation_text=f"+H = {H_p1:.2f}")
                fig_cusum.add_hline(y=-H_p1, line_color="red", line_dash="dash", line_width=2, annotation_text=f"-H = {-H_p1:.2f}", annotation_position="bottom right")
                
            fig_cusum.add_hline(y=0, line_color="black", line_width=1.5)
            
            cusum_alarms = []
            v_x_plus, v_y_plus = [], []
            v_x_minus, v_y_minus = [], []
            
            for i, idx in enumerate(df_cusum.index):
                disp_idx = str(idx) # Using the full index label like 'P2-Sample 1'
                current_H = df_cusum.loc[idx, "H_Active"]
                if df_cusum.loc[idx, "Ci+"] > current_H: 
                    cusum_alarms.append(f"**[Type C+]** Point **{disp_idx}** (C+) exceeds H ({current_H:.2f}).")
                    v_x_plus.append(i); v_y_plus.append(df_cusum.loc[idx, "Ci+"])
                if df_cusum.loc[idx, "Ci-"] > current_H: 
                    cusum_alarms.append(f"**[Type C-]** Point **{disp_idx}** (C-) exceeds H ({current_H:.2f}).")
                    v_x_minus.append(i); v_y_minus.append(-df_cusum.loc[idx, "Ci-"])

            if v_x_plus:
                fig_cusum.add_trace(go.Scatter(x=v_x_plus, y=v_y_plus, mode='markers', marker=dict(color='red', size=12, line=dict(color='black', width=1.5)), name='Alarm (C+)', hoverinfo='skip'))
            if v_x_minus:
                fig_cusum.add_trace(go.Scatter(x=v_x_minus, y=v_y_minus, mode='markers', marker=dict(color='red', size=12, line=dict(color='black', width=1.5)), name='Alarm (C-)', hoverinfo='skip'))

            fig_cusum.update_layout(
                title=dict(text="Tabular CUSUM Status Chart", font=dict(size=18, color="black"), x=0.05),
                xaxis=dict(tickmode='array', tickvals=x_vals, ticktext=display_labels, tickangle=-45),
                yaxis_title="Cumulative Sum",
                hovermode="x unified",
                margin=dict(l=40, r=80, t=60, b=40),
                plot_bgcolor="white"
            )
            fig_cusum.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
            fig_cusum.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
            
            st.plotly_chart(fig_cusum, use_container_width=True)
            
        with col_alarm:
            st.markdown("#### 🚨 CUSUM Alarm Analysis")
            if not cusum_alarms: st.success("✅ Process is in control.")
            else:
                for a in cusum_alarms: st.error(a)
                
        # --- TABLE RENDERING ---
        st.markdown("#### 📋 Calculation Table (CUSUM)")
        df_cusum_display = df_cusum.drop(columns=["H_Active"]).copy()
        df_cusum_display.index = [str(x).split('-', 1)[1] for x in df_cusum_display.index]
        st.dataframe(df_cusum_display.style.format(precision=3).highlight_max(subset=["Ci+", "Ci-"], color="#ffebcc"), use_container_width=True)

    # ==========================================
    # 2. X-BAR MR CHART LOGIC
    # ==========================================
    if "X-Bar MR" in chart_options:
        st.divider()
        st.subheader("📊 X (Individual) & MR Chart")
        
        ind_p1_full = df_phase1.iloc[:, 0].drop(index=excluded_samples)
        mr_p1_full = ind_p1_full.diff().abs()
        
        x_bar_limit = ind_p1_full.mean()
        mr_bar_limit = mr_p1_full.mean()
        
        c_mr = get_constants(2)
        d2, D4, D3 = c_mr['d2'], c_mr['D4'], c_mr['D3']
        
        ucl_i = x_bar_limit + 3 * (mr_bar_limit / d2)
        lcl_i = x_bar_limit - 3 * (mr_bar_limit / d2)
        ucl_mr = D4 * mr_bar_limit
        lcl_mr = D3 * mr_bar_limit
        
        ind_p2 = df_phase2.iloc[:, 0] if use_phase2 and not df_phase2.empty else None
        mr_p2 = ind_p2.diff().abs() if ind_p2 is not None else None
        
        # Calculate Phase 2 Limits if Recalculate Mode
        ucl_i_p2, lcl_i_p2, x_bar_limit_p2 = None, None, None
        ucl_mr_p2, lcl_mr_p2, mr_bar_limit_p2 = None, None, None
        
        if is_recalc and ind_p2 is not None:
            x_bar_limit_p2 = ind_p2.mean()
            mr_bar_limit_p2 = mr_p2.mean()
            ucl_i_p2 = x_bar_limit_p2 + 3 * (mr_bar_limit_p2 / d2)
            lcl_i_p2 = x_bar_limit_p2 - 3 * (mr_bar_limit_p2 / d2)
            ucl_mr_p2 = D4 * mr_bar_limit_p2
            lcl_mr_p2 = D3 * mr_bar_limit_p2

        render_combined_row(ind_p1_full, ind_p2, x_bar_limit, ucl_i, lcl_i, "Individuals (X) Chart", "Measurement", x_bar_limit_p2, ucl_i_p2, lcl_i_p2)
        render_combined_row(mr_p1_full, mr_p2, mr_bar_limit, ucl_mr, lcl_mr, "Moving Range (MR) Chart", "Moving Range", mr_bar_limit_p2, ucl_mr_p2, lcl_mr_p2)

    # ==========================================
    # 3. X-BAR R CHART LOGIC
    # ==========================================
    if "X-Bar R" in chart_options:
        if sample_size_p1 > 1:
            st.divider()
            st.subheader("📉 X-Bar & R Chart (Group Data)")
            
            df_p1_clean = df_phase1.drop(columns=excluded_groups)
            if df_p1_clean.shape[1] == 0:
                st.error("All Phase 1 groups excluded! Cannot calculate limits.")
            else:
                xbar_p1 = df_p1_clean.mean(axis=0)
                r_p1 = df_p1_clean.max(axis=0) - df_p1_clean.min(axis=0)
                
                r_bar_limit = r_p1.mean()
                x_dbl_bar_limit = xbar_p1.mean()
                c = get_constants(sample_size_p1)
                
                ucl_x = x_dbl_bar_limit + (c['A2'] * r_bar_limit)
                lcl_x = x_dbl_bar_limit - (c['A2'] * r_bar_limit)
                ucl_r = c['D4'] * r_bar_limit
                lcl_r = c['D3'] * r_bar_limit
                
                xbar_p2 = df_phase2.mean(axis=0) if use_phase2 and not df_phase2.empty else None
                r_p2 = (df_phase2.max(axis=0) - df_phase2.min(axis=0)) if use_phase2 and not df_phase2.empty else None
                
                ucl_x_p2, lcl_x_p2, x_dbl_bar_limit_p2 = None, None, None
                ucl_r_p2, lcl_r_p2, r_bar_limit_p2 = None, None, None
                
                if is_recalc and xbar_p2 is not None:
                    c2 = get_constants(sample_size_p2) # Dynamic constant in case Phase 2 sample size differs
                    r_bar_limit_p2 = r_p2.mean()
                    x_dbl_bar_limit_p2 = xbar_p2.mean()
                    ucl_x_p2 = x_dbl_bar_limit_p2 + (c2['A2'] * r_bar_limit_p2)
                    lcl_x_p2 = x_dbl_bar_limit_p2 - (c2['A2'] * r_bar_limit_p2)
                    ucl_r_p2 = c2['D4'] * r_bar_limit_p2
                    lcl_r_p2 = c2['D3'] * r_bar_limit_p2
                
                render_combined_row(xbar_p1, xbar_p2, x_dbl_bar_limit, ucl_x, lcl_x, "X-Bar Chart", "Mean", x_dbl_bar_limit_p2, ucl_x_p2, lcl_x_p2)
                render_combined_row(r_p1, r_p2, r_bar_limit, ucl_r, lcl_r, "R Chart", "Range", r_bar_limit_p2, ucl_r_p2, lcl_r_p2)
        else:
            st.warning("⚠️ X-Bar R Chart requires Sample Size (n) > 1.")

    # ==========================================
    # 4. X-BAR S CHART LOGIC
    # ==========================================
    if "X-Bar S" in chart_options:
        if sample_size_p1 > 1:
            st.divider()
            st.subheader("📈 X-Bar & S Chart (Group Data)")
            
            df_p1_clean = df_phase1.drop(columns=excluded_groups)
            if df_p1_clean.shape[1] == 0:
                st.error("All Phase 1 groups excluded! Cannot calculate limits.")
            else:
                xbar_p1 = df_p1_clean.mean(axis=0)
                s_p1 = df_p1_clean.std(axis=0, ddof=1)
                
                s_bar_limit = s_p1.mean()
                x_dbl_bar_limit = xbar_p1.mean()
                c = get_constants(sample_size_p1)
                
                ucl_x_s = x_dbl_bar_limit + (c['A3'] * s_bar_limit)
                lcl_x_s = x_dbl_bar_limit - (c['A3'] * s_bar_limit)
                ucl_s = c['B4'] * s_bar_limit
                lcl_s = c['B3'] * s_bar_limit
                
                xbar_p2 = df_phase2.mean(axis=0) if use_phase2 and not df_phase2.empty else None
                s_p2 = df_phase2.std(axis=0, ddof=1) if use_phase2 and not df_phase2.empty else None
                
                ucl_x_s_p2, lcl_x_s_p2, x_dbl_bar_limit_s_p2 = None, None, None
                ucl_s_p2, lcl_s_p2, s_bar_limit_p2 = None, None, None
                
                if is_recalc and xbar_p2 is not None:
                    c2 = get_constants(sample_size_p2)
                    s_bar_limit_p2 = s_p2.mean()
                    x_dbl_bar_limit_s_p2 = xbar_p2.mean()
                    ucl_x_s_p2 = x_dbl_bar_limit_s_p2 + (c2['A3'] * s_bar_limit_p2)
                    lcl_x_s_p2 = x_dbl_bar_limit_s_p2 - (c2['A3'] * s_bar_limit_p2)
                    ucl_s_p2 = c2['B4'] * s_bar_limit_p2
                    lcl_s_p2 = c2['B3'] * s_bar_limit_p2
                
                render_combined_row(xbar_p1, xbar_p2, x_dbl_bar_limit, ucl_x_s, lcl_x_s, "X-Bar Chart", "Mean", x_dbl_bar_limit_s_p2, ucl_x_s_p2, lcl_x_s_p2)
                render_combined_row(s_p1, s_p2, s_bar_limit, ucl_s, lcl_s, "S Chart", "Standard Deviation", s_bar_limit_p2, ucl_s_p2, lcl_s_p2)
        else:
            st.warning("⚠️ X-Bar S Chart requires Sample Size (n) > 1.")