import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QC Chart Generator", layout="wide")
st.title("📊 Quality Control Chart Generator (Interactive via Plotly)")
st.markdown("Application to generate interactive X-Bar R, X-Bar S, X-Bar MR, and CUSUM Charts with Dynamic Multi-Phase capabilities and Data Tables.")

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

# --- ALARM LOGIC (Full Western Electric Rules upgraded for Dynamic Limits) ---
def check_alarms(data, mean_val, ucl, lcl):
    alarms = []
    violation_indices = set()
    n = len(data)
    valid_mask = ~data.isna()
    
    if not isinstance(ucl, pd.Series): ucl = pd.Series([ucl]*n, index=data.index)
    if not isinstance(lcl, pd.Series): lcl = pd.Series([lcl]*n, index=data.index)
    if not isinstance(mean_val, pd.Series): mean_val = pd.Series([mean_val]*n, index=data.index)
    
    sigma = (ucl - mean_val) / 3
    signs = np.sign(data - mean_val)
    
    # 0. Out of bounds (>UCL or <LCL)
    for i in range(n):
        if not valid_mask.iloc[i]: continue
        if data.iloc[i] > ucl.iloc[i] or data.iloc[i] < lcl.iloc[i]:
            disp_idx = str(data.index[i]).split('-', 1)[-1] if '-' in str(data.index[i]) else data.index[i]
            alarms.append(f"**[Type 0]** Point **{disp_idx}** is out of Control Limits.")
            violation_indices.add(data.index[i])

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
        mean_w = mean_val.iloc[i:i+3]
        sig_w = sigma.iloc[i:i+3]
        if np.sum(window > (mean_w + 2*sig_w)) >= 2 or np.sum(window < (mean_w - 2*sig_w)) >= 2:
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+2]).split('-', 1)[-1]
            alarms.append(f"**[Type 5]** **{d_start} to {d_end}**: 2 out of 3 points in Zone A.")
            violation_indices.update(data.index[i:i+3])
            
    # 6. 4 out of 5 in Zone B or beyond (same side)
    for i in range(n - 4):
        if not valid_mask.iloc[i:i+5].all(): continue
        window = data.iloc[i:i+5]
        mean_w = mean_val.iloc[i:i+5]
        sig_w = sigma.iloc[i:i+5]
        if np.sum(window > (mean_w + sig_w)) >= 4 or np.sum(window < (mean_w - sig_w)) >= 4:
            d_start = str(data.index[i]).split('-', 1)[-1]
            d_end = str(data.index[i+4]).split('-', 1)[-1]
            alarms.append(f"**[Type 6]** **{d_start} to {d_end}**: 4 out of 5 points in Zone B.")
            violation_indices.update(data.index[i:i+5])

    if not alarms:
        alarms.append("✅ Process is in control.")
        
    return list(dict.fromkeys(alarms)), list(violation_indices)

# --- DYNAMIC MULTI-PHASE CHART RENDERER ---
def render_dynamic_variable_chart(configs, title, y_label):
    col1, col2 = st.columns([3, 1])
    
    all_alarms = []
    all_violations = []
    
    combined_data = pd.Series(dtype=float)
    combined_ucl = pd.Series(dtype=float)
    combined_cl = pd.Series(dtype=float)
    combined_lcl = pd.Series(dtype=float)
    
    phase_transitions = []
    
    for cfg in configs:
        data = cfg['data'].copy()
        idx_names = [f"{cfg['name']}-{x}" for x in data.index]
        data.index = idx_names
        
        ucl = cfg['ucl'].copy() if isinstance(cfg['ucl'], pd.Series) else pd.Series([cfg['ucl']]*len(data))
        cl = cfg['cl'].copy() if isinstance(cfg['cl'], pd.Series) else pd.Series([cfg['cl']]*len(data))
        lcl = cfg['lcl'].copy() if isinstance(cfg['lcl'], pd.Series) else pd.Series([cfg['lcl']]*len(data))
        
        ucl.index = idx_names
        cl.index = idx_names
        lcl.index = idx_names
        
        alarms, v_idx = check_alarms(data, cl, ucl, lcl)
        all_alarms += [f"**[{cfg['name']}]** {a}" for a in alarms if "✅" not in a]
        all_violations.extend(list(v_idx))
        
        if not data.empty:
            phase_transitions.append((cfg['name'], len(combined_data)))
            
        combined_data = pd.concat([combined_data, data])
        combined_ucl = pd.concat([combined_ucl, ucl])
        combined_cl = pd.concat([combined_cl, cl])
        combined_lcl = pd.concat([combined_lcl, lcl])
        
    if not all_alarms:
        all_alarms.append("✅ Process is in control across all phases.")
        
    valid_combined = combined_data.dropna()
    
    with col1:
        fig = go.Figure()
        x_vals = list(range(len(valid_combined)))
        # Format label cleanups for the x-axis
        display_labels = [str(x).split('-', 1)[1].replace('Group ', 'G').replace('Sample ', 'S') for x in valid_combined.index]
        
        # Calculate dynamic sigmas for Zone calculations
        sig_dyn = (combined_ucl - combined_cl) / 3
        
        uc = combined_cl + sig_dyn
        ub = combined_cl + 2 * sig_dyn
        ua = combined_ucl
        
        lc = combined_cl - sig_dyn
        lb = combined_cl - 2 * sig_dyn
        la = combined_lcl

        # ZONES BACKGROUND (DYNAMIC STEP FILLS)
        fig.add_trace(go.Scatter(x=x_vals, y=combined_cl.values, mode='lines', line=dict(width=0, shape='hvh'), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=uc.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(168, 230, 207, 0.3)', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=ub.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 211, 182, 0.3)', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=ua.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 170, 165, 0.3)', showlegend=False, hoverinfo='skip'))

        fig.add_trace(go.Scatter(x=x_vals, y=combined_cl.values, mode='lines', line=dict(width=0, shape='hvh'), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=lc.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(168, 230, 207, 0.3)', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=lb.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 211, 182, 0.3)', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=la.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 170, 165, 0.3)', showlegend=False, hoverinfo='skip'))

        # BOUNDARY LINES
        fig.add_trace(go.Scatter(x=x_vals, y=combined_ucl.values, mode='lines', line=dict(color='red', width=1.5, shape='hvh', dash='dash'), name='UCL', hoverinfo='y'))
        fig.add_trace(go.Scatter(x=x_vals, y=combined_cl.values, mode='lines', line=dict(color='blue', width=1.5, shape='hvh'), name='Center Line', hoverinfo='y'))
        fig.add_trace(go.Scatter(x=x_vals, y=combined_lcl.values, mode='lines', line=dict(color='red', width=1.5, shape='hvh', dash='dash'), name='LCL', hoverinfo='y'))

        # Zone separators
        for bound in [uc, ub, lc, lb]:
            fig.add_trace(go.Scatter(x=x_vals, y=bound.values, mode='lines', line=dict(color='gray', width=1, shape='hvh', dash='dash'), opacity=0.3, showlegend=False, hoverinfo='skip'))

        if len(valid_combined) > 0:
            last_x = x_vals[-1]
            fig.add_annotation(x=last_x, y=combined_ucl.iloc[-1], text=f"UCL: {combined_ucl.iloc[-1]:.2f}", showarrow=False, xanchor="left", xshift=10)
            fig.add_annotation(x=last_x, y=combined_cl.iloc[-1], text=f"CL: {combined_cl.iloc[-1]:.2f}", showarrow=False, xanchor="left", xshift=10)
            fig.add_annotation(x=last_x, y=combined_lcl.iloc[-1], text=f"LCL: {combined_lcl.iloc[-1]:.2f}", showarrow=False, xanchor="left", xshift=10)

        # Phase transition vertical lines
        for i, (name, start_idx) in enumerate(phase_transitions):
            if i > 0:
                fig.add_vline(x=start_idx - 0.5, line_color="black", line_width=2, line_dash="dashdot", annotation_text=name, annotation_position="top right")

        # Plot Data Line connection
        fig.add_trace(go.Scatter(x=x_vals, y=valid_combined.values, mode='lines', line=dict(color='#888888', width=1.5), showlegend=False, hoverinfo='skip'))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
        
        for i, cfg in enumerate(configs):
            data = cfg['data'].dropna()
            if data.empty: continue
            
            start_idx = phase_transitions[i][1]
            end_idx = start_idx + len(data)
            
            fig.add_trace(go.Scatter(
                x=list(range(start_idx, end_idx)),
                y=data.values,
                mode='markers',
                marker=dict(color=colors[i % len(colors)], size=8, symbol=symbols[i % len(symbols)]),
                name=f"{cfg['name']} Data",
                hovertemplate="%{x}<br>Value: %{y:.2f}<extra></extra>"
            ))

        # Highlight Alarm points
        if all_violations:
            valid_v = [v for v in all_violations if v in valid_combined.index]
            if valid_v:
                v_x = [valid_combined.index.get_loc(v) for v in valid_v]
                v_y = valid_combined.loc[valid_v].values
                fig.add_trace(go.Scatter(x=v_x, y=v_y, mode='markers', marker=dict(color='red', size=12, line=dict(color='black', width=1.5)), name='Alarm Marker', hoverinfo='skip'))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="black"), x=0.05),
            yaxis_title=y_label,
            xaxis=dict(tickmode='array', tickvals=x_vals, ticktext=display_labels, tickangle=-45),
            hovermode="x unified",
            margin=dict(l=40, r=80, t=60, b=40),
            plot_bgcolor="white"
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("#### 🚨 Alarm Analysis")
        for a in all_alarms:
            if "✅" in a: st.success(a)
            else: st.error(a)


# --- MAIN GUI: DATA INPUT ---
st.subheader("📝 Phase 1: Base Observation Data")
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

# --- UI: DYNAMIC ADDITIONAL PHASES ---
st.subheader("🔄 Additional Phases (Monitoring / Recalculate)")
num_additional_phases = st.number_input("How many additional phases?", min_value=0, max_value=10, value=0, step=1)

additional_phases_data = []

for i in range(num_additional_phases):
    phase_num = i + 2
    st.markdown(f"#### 📝 Phase {phase_num}")
    
    col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
    with col_m1:
        mode = st.radio(
            f"Phase {phase_num} Analysis Mode:",
            options=[
                f"📊 Monitoring Option (Evaluate using currently active limits)",
                f"🔄 Recalculate Option (Calculate fresh limits based on Phase {phase_num} data)"
            ],
            key=f"mode_{phase_num}"
        )
    with col_m2:
        num_groups_pn = st.number_input(f"Number of Groups (k) - Ph {phase_num}", min_value=1, value=5, step=1, key=f"k_{phase_num}")
    with col_m3:
        sample_size_pn = st.number_input(f"Sample per Group (n) - Ph {phase_num}", min_value=1, value=sample_size_p1, step=1, key=f"n_{phase_num}")
        
    columns_pn = [f"Group {j+1}" for j in range(num_groups_pn)]
    index_pn = [f"Sample {j+1}" for j in range(sample_size_pn)]
    
    df_pn = st.data_editor(pd.DataFrame(0.0, index=index_pn, columns=columns_pn), use_container_width=True, key=f"editor_{phase_num}")
    
    additional_phases_data.append({
        'phase_num': phase_num,
        'mode': mode,
        'data': df_pn,
        'n': sample_size_pn
    })

# --- HELPER FUNCTION TO DISPLAY TABLE ---
def display_calculation_table(df_summary, chart_name):
    st.markdown(f"#### 📋 Step-by-step Calculation Table ({chart_name})")
    st.dataframe(df_summary.style.format(precision=3), use_container_width=True)

# --- CHART GENERATOR EXECUTION ---
if st.button("Generate Combined Charts & Analysis", type="primary"):
    if not chart_options:
        st.warning("Please select at least 1 chart from the left sidebar.")

    # ==========================================
    # 1. CUSUM CHART LOGIC
    # ==========================================
    if "CUSUM" in chart_options:
        st.divider()
        st.subheader("📈 CUSUM Tabular Analysis")
        st.info("💡 CUSUM uses individual observations. Data is taken vertically from the first column.")
        
        cusum_p1 = df_phase1.iloc[:, 0].drop(index=excluded_samples)
        cusum_p1.index = [f"P1-{x}" for x in cusum_p1.index]
        
        K_p1 = cusum_k * cusum_sigma
        H_p1 = cusum_h * cusum_sigma
        
        all_cusum_data = [cusum_p1]
        phase_labels = [('Phase 1', len(cusum_p1), cusum_target, K_p1, H_p1)]
        
        current_target = cusum_target
        current_K = K_p1
        current_H = H_p1
        
        for p_dict in additional_phases_data:
            c_data = p_dict['data'].iloc[:, 0].copy()
            c_data.index = [f"P{p_dict['phase_num']}-{x}" for x in c_data.index]
            
            if "Recalculate Option" in p_dict['mode'] and not c_data.empty:
                current_target = c_data.mean()
                sigma_p2 = c_data.std(ddof=1)
                if pd.isna(sigma_p2) or sigma_p2 == 0: sigma_p2 = cusum_sigma 
                current_K = cusum_k * sigma_p2
                current_H = cusum_h * sigma_p2
                
            all_cusum_data.append(c_data)
            phase_labels.append((f"Phase {p_dict['phase_num']}", len(c_data), current_target, current_K, current_H))
            
        combined_cusum_data = pd.concat(all_cusum_data)
        
        cusum_steps = []
        cp_prev, cm_prev = 0, 0
        
        idx_counter = 0
        phase_idx = 0
        current_phase_limit = phase_labels[0][1]
        
        for i, xi in enumerate(combined_cusum_data):
            if i >= current_phase_limit and phase_idx < len(phase_labels) - 1:
                phase_idx += 1
                current_phase_limit += phase_labels[phase_idx][1]
                
            _, _, targ, K_val, H_val = phase_labels[phase_idx]
            
            dev_plus = xi - (targ + K_val)
            cp = max(0, dev_plus + cp_prev)
            
            dev_minus = (targ - K_val) - xi
            cm = max(0, dev_minus + cm_prev)
            
            cusum_steps.append({
                "Phase": phase_labels[phase_idx][0],
                "ID": combined_cusum_data.index[i],
                "xi": xi,
                "Ci+": cp,
                "Ci-": cm,
                "H_Active": H_val
            })
            cp_prev, cm_prev = cp, cm
            
        df_cusum = pd.DataFrame(cusum_steps).set_index("ID")
        
        # --- CHART RENDERING (Plotly) ---
        col_chart, col_alarm = st.columns([3, 1])
        with col_chart:
            fig_cusum = go.Figure()
            x_vals = list(range(len(df_cusum)))
            display_labels = [str(x).split('-', 1)[1].replace('Sample ', 'S') for x in df_cusum.index]
            
            fig_cusum.add_trace(go.Scatter(x=x_vals, y=df_cusum["Ci+"], mode='lines', line=dict(color='#888888', width=1), showlegend=False, hoverinfo='skip'))
            fig_cusum.add_trace(go.Scatter(x=x_vals, y=-df_cusum["Ci-"], mode='lines', line=dict(color='#888888', width=1), showlegend=False, hoverinfo='skip'))
            
            # Plot dynamic decision intervals (H)
            h_line = df_cusum["H_Active"].values
            fig_cusum.add_trace(go.Scatter(x=x_vals, y=h_line, mode='lines', line=dict(color='red', width=2, dash='dash', shape='hvh'), name='+H'))
            fig_cusum.add_trace(go.Scatter(x=x_vals, y=-h_line, mode='lines', line=dict(color='red', width=2, dash='dash', shape='hvh'), name='-H'))
            fig_cusum.add_hline(y=0, line_color="black", line_width=1.5)
            
            cusum_alarms = []
            v_x_plus, v_y_plus = [], []
            v_x_minus, v_y_minus = [], []
            
            for i, idx in enumerate(df_cusum.index):
                disp_idx = str(idx)
                current_H = df_cusum.loc[idx, "H_Active"]
                if df_cusum.loc[idx, "Ci+"] > current_H: 
                    cusum_alarms.append(f"**[Type C+]** Point **{disp_idx}** (C+) exceeds H ({current_H:.2f}).")
                    v_x_plus.append(i); v_y_plus.append(df_cusum.loc[idx, "Ci+"])
                if df_cusum.loc[idx, "Ci-"] > current_H: 
                    cusum_alarms.append(f"**[Type C-]** Point **{disp_idx}** (C-) exceeds H ({current_H:.2f}).")
                    v_x_minus.append(i); v_y_minus.append(-df_cusum.loc[idx, "Ci-"])

            # Plot points
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
            
            start_i = 0
            for p_idx, (name, length, _, _, _) in enumerate(phase_labels):
                if length == 0: continue
                sub_df = df_cusum.iloc[start_i:start_i+length]
                fig_cusum.add_trace(go.Scatter(x=list(range(start_i, start_i+length)), y=sub_df["Ci+"], mode='markers', marker=dict(color=colors[p_idx%len(colors)], size=8, symbol=symbols[p_idx%len(symbols)]), name=f'{name}: C+'))
                fig_cusum.add_trace(go.Scatter(x=list(range(start_i, start_i+length)), y=-sub_df["Ci-"], mode='markers', marker=dict(color=colors[p_idx%len(colors)], size=8, symbol=symbols[p_idx%len(symbols)], opacity=0.5), name=f'{name}: C-'))
                if p_idx > 0:
                    fig_cusum.add_vline(x=start_i - 0.5, line_color="black", line_width=2, line_dash="dashdot", annotation_text=name, annotation_position="top right")
                start_i += length

            if v_x_plus: fig_cusum.add_trace(go.Scatter(x=v_x_plus, y=v_y_plus, mode='markers', marker=dict(color='red', size=12, line=dict(color='black', width=1.5)), name='Alarm (C+)', hoverinfo='skip'))
            if v_x_minus: fig_cusum.add_trace(go.Scatter(x=v_x_minus, y=v_y_minus, mode='markers', marker=dict(color='red', size=12, line=dict(color='black', width=1.5)), name='Alarm (C-)', hoverinfo='skip'))

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
        df_cusum_disp = df_cusum.copy()
        df_cusum_disp.index = [str(x).split('-', 1)[1] for x in df_cusum_disp.index]
        st.dataframe(df_cusum_disp.style.format(precision=3).highlight_max(subset=["Ci+", "Ci-"], color="#ffebcc"), use_container_width=True)

    # ==========================================
    # 2. X-BAR MR CHART LOGIC
    # ==========================================
    if "X-Bar MR" in chart_options:
        st.divider()
        st.subheader("📊 X (Individual) & MR Chart")
        
        configs_x = []
        configs_mr = []
        df_tables = []
        
        ind_p1 = df_phase1.iloc[:, 0].drop(index=excluded_samples)
        mr_p1 = ind_p1.diff().abs()
        
        x_bar_1 = ind_p1.mean()
        mr_bar_1 = mr_p1.mean()
        
        c_mr = get_constants(2)
        d2, D4, D3 = c_mr['d2'], c_mr['D4'], c_mr['D3']
        
        ucl_i_1 = x_bar_1 + 3 * (mr_bar_1 / d2)
        lcl_i_1 = x_bar_1 - 3 * (mr_bar_1 / d2)
        ucl_mr_1 = D4 * mr_bar_1
        lcl_mr_1 = D3 * mr_bar_1
        
        configs_x.append({'name': 'Phase 1', 'data': ind_p1, 'cl': pd.Series(x_bar_1, index=ind_p1.index), 'ucl': pd.Series(ucl_i_1, index=ind_p1.index), 'lcl': pd.Series(lcl_i_1, index=ind_p1.index)})
        configs_mr.append({'name': 'Phase 1', 'data': mr_p1, 'cl': pd.Series(mr_bar_1, index=mr_p1.index), 'ucl': pd.Series(ucl_mr_1, index=mr_p1.index), 'lcl': pd.Series(lcl_mr_1, index=mr_p1.index)})
        
        df_p1 = pd.DataFrame({"Phase": "Phase 1", "Measurement (X)": ind_p1, "Moving Range (MR)": mr_p1, "CL (X)": x_bar_1, "UCL (X)": ucl_i_1, "LCL (X)": lcl_i_1, "CL (MR)": mr_bar_1, "UCL (MR)": ucl_mr_1, "LCL (MR)": lcl_mr_1})
        df_tables.append(df_p1)
        
        curr_x_cl, curr_x_ucl, curr_x_lcl = x_bar_1, ucl_i_1, lcl_i_1
        curr_mr_cl, curr_mr_ucl, curr_mr_lcl = mr_bar_1, ucl_mr_1, lcl_mr_1
        
        for p_dict in additional_phases_data:
            ind_n = p_dict['data'].iloc[:, 0]
            mr_n = ind_n.diff().abs()
            
            if "Recalculate Option" in p_dict['mode'] and not ind_n.empty:
                curr_x_cl = ind_n.mean()
                curr_mr_cl = mr_n.mean()
                curr_x_ucl = curr_x_cl + 3 * (curr_mr_cl / d2)
                curr_x_lcl = curr_x_cl - 3 * (curr_mr_cl / d2)
                curr_mr_ucl = D4 * curr_mr_cl
                curr_mr_lcl = D3 * curr_mr_cl
                
            configs_x.append({'name': f"Phase {p_dict['phase_num']}", 'data': ind_n, 'cl': pd.Series(curr_x_cl, index=ind_n.index), 'ucl': pd.Series(curr_x_ucl, index=ind_n.index), 'lcl': pd.Series(curr_x_lcl, index=ind_n.index)})
            configs_mr.append({'name': f"Phase {p_dict['phase_num']}", 'data': mr_n, 'cl': pd.Series(curr_mr_cl, index=mr_n.index), 'ucl': pd.Series(curr_mr_ucl, index=mr_n.index), 'lcl': pd.Series(curr_mr_lcl, index=mr_n.index)})
            
            df_pn = pd.DataFrame({"Phase": f"Phase {p_dict['phase_num']}", "Measurement (X)": ind_n, "Moving Range (MR)": mr_n, "CL (X)": curr_x_cl, "UCL (X)": curr_x_ucl, "LCL (X)": curr_x_lcl, "CL (MR)": curr_mr_cl, "UCL (MR)": curr_mr_ucl, "LCL (MR)": curr_mr_lcl})
            df_tables.append(df_pn)

        display_calculation_table(pd.concat(df_tables), "X-Bar MR Chart")
        render_dynamic_variable_chart(configs_x, "Individuals (X) Chart", "Measurement")
        render_dynamic_variable_chart(configs_mr, "Moving Range (MR) Chart", "Moving Range")

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
                configs_x = []
                configs_r = []
                df_tables = []
                
                xbar_1 = df_p1_clean.mean(axis=0)
                r_1 = df_p1_clean.max(axis=0) - df_p1_clean.min(axis=0)
                
                r_bar_1 = r_1.mean()
                x_dbl_bar_1 = xbar_1.mean()
                c = get_constants(sample_size_p1)
                
                ucl_x_1 = x_dbl_bar_1 + (c['A2'] * r_bar_1)
                lcl_x_1 = x_dbl_bar_1 - (c['A2'] * r_bar_1)
                ucl_r_1 = c['D4'] * r_bar_1
                lcl_r_1 = c['D3'] * r_bar_1
                
                configs_x.append({'name': 'Phase 1', 'data': xbar_1, 'cl': pd.Series(x_dbl_bar_1, index=xbar_1.index), 'ucl': pd.Series(ucl_x_1, index=xbar_1.index), 'lcl': pd.Series(lcl_x_1, index=xbar_1.index)})
                configs_r.append({'name': 'Phase 1', 'data': r_1, 'cl': pd.Series(r_bar_1, index=r_1.index), 'ucl': pd.Series(ucl_r_1, index=r_1.index), 'lcl': pd.Series(lcl_r_1, index=r_1.index)})
                
                df_p1 = pd.DataFrame({"Phase": "Phase 1", "Mean (X-Bar)": xbar_1, "Range (R)": r_1, "CL (X)": x_dbl_bar_1, "UCL (X)": ucl_x_1, "LCL (X)": lcl_x_1, "CL (R)": r_bar_1, "UCL (R)": ucl_r_1, "LCL (R)": lcl_r_1})
                df_tables.append(df_p1)
                
                curr_x_cl, curr_x_ucl, curr_x_lcl = x_dbl_bar_1, ucl_x_1, lcl_x_1
                curr_r_cl, curr_r_ucl, curr_r_lcl = r_bar_1, ucl_r_1, lcl_r_1
                
                for p_dict in additional_phases_data:
                    xbar_n = p_dict['data'].mean(axis=0)
                    r_n = p_dict['data'].max(axis=0) - p_dict['data'].min(axis=0)
                    
                    if "Recalculate Option" in p_dict['mode'] and not p_dict['data'].empty:
                        c_n = get_constants(p_dict['n'])
                        curr_x_cl = xbar_n.mean()
                        curr_r_cl = r_n.mean()
                        curr_x_ucl = curr_x_cl + (c_n['A2'] * curr_r_cl)
                        curr_x_lcl = curr_x_cl - (c_n['A2'] * curr_r_cl)
                        curr_r_ucl = c_n['D4'] * curr_r_cl
                        curr_r_lcl = c_n['D3'] * curr_r_cl
                        
                    configs_x.append({'name': f"Phase {p_dict['phase_num']}", 'data': xbar_n, 'cl': pd.Series(curr_x_cl, index=xbar_n.index), 'ucl': pd.Series(curr_x_ucl, index=xbar_n.index), 'lcl': pd.Series(curr_x_lcl, index=xbar_n.index)})
                    configs_r.append({'name': f"Phase {p_dict['phase_num']}", 'data': r_n, 'cl': pd.Series(curr_r_cl, index=r_n.index), 'ucl': pd.Series(curr_r_ucl, index=r_n.index), 'lcl': pd.Series(curr_r_lcl, index=r_n.index)})
                    
                    df_pn = pd.DataFrame({"Phase": f"Phase {p_dict['phase_num']}", "Mean (X-Bar)": xbar_n, "Range (R)": r_n, "CL (X)": curr_x_cl, "UCL (X)": curr_x_ucl, "LCL (X)": curr_x_lcl, "CL (R)": curr_r_cl, "UCL (R)": curr_r_ucl, "LCL (R)": curr_r_lcl})
                    df_tables.append(df_pn)

                display_calculation_table(pd.concat(df_tables), "X-Bar R Chart")
                render_dynamic_variable_chart(configs_x, "X-Bar Chart", "Mean")
                render_dynamic_variable_chart(configs_r, "R Chart", "Range")
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
                configs_x = []
                configs_s = []
                df_tables = []
                
                xbar_1 = df_p1_clean.mean(axis=0)
                s_1 = df_p1_clean.std(axis=0, ddof=1)
                
                s_bar_1 = s_1.mean()
                x_dbl_bar_1 = xbar_1.mean()
                c = get_constants(sample_size_p1)
                
                ucl_x_1 = x_dbl_bar_1 + (c['A3'] * s_bar_1)
                lcl_x_1 = x_dbl_bar_1 - (c['A3'] * s_bar_1)
                ucl_s_1 = c['B4'] * s_bar_1
                lcl_s_1 = c['B3'] * s_bar_1
                
                configs_x.append({'name': 'Phase 1', 'data': xbar_1, 'cl': pd.Series(x_dbl_bar_1, index=xbar_1.index), 'ucl': pd.Series(ucl_x_1, index=xbar_1.index), 'lcl': pd.Series(lcl_x_1, index=xbar_1.index)})
                configs_s.append({'name': 'Phase 1', 'data': s_1, 'cl': pd.Series(s_bar_1, index=s_1.index), 'ucl': pd.Series(ucl_s_1, index=s_1.index), 'lcl': pd.Series(lcl_s_1, index=s_1.index)})
                
                df_p1 = pd.DataFrame({"Phase": "Phase 1", "Mean (X-Bar)": xbar_1, "StdDev (S)": s_1, "CL (X)": x_dbl_bar_1, "UCL (X)": ucl_x_1, "LCL (X)": lcl_x_1, "CL (S)": s_bar_1, "UCL (S)": ucl_s_1, "LCL (S)": lcl_s_1})
                df_tables.append(df_p1)
                
                curr_x_cl, curr_x_ucl, curr_x_lcl = x_dbl_bar_1, ucl_x_1, lcl_x_1
                curr_s_cl, curr_s_ucl, curr_s_lcl = s_bar_1, ucl_s_1, lcl_s_1
                
                for p_dict in additional_phases_data:
                    xbar_n = p_dict['data'].mean(axis=0)
                    s_n = p_dict['data'].std(axis=0, ddof=1)
                    
                    if "Recalculate Option" in p_dict['mode'] and not p_dict['data'].empty:
                        c_n = get_constants(p_dict['n'])
                        curr_x_cl = xbar_n.mean()
                        curr_s_cl = s_n.mean()
                        curr_x_ucl = curr_x_cl + (c_n['A3'] * curr_s_cl)
                        curr_x_lcl = curr_x_cl - (c_n['A3'] * curr_s_cl)
                        curr_s_ucl = c_n['B4'] * curr_s_cl
                        curr_s_lcl = c_n['B3'] * curr_s_cl
                        
                    configs_x.append({'name': f"Phase {p_dict['phase_num']}", 'data': xbar_n, 'cl': pd.Series(curr_x_cl, index=xbar_n.index), 'ucl': pd.Series(curr_x_ucl, index=xbar_n.index), 'lcl': pd.Series(curr_x_lcl, index=xbar_n.index)})
                    configs_s.append({'name': f"Phase {p_dict['phase_num']}", 'data': s_n, 'cl': pd.Series(curr_s_cl, index=s_n.index), 'ucl': pd.Series(curr_s_ucl, index=s_n.index), 'lcl': pd.Series(curr_s_lcl, index=s_n.index)})
                    
                    df_pn = pd.DataFrame({"Phase": f"Phase {p_dict['phase_num']}", "Mean (X-Bar)": xbar_n, "StdDev (S)": s_n, "CL (X)": curr_x_cl, "UCL (X)": curr_x_ucl, "LCL (X)": curr_x_lcl, "CL (S)": curr_s_cl, "UCL (S)": curr_s_ucl, "LCL (S)": curr_s_lcl})
                    df_tables.append(df_pn)

                display_calculation_table(pd.concat(df_tables), "X-Bar S Chart")
                render_dynamic_variable_chart(configs_x, "X-Bar Chart", "Mean")
                render_dynamic_variable_chart(configs_s, "S Chart", "Standard Deviation")
        else:
            st.warning("⚠️ X-Bar S Chart requires Sample Size (n) > 1.")