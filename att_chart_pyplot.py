import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION MAIN ---
st.set_page_config(page_title="Attribute QC Chart Generator", layout="wide")
st.title("📊 Attribute Control Chart Generator (Plotly)")
st.markdown("Interactive application for attribute control charts (**p, np, c, u Chart**) with Detailed Calculation Tables and Variable Sample Size.")

# --- SIDEBAR (INPUT PARAMETERS) ---
st.sidebar.header("⚙️ Chart Parameters")

chart_options = st.sidebar.multiselect(
    "Select Charts to Generate:",
    ["p-Chart", "np-Chart", "c-Chart", "u-Chart"],
    default=["p-Chart"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Phase 1 Settings (Base Data)")
num_samples_p1 = st.sidebar.number_input("Phase 1: Number of Samples (m)", min_value=1, value=25, step=1)
default_n = st.sidebar.number_input("Default Sample Size pre-fill (n)", min_value=1, value=100, step=1)

# --- ALARM LOGIC (Full Western Electric Rules with Vector Limits) ---
def check_alarms(data, cl_stat, ucl, lcl):
    alarms = []
    violation_indices = set()
    n = len(data)
    valid_mask = ~data.isna()
    
    if not isinstance(ucl, pd.Series): ucl = pd.Series([ucl]*n, index=data.index)
    if not isinstance(lcl, pd.Series): lcl = pd.Series([lcl]*n, index=data.index)
    if not isinstance(cl_stat, pd.Series): cl_stat = pd.Series([cl_stat]*n, index=data.index)
    
    sigma = (ucl - cl_stat) / 3
    signs = np.sign(data - cl_stat)
    
    # 0. Out of bounds (>UCL or <LCL)
    for i in range(n):
        if not valid_mask.iloc[i]: continue
        if data.iloc[i] > ucl.iloc[i] or data.iloc[i] < lcl.iloc[i]:
            disp_idx = str(data.index[i]) 
            alarms.append(f"**[Type 0]** Point **{disp_idx}** is out of Control Limits.")
            violation_indices.add(data.index[i])
    
    # 1. 7 points in a row on one side
    for i in range(n - 6):
        if not valid_mask.iloc[i:i+7].all(): continue
        if np.all(signs.iloc[i:i+7] == 1) or np.all(signs.iloc[i:i+7] == -1):
            alarms.append(f"**[Type 1]** **{str(data.index[i])} to {str(data.index[i+6])}**: 7 consecutive points on one side.")
            violation_indices.update(data.index[i:i+7])
            
    # 2. 10 out of 11 points on one side
    for i in range(n - 10):
        if not valid_mask.iloc[i:i+11].all(): continue
        window = signs.iloc[i:i+11]
        if np.sum(window == 1) >= 10 or np.sum(window == -1) >= 10:
            alarms.append(f"**[Type 2]** **{str(data.index[i])} to {str(data.index[i+10])}**: 10 out of 11 points on one side.")
            violation_indices.update(data.index[i:i+11])
            
    # 3. 12 out of 14 points on one side
    for i in range(n - 13):
        if not valid_mask.iloc[i:i+14].all(): continue
        window = signs.iloc[i:i+14]
        if np.sum(window == 1) >= 12 or np.sum(window == -1) >= 12:
            alarms.append(f"**[Type 3]** **{str(data.index[i])} to {str(data.index[i+13])}**: 12 out of 14 points on one side.")
            violation_indices.update(data.index[i:i+14])
            
    # 4. 6 points steadily increasing/decreasing
    for i in range(n - 5):
        if not valid_mask.iloc[i:i+6].all(): continue
        diffs = np.diff(data.iloc[i:i+6])
        if np.all(diffs > 0) or np.all(diffs < 0):
            alarms.append(f"**[Type 4]** **{str(data.index[i])} to {str(data.index[i+5])}**: 6 consecutive points steadily increasing/decreasing.")
            violation_indices.update(data.index[i:i+6])
            
    # 5. 2 out of 3 in Zone A (same side)
    for i in range(n - 2):
        if not valid_mask.iloc[i:i+3].all(): continue
        window = data.iloc[i:i+3]
        mean_w = cl_stat.iloc[i:i+3]
        sig_w = sigma.iloc[i:i+3]
        
        if np.sum(window > (mean_w + 2*sig_w)) >= 2 or np.sum(window < (mean_w - 2*sig_w)) >= 2:
            alarms.append(f"**[Type 5]** **{str(data.index[i])} to {str(data.index[i+2])}**: 2 out of 3 points in Zone A.")
            violation_indices.update(data.index[i:i+3])
            
    # 6. 4 out of 5 in Zone B or beyond (same side)
    for i in range(n - 4):
        if not valid_mask.iloc[i:i+5].all(): continue
        window = data.iloc[i:i+5]
        mean_w = cl_stat.iloc[i:i+5]
        sig_w = sigma.iloc[i:i+5]
        
        if np.sum(window > (mean_w + sig_w)) >= 4 or np.sum(window < (mean_w - sig_w)) >= 4:
            alarms.append(f"**[Type 6]** **{str(data.index[i])} to {str(data.index[i+4])}**: 4 out of 5 points in Zone B.")
            violation_indices.update(data.index[i:i+5])

    if not alarms:
        alarms.append("✅ Process is in control.")
        
    return list(dict.fromkeys(alarms)), list(violation_indices)

# --- MULTI-PHASE DYNAMIC RENDERER ---
def render_dynamic_chart(configs, title, y_label):
    col1, col2 = st.columns([3, 1])
    
    all_alarms = []
    all_violations = []
    
    combined_data = pd.Series(dtype=float)
    combined_ucl = pd.Series(dtype=float)
    combined_cl_plot = pd.Series(dtype=float)
    combined_cl_stat = pd.Series(dtype=float)
    combined_lcl = pd.Series(dtype=float)
    
    phase_transitions = []
    
    for cfg in configs:
        data = cfg['data'].copy()
        idx_names = [f"{cfg['name']}-{x}" for x in data.index]
        data.index = idx_names
        
        ucl = cfg['ucl'].copy() if isinstance(cfg['ucl'], pd.Series) else pd.Series([cfg['ucl']]*len(data))
        cl_plot = cfg['cl_plot'].copy() if isinstance(cfg['cl_plot'], pd.Series) else pd.Series([cfg['cl_plot']]*len(data))
        cl_stat = cfg['cl_stat'].copy() if isinstance(cfg['cl_stat'], pd.Series) else pd.Series([cfg['cl_stat']]*len(data))
        lcl = cfg['lcl'].copy() if isinstance(cfg['lcl'], pd.Series) else pd.Series([cfg['lcl']]*len(data))
        
        ucl.index = idx_names
        cl_plot.index = idx_names
        cl_stat.index = idx_names
        lcl.index = idx_names
        
        lcl = lcl.clip(lower=0.0)
        
        alarms, v_idx = check_alarms(data.dropna(), cl_stat, ucl, lcl)
        all_alarms += [f"**[{cfg['name']}]** {a}" for a in alarms if "✅" not in a]
        all_violations.extend(list(v_idx))
        
        if not data.empty:
            phase_transitions.append((cfg['name'], len(combined_data)))
            
        combined_data = pd.concat([combined_data, data])
        combined_ucl = pd.concat([combined_ucl, ucl])
        combined_cl_plot = pd.concat([combined_cl_plot, cl_plot])
        combined_cl_stat = pd.concat([combined_cl_stat, cl_stat])
        combined_lcl = pd.concat([combined_lcl, lcl])
    
    if not all_alarms:
        all_alarms.append("✅ Process is in control across all phases.")
        
    valid_combined = combined_data.dropna()
    
    with col1:
        fig = go.Figure()
        x_vals = list(range(len(valid_combined)))
        display_labels = [str(x).split('-', 1)[1].replace('Sample ', 'S') for x in valid_combined.index]
        
        # Calculate dynamic sigmas for Zone calculations
        sig_dyn = (combined_ucl - combined_cl_stat) / 3
        
        # Upper Bounds
        uc = combined_cl_stat + sig_dyn
        ub = combined_cl_stat + 2 * sig_dyn
        ua = combined_ucl
        
        # Lower Bounds (Clip to 0 for attribute charts)
        lc = np.maximum(combined_lcl, combined_cl_stat - sig_dyn)
        lb = np.maximum(combined_lcl, combined_cl_stat - 2 * sig_dyn)
        la = combined_lcl

        # ----------------------------------------------------
        # ZONES BACKGROUND (DYNAMIC STEP FILLS)
        # ----------------------------------------------------
        # Upper Zone C (Green)
        fig.add_trace(go.Scatter(x=x_vals, y=combined_cl_stat.values, mode='lines', line=dict(width=0, shape='hvh'), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=uc.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(168, 230, 207, 0.3)', showlegend=False, hoverinfo='skip'))
        # Upper Zone B (Orange)
        fig.add_trace(go.Scatter(x=x_vals, y=ub.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 211, 182, 0.3)', showlegend=False, hoverinfo='skip'))
        # Upper Zone A (Red)
        fig.add_trace(go.Scatter(x=x_vals, y=ua.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 170, 165, 0.3)', showlegend=False, hoverinfo='skip'))

        # Lower Zone C (Green)
        fig.add_trace(go.Scatter(x=x_vals, y=combined_cl_stat.values, mode='lines', line=dict(width=0, shape='hvh'), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=lc.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(168, 230, 207, 0.3)', showlegend=False, hoverinfo='skip'))
        # Lower Zone B (Orange)
        fig.add_trace(go.Scatter(x=x_vals, y=lb.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 211, 182, 0.3)', showlegend=False, hoverinfo='skip'))
        # Lower Zone A (Red)
        fig.add_trace(go.Scatter(x=x_vals, y=la.values, mode='lines', line=dict(width=0, shape='hvh'), fill='tonexty', fillcolor='rgba(255, 170, 165, 0.3)', showlegend=False, hoverinfo='skip'))

        # ----------------------------------------------------
        # BOUNDARY LINES
        # ----------------------------------------------------
        fig.add_trace(go.Scatter(x=x_vals, y=combined_ucl.values, mode='lines', line=dict(color='red', width=1.5, shape='hvh', dash='dash'), name='UCL', hoverinfo='y'))
        fig.add_trace(go.Scatter(x=x_vals, y=combined_cl_plot.values, mode='lines', line=dict(color='blue', width=1.5, shape='hvh'), name='Center Line', hoverinfo='y'))
        fig.add_trace(go.Scatter(x=x_vals, y=combined_lcl.values, mode='lines', line=dict(color='red', width=1.5, shape='hvh', dash='dash'), name='LCL', hoverinfo='y'))

        # Optional dashed separators for zones
        fig.add_trace(go.Scatter(x=x_vals, y=uc.values, mode='lines', line=dict(color='gray', width=1, shape='hvh', dash='dash'), opacity=0.3, showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=ub.values, mode='lines', line=dict(color='gray', width=1, shape='hvh', dash='dash'), opacity=0.3, showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=lc.values, mode='lines', line=dict(color='gray', width=1, shape='hvh', dash='dash'), opacity=0.3, showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_vals, y=lb.values, mode='lines', line=dict(color='gray', width=1, shape='hvh', dash='dash'), opacity=0.3, showlegend=False, hoverinfo='skip'))

        if len(valid_combined) > 0:
            last_x = x_vals[-1]
            fig.add_annotation(x=last_x, y=combined_ucl.iloc[-1], text=f"UCL: {combined_ucl.iloc[-1]:.3f}", showarrow=False, xanchor="left", xshift=10)
            fig.add_annotation(x=last_x, y=combined_cl_plot.iloc[-1], text=f"CL: {combined_cl_plot.iloc[-1]:.3f}", showarrow=False, xanchor="left", xshift=10)
            fig.add_annotation(x=last_x, y=combined_lcl.iloc[-1], text=f"LCL: {combined_lcl.iloc[-1]:.3f}", showarrow=False, xanchor="left", xshift=10)

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
                hovertemplate="Sample: %{x}<br>Value: %{y:.3f}<extra></extra>"
            ))

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
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)', rangemode='tozero')
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("#### 🚨 Alarm Analysis")
        for a in all_alarms:
            if "✅" in a: st.success(a)
            else: st.error(a)


# --- UI: DATA INPUT FOR BASE PHASE (PHASE 1) ---
st.subheader("📝 Phase 1: Base Observation Data")
index_p1 = [f"{i+1}" for i in range(num_samples_p1)] 

df_init_p1 = pd.DataFrame({
    "Sample Size (n)": [default_n] * num_samples_p1,
    "Count (D or c)": [0] * num_samples_p1
}, index=index_p1)

df_phase1 = st.data_editor(df_init_p1, use_container_width=True, key="p1_editor")

st.markdown("### ✂️ Data Cleaning (Phase 1)")
st.info("Select samples to exclude (e.g., outliers) so they are not included in Phase 1 CL and Limits calculations.")
excluded_samples = st.multiselect("Exclude Phase 1 Samples:", options=index_p1)
valid_data_p1 = df_phase1.drop(index=excluded_samples)

st.divider()

# --- UI: DYNAMIC ADDITIONAL PHASES ---
st.subheader("🔄 Additional Phases (Monitoring / Recalculate)")
num_additional_phases = st.number_input("How many additional phases?", min_value=0, max_value=10, value=0, step=1)

additional_phases_data = []

for i in range(num_additional_phases):
    phase_num = i + 2
    st.markdown(f"#### 📝 Phase {phase_num}")
    
    col_m1, col_m2 = st.columns([2, 1])
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
        num_samples_pn = st.number_input(f"Samples (m) for Phase {phase_num}", min_value=1, value=10, step=1, key=f"m_{phase_num}")
        
    index_pn = [f"{j+1}" for j in range(num_samples_pn)]
    
    df_init_pn = pd.DataFrame({
        "Sample Size (n)": [default_n] * num_samples_pn,
        "Count (D or c)": [0] * num_samples_pn
    }, index=index_pn)
    
    df_pn = st.data_editor(df_init_pn, use_container_width=True, key=f"editor_{phase_num}")
    
    additional_phases_data.append({
        'phase_num': phase_num,
        'mode': mode,
        'data': df_pn
    })

# --- HELPER FUNCTION TO DISPLAY TABLE ---
def display_calculation_table(df_summary, chart_name):
    st.markdown(f"#### 📋 Step-by-step Calculation Table ({chart_name})")
    st.dataframe(df_summary.style.format({
        "Fraction (p_i)": "{:.3f}",
        "Center Line": "{:.3f}",
        "Std Dev (sigma)": "{:.3f}",
        "LCL": "{:.3f}",
        "UCL": "{:.3f}"
    }), use_container_width=True)


# --- CHART GENERATOR EXECUTION ---
if st.button("Generate Dynamic Combined Charts & Analysis", type="primary"):
    if not chart_options:
        st.warning("Please select at least 1 chart from the left sidebar.")

    # ==========================================
    # 1. p-CHART LOGIC
    # ==========================================
    if "p-Chart" in chart_options:
        st.divider()
        st.subheader("📉 p-Chart (Fraction Nonconforming - Variable Sample Size)")
        
        configs = []
        df_tables = []
        
        n_1 = valid_data_p1["Sample Size (n)"]
        D_1 = valid_data_p1["Count (D or c)"]
        
        p_data_1 = D_1 / n_1
        p_bar_1 = D_1.sum() / n_1.sum() if n_1.sum() > 0 else 0
        
        cl_1 = pd.Series([p_bar_1]*len(n_1), index=n_1.index)
        sigma_p_1 = np.sqrt(p_bar_1 * (1 - p_bar_1) / n_1)
        ucl_1 = cl_1 + 3 * sigma_p_1
        lcl_1 = np.maximum(0, cl_1 - 3 * sigma_p_1)
        
        configs.append({'name': 'Phase 1', 'data': p_data_1, 'cl_plot': cl_1, 'cl_stat': cl_1, 'ucl': ucl_1, 'lcl': lcl_1})
        current_p_bar = p_bar_1 
        
        df_p1 = pd.DataFrame({
            "Phase": "Phase 1",
            "Sample Size (n_i)": n_1,
            "Nonconforming (D_i)": D_1,
            "Fraction (p_i)": p_data_1,
            "Center Line": cl_1,
            "Std Dev (sigma)": sigma_p_1,
            "LCL": lcl_1,
            "UCL": ucl_1
        })
        df_tables.append(df_p1)
        
        for p_dict in additional_phases_data:
            n_n = p_dict['data']["Sample Size (n)"]
            D_n = p_dict['data']["Count (D or c)"]
            p_data_n = D_n / n_n
            
            if "Recalculate Option" in p_dict['mode'] and not D_n.empty:
                current_p_bar = D_n.sum() / n_n.sum() if n_n.sum() > 0 else 0
                
            cl_n = pd.Series([current_p_bar]*len(n_n), index=n_n.index)
            sigma_p_n = np.sqrt(current_p_bar * (1 - current_p_bar) / n_n)
            ucl_n = cl_n + 3 * sigma_p_n
            lcl_n = np.maximum(0, cl_n - 3 * sigma_p_n)
                
            configs.append({'name': f"Phase {p_dict['phase_num']}", 'data': p_data_n, 'cl_plot': cl_n, 'cl_stat': cl_n, 'ucl': ucl_n, 'lcl': lcl_n})
            
            df_pn = pd.DataFrame({
                "Phase": f"Phase {p_dict['phase_num']}",
                "Sample Size (n_i)": n_n,
                "Nonconforming (D_i)": D_n,
                "Fraction (p_i)": p_data_n,
                "Center Line": cl_n,
                "Std Dev (sigma)": sigma_p_n,
                "LCL": lcl_n,
                "UCL": ucl_n
            })
            df_tables.append(df_pn)

        display_calculation_table(pd.concat(df_tables), "p-Chart")
        render_dynamic_chart(configs, "p-Chart", "Fraction (p)")

    # ==========================================
    # 2. np-CHART LOGIC
    # ==========================================
    if "np-Chart" in chart_options:
        st.divider()
        st.subheader("📉 np-Chart (Number of Defective Items - Variable Sample Size)")
        
        configs = []
        df_tables = []
        
        n_1 = valid_data_p1["Sample Size (n)"]
        D_1 = valid_data_p1["Count (D or c)"]
        
        p_bar_1 = D_1.sum() / n_1.sum() if n_1.sum() > 0 else 0
        
        cl_plot_1 = pd.Series([D_1.mean()]*len(n_1), index=n_1.index)
        cl_stat_1 = n_1 * p_bar_1
        sigma_np_1 = np.sqrt(n_1 * p_bar_1 * (1 - p_bar_1))
        
        ucl_1 = cl_stat_1 + 3 * sigma_np_1
        lcl_1 = np.maximum(0, cl_stat_1 - 3 * sigma_np_1)
        
        configs.append({'name': 'Phase 1', 'data': D_1, 'cl_plot': cl_plot_1, 'cl_stat': cl_stat_1, 'ucl': ucl_1, 'lcl': lcl_1})
        current_p_bar = p_bar_1
        current_D_bar = D_1.mean()
        
        df_p1 = pd.DataFrame({
            "Phase": "Phase 1",
            "Sample Size (n_i)": n_1,
            "Nonconforming (D_i)": D_1,
            "Center Line (n_i * p_bar)": cl_stat_1,
            "Std Dev (sigma)": sigma_np_1,
            "LCL": lcl_1,
            "UCL": ucl_1
        })
        df_tables.append(df_p1)
        
        for p_dict in additional_phases_data:
            n_n = p_dict['data']["Sample Size (n)"]
            D_n = p_dict['data']["Count (D or c)"]
            
            if "Recalculate Option" in p_dict['mode'] and not D_n.empty:
                current_p_bar = D_n.sum() / n_n.sum() if n_n.sum() > 0 else 0
                current_D_bar = D_n.mean()
                
            cl_plot_n = pd.Series([current_D_bar]*len(n_n), index=n_n.index)
            cl_stat_n = n_n * current_p_bar
            sigma_np_n = np.sqrt(n_n * current_p_bar * (1 - current_p_bar))
            
            ucl_n = cl_stat_n + 3 * sigma_np_n
            lcl_n = np.maximum(0, cl_stat_n - 3 * sigma_np_n)
                
            configs.append({'name': f"Phase {p_dict['phase_num']}", 'data': D_n, 'cl_plot': cl_plot_n, 'cl_stat': cl_stat_n, 'ucl': ucl_n, 'lcl': lcl_n})
            
            df_pn = pd.DataFrame({
                "Phase": f"Phase {p_dict['phase_num']}",
                "Sample Size (n_i)": n_n,
                "Nonconforming (D_i)": D_n,
                "Center Line (n_i * p_bar)": cl_stat_n,
                "Std Dev (sigma)": sigma_np_n,
                "LCL": lcl_n,
                "UCL": ucl_n
            })
            df_tables.append(df_pn)

        display_calculation_table(pd.concat(df_tables), "np-Chart")
        render_dynamic_chart(configs, "np-Chart", "Count (np)")

    # ==========================================
    # 3. c-CHART LOGIC
    # ==========================================
    if "c-Chart" in chart_options:
        st.divider()
        st.subheader("📉 c-Chart (Number of Defects)")
        st.info("💡 c-Chart typically assumes constant sample unit size.")
        
        configs = []
        df_tables = []
        D_1 = valid_data_p1["Count (D or c)"]
        
        c_bar_1 = D_1.mean()
        sigma_c_1 = np.sqrt(c_bar_1) if c_bar_1 > 0 else 0
        cl_1 = pd.Series([c_bar_1]*len(D_1), index=D_1.index)
        ucl_1 = cl_1 + 3 * sigma_c_1
        lcl_1 = np.maximum(0, cl_1 - 3 * sigma_c_1)
        
        configs.append({'name': 'Phase 1', 'data': D_1, 'cl_plot': cl_1, 'cl_stat': cl_1, 'ucl': ucl_1, 'lcl': lcl_1})
        current_c_bar = c_bar_1
        
        df_p1 = pd.DataFrame({
            "Phase": "Phase 1",
            "Defects (c_i)": D_1,
            "Center Line": cl_1,
            "Std Dev (sigma)": sigma_c_1,
            "LCL": lcl_1,
            "UCL": ucl_1
        })
        df_tables.append(df_p1)
        
        for p_dict in additional_phases_data:
            D_n = p_dict['data']["Count (D or c)"]
            
            if "Recalculate Option" in p_dict['mode'] and not D_n.empty:
                current_c_bar = D_n.mean()
                
            sigma_c_n = np.sqrt(current_c_bar) if current_c_bar > 0 else 0
            cl_n = pd.Series([current_c_bar]*len(D_n), index=D_n.index)
            ucl_n = cl_n + 3 * sigma_c_n
            lcl_n = np.maximum(0, cl_n - 3 * sigma_c_n)
                
            configs.append({'name': f"Phase {p_dict['phase_num']}", 'data': D_n, 'cl_plot': cl_n, 'cl_stat': cl_n, 'ucl': ucl_n, 'lcl': lcl_n})
            
            df_pn = pd.DataFrame({
                "Phase": f"Phase {p_dict['phase_num']}",
                "Defects (c_i)": D_n,
                "Center Line": cl_n,
                "Std Dev (sigma)": sigma_c_n,
                "LCL": lcl_n,
                "UCL": ucl_n
            })
            df_tables.append(df_pn)

        display_calculation_table(pd.concat(df_tables), "c-Chart")
        render_dynamic_chart(configs, "c-Chart", "Defects (c)")

    # ==========================================
    # 4. u-CHART LOGIC
    # ==========================================
    if "u-Chart" in chart_options:
        st.divider()
        st.subheader("📉 u-Chart (Defects per Unit - Variable Sample Size)")
        
        configs = []
        df_tables = []
        n_1 = valid_data_p1["Sample Size (n)"]
        D_1 = valid_data_p1["Count (D or c)"]
        
        u_data_1 = D_1 / n_1
        u_bar_1 = D_1.sum() / n_1.sum() if n_1.sum() > 0 else 0
        
        cl_1 = pd.Series([u_bar_1]*len(n_1), index=n_1.index)
        sigma_u_1 = np.sqrt(u_bar_1 / n_1)
        ucl_1 = cl_1 + 3 * sigma_u_1
        lcl_1 = np.maximum(0, cl_1 - 3 * sigma_u_1)
        
        configs.append({'name': 'Phase 1', 'data': u_data_1, 'cl_plot': cl_1, 'cl_stat': cl_1, 'ucl': ucl_1, 'lcl': lcl_1})
        current_u_bar = u_bar_1 
        
        df_p1 = pd.DataFrame({
            "Phase": "Phase 1",
            "Sample Size (n_i)": n_1,
            "Defects (D_i)": D_1,
            "Ratio (u_i = D/n)": u_data_1,
            "Center Line": cl_1,
            "Std Dev (sigma)": sigma_u_1,
            "LCL": lcl_1,
            "UCL": ucl_1
        })
        df_tables.append(df_p1)
        
        for p_dict in additional_phases_data:
            n_n = p_dict['data']["Sample Size (n)"]
            D_n = p_dict['data']["Count (D or c)"]
            u_data_n = D_n / n_n
            
            if "Recalculate Option" in p_dict['mode'] and not D_n.empty:
                current_u_bar = D_n.sum() / n_n.sum() if n_n.sum() > 0 else 0
                
            cl_n = pd.Series([current_u_bar]*len(n_n), index=n_n.index)
            sigma_u_n = np.sqrt(current_u_bar / n_n)
            ucl_n = cl_n + 3 * sigma_u_n
            lcl_n = np.maximum(0, cl_n - 3 * sigma_u_n)
                
            configs.append({'name': f"Phase {p_dict['phase_num']}", 'data': u_data_n, 'cl_plot': cl_n, 'cl_stat': cl_n, 'ucl': ucl_n, 'lcl': lcl_n})

            df_pn = pd.DataFrame({
                "Phase": f"Phase {p_dict['phase_num']}",
                "Sample Size (n_i)": n_n,
                "Defects (D_i)": D_n,
                "Ratio (u_i = D/n)": u_data_n,
                "Center Line": cl_n,
                "Std Dev (sigma)": sigma_u_n,
                "LCL": lcl_n,
                "UCL": ucl_n
            })
            df_tables.append(df_pn)

        display_calculation_table(pd.concat(df_tables), "u-Chart")
        render_dynamic_chart(configs, "u-Chart", "Defects per Unit (u)")