"""
Streamlit web interface for Bioreactor Process Application
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

API_URL = "http://localhost:8080/api"

st.set_page_config(
    page_title="Bioreactor Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Sidebar ----
st.sidebar.title("🧬 Bioreactor Simulator")
st.sidebar.markdown("Configure and run mechanistic bioprocess simulations")

# Load organisms from API
@st.cache_data(ttl=30)
def load_organisms():
    try:
        r = requests.get(f"{API_URL}/organisms", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

organisms = load_organisms()
org_options = {o['id']: f"{o['name']} ({o['type']})" for o in organisms} if organisms else {"ecoli_k12": "E. coli K-12"}

st.sidebar.divider()

# Organism selection
selected_org = st.sidebar.selectbox(
    "Organism",
    options=list(org_options.keys()),
    format_func=lambda x: org_options[x],
    index=list(org_options.keys()).index('ecoli_k12') if 'ecoli_k12' in org_options else 0
)

# Get organism details
org_details = next((o for o in organisms if o['id'] == selected_org), None)
if org_details:
    st.sidebar.caption(f"**Description:** {org_details.get('description', 'N/A')[:200]}...")

st.sidebar.divider()

# Bioreactor parameters
st.sidebar.header("Initial Conditions")
col1, col2 = st.sidebar.columns(2)
with col1:
    xv_init = st.number_input("X_v [g/L]", 0.0, 100.0, 1.0, 0.1, key="xv")
    s_init = st.number_input("S₀ [g/L]", 0.0, 500.0, 5.0, 0.5, key="s")
    v_init = st.number_input("Volume [L]", 0.1, 100000.0, 10.0, 1.0, key="v")
with col2:
    temp_init = st.number_input("Temp [°C]", 10.0, 60.0, 37.0, 0.5, key="temp")
    ph_init = st.number_input("pH", 3.0, 10.0, 7.0, 0.1, key="ph")
    t_end = st.number_input("Duration [h]", 1.0, 500.0, 24.0, 1.0, key="tend")

st.sidebar.divider()

if st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True):
    with st.spinner(f"Simulating {org_options.get(selected_org, selected_org)}..."):
        try:
            payload = {
                "organism_id": selected_org,
                "X_v": xv_init,
                "S": s_init,
                "V": v_init,
                "temperature": temp_init,
                "pH": ph_init,
                "t_end": t_end,
                "name": f"Sim: {selected_org}"
            }
            
            r = requests.post(f"{API_URL}/simulate", json=payload, timeout=120)
            if r.status_code == 200:
                result = r.json()
                
                # Extract results
                if result.get("status") == "completed" and "raw" in result:
                    sim_data = json.loads(result["raw"])
                    st.success(f"✅ {sim_data.get('message', 'Done')}")
                    
                    # Store in session
                    st.session_state.last_sim = {
                        "organism": sim_data.get("organism", selected_org),
                        "final_X_v": sim_data.get("final_X_v"),
                        "final_S": sim_data.get("final_S"),
                        "final_B": sim_data.get("final_B"),
                        "final_viability": sim_data.get("final_viability"),
                        "time_points": sim_data.get("time_points")
                    }
                    
                    # Fetch time series from DB
                    sim_id = result.get("simulation_id")
                    if sim_id:
                        r2 = requests.get(f"{API_URL}/simulations/{sim_id}/results", timeout=10)
                        if r2.status_code == 200:
                            st.session_state.last_timeseries = r2.json()
                else:
                    st.warning(result.get("message", "Unknown result"))
                    st.json(result)
            else:
                st.error(f"API error: {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Backend API not running. Start with: `java -jar backend/target/bioreactor-app-0.1.0.jar`")
        except Exception as e:
            st.error(f"Error: {e}")

# ---- Main area ----
st.title("🧬 Bioreactor Process Simulator")
st.markdown("Mechanistic ODE-based simulation of microbial growth, substrate consumption, and product formation.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Simulation", "📈 Compare", "📋 Parameters", "📚 About"])

with tab1:
    if "last_sim" in st.session_state:
        sim = st.session_state.last_sim
        ts = st.session_state.get("last_timeseries", [])
        
        # Summary cards
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Final VCD", f"{sim['final_X_v']:.3f} g/L")
        metrics_cols[1].metric("Final Substrate", f"{sim['final_S']:.3f} g/L")
        metrics_cols[2].metric("Final Byproduct", f"{sim['final_B']:.3f} g/L")
        metrics_cols[3].metric("Viability", f"{sim['final_viability']:.1%}")
        
        # Charts
        if ts and len(ts) > 0 and "time" in ts[0]:
            d = ts[0]
            df = pd.DataFrame(d)
            
            # Growth + substrate
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Scatter(x=df['time'], y=df['X_v'], name="VCD", line=dict(color='green', width=2)))
            fig1.add_trace(go.Scatter(x=df['time'], y=df['X_d'], name="Dead", line=dict(color='red', dash='dot', width=1)))
            fig1.add_trace(go.Scatter(x=df['time'], y=df['substrate'], name="Substrate", line=dict(color='blue', width=2)), secondary_y=True)
            fig1.update_layout(title="Growth & Substrate", xaxis_title="Time [h]", hovermode='x unified')
            fig1.update_yaxes(title_text="Biomass [g/L]", secondary_y=False)
            fig1.update_yaxes(title_text="Substrate [g/L]", secondary_y=True)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Products
            fig2 = make_subplots(specs=[[{"secondary_y": False}]])
            fig2.add_trace(go.Scatter(x=df['time'], y=df['product'], name="Product", line=dict(color='purple', width=2)))
            fig2.add_trace(go.Scatter(x=df['time'], y=df['byproduct'], name="Byproduct", line=dict(color='orange', width=2)))
            fig2.update_layout(title="Products", xaxis_title="Time [h]", yaxis_title="Concentration [g/L]", hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Environment
            fig3 = make_subplots(rows=1, cols=3, subplot_titles=["Temperature", "pH", "Viability"])
            fig3.add_trace(go.Scatter(x=df['time'], y=df['temperature'], name="T", line=dict(color='red')), row=1, col=1)
            fig3.add_trace(go.Scatter(x=df['time'], y=df['pH'], name="pH", line=dict(color='cyan')), row=1, col=2)
            fig3.add_trace(go.Scatter(x=df['time'], y=df['viability'], name="Viability", line=dict(color='brown'), fill='tozeroy'), row=1, col=3)
            fig3.update_layout(height=300, hovermode='x unified', showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Raw data
            with st.expander("Raw time series data"):
                st.dataframe(df.round(4), use_container_width=True)
    else:
        st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to get started.")
        
        # Placeholder
        fig = go.Figure()
        fig.add_annotation(x=12, y=5, text="Run a simulation to see results", showarrow=False)
        fig.update_layout(xaxis_range=[0,24], yaxis_range=[0,10], xaxis_title="Time [h]", yaxis_title="Concentration [g/L]")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Multi-organism Comparison")
    st.markdown("Compare growth curves across different organisms under the same conditions.")
    
    # Quick comparison
    if st.button("🔬 Compare All Organisms"):
        organisms_to_compare = {"ecoli_k12": "E. coli K-12", "scerevisiae": "S. cerevisiae", "bsubtilis": "B. subtilis"}
        
        fig_comp = make_subplots(rows=1, cols=2, subplot_titles=["Biomass (X_v)", "Substrate (S)"])
        colors = {'ecoli_k12': 'green', 'scerevisiae': 'red', 'bsubtilis': 'blue'}
        
        with st.spinner("Running comparisons..."):
            for org_id, org_name in organisms_to_compare.items():
                try:
                    payload = {
                        "organism_id": org_id, "X_v": 1.0, "S": 10.0, "V": 10.0,
                        "temperature": 37.0 if org_id != 'scerevisiae' else 30.0,
                        "pH": 7.0 if org_id != 'scerevisiae' else 5.5,
                        "t_end": 24.0
                    }
                    r = requests.post(f"{API_URL}/simulate", json=payload, timeout=120)
                    if r.status_code == 200:
                        res = r.json()
                        sim_id = res.get("simulation_id")
                        if sim_id:
                            r2 = requests.get(f"{API_URL}/simulations/{sim_id}/results", timeout=10)
                            if r2.status_code == 200:
                                ts = r2.json()
                                if ts and "time" in ts[0]:
                                    d = ts[0]
                                    c = colors.get(org_id, 'gray')
                                    fig_comp.add_trace(go.Scatter(x=d['time'], y=d['X_v'], name=f"{org_name}", line=dict(color=c, width=2)), row=1, col=1)
                                    fig_comp.add_trace(go.Scatter(x=d['time'], y=d['substrate'], name=f"{org_name}", line=dict(color=c, width=2, dash='dash'), showlegend=False), row=1, col=2)
                except Exception as e:
                    st.warning(f"Failed for {org_name}: {e}")
        
        fig_comp.update_layout(height=400, hovermode='x unified')
        fig_comp.update_xaxes(title_text="Time [h]", row=1, col=1)
        fig_comp.update_xaxes(title_text="Time [h]", row=1, col=2)
        fig_comp.update_yaxes(title_text="Biomass [g/L]", row=1, col=1)
        fig_comp.update_yaxes(title_text="Substrate [g/L]", row=1, col=2)
        st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    st.header("Organism Parameters")
    
    if organisms:
        df_orgs = pd.DataFrame(organisms).drop('description', axis=1)
        df_orgs.columns = ['ID', 'Name', 'Type', 'Gram Stain', 'Genome Model']
        st.dataframe(df_orgs, use_container_width=True, hide_index=True)
    
    st.divider()
    st.markdown("""
    ### Growth Models
    | Model | Equation | Use Case |
    |-------|----------|----------|
    | Monod | μ = μₘₐₓ · S/(Kₛ+S) | Simple, no inhibition |
    | Moser | μ = μₘₐₓ · Sⁿ/(Kₛ+Sⁿ) | Adjustable sensitivity |
    | Andrews | μ = μₘₐₓ · S/(Kₛ+S+S²/Kᵢ) | Substrate inhibition |
    | Haldane | μ = μₘₐₓ · S/(Kₛ+S)·(1-S/Sₘₐₓ)ⁿ | Strong inhibition at high S |
    | Logistic | μ = μₘₐₓ·(1−X/Xₘₐₓ) | Carrying capacity limited |
    
    ### Product Kinetics
    - **Luedeking-Piret**: qₚ = α·μ + β (growth-associated + non-growth-associated)
    - **Overflow metabolism**: triggered when q_S > q_S_crit (e.g., aerobic acetate in E. coli, ethanol in S. cerevisiae)
    
    ### Environment Effects
    - **Temperature**: Cardinal model (T_min, T_opt, T_max) or Arrhenius
    - **pH**: Gaussian tolerance φ(pH) = exp(-(pH-pH_opt)²/(2σ²))
    """)

with tab4:
    st.header("About")
    st.markdown("""
    ### Bioreactor Process Application
    Built for the **SDSC-STEM4Health Hackathon Group 10**.
    
    #### Architecture
    ```
    Streamlit UI  ←─ HTTP ─→  Java/Spring Boot REST API  ←─ subprocess ─→  Python ODE Engine
         │                          │                                       │
         └──────────────────────────┴── PostgreSQL ──────────────────────────┘
    ```
    
    #### Key Features
    - **9 organism strains** with calibrated kinetic parameters
    - **Mechanistic ODE model** (9 state variables)
    - **Multiple growth models**: Monod, Moser, Andrews, Haldane, Logistic
    - **Population-resolved**: Viable + dead cell compartments
    - **Overflow metabolism** with byproduct formation
    - **Environmental effects**: Temperature, pH, dissolved oxygen
    - **418,023 experimental data points** for validation
    
    #### Data Sources
    - KEGG, BRENDA, BioCyc, BioModels, BiGG public databases
    - DTB Project (61 batches) and BRPMM Project (162 batches)
    """)

# Footer
st.divider()
st.caption("SDSC-STEM4Health Group 10 · Bioreactor Process Application · v0.1.0")
