import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Glucose-Insulin Minimal Model Simulator",
    page_icon="🧪"
)
st.title("Glucose-Insulin Minimal Model Simulator")
st.markdown("An interactive simulator based on a simplified model of glucose and insulin dynamics.")

# --- The Mathematical Model (ODEs) ---

def glucose_insulin_model(t, y, k1, k2, k3, k4, meal_time, meal_amount):
    """
    Defines the system of Ordinary Differential Equations (ODEs).
    y[0] = G (Glucose concentration)
    y[1] = I (Insulin concentration)
    
    This is a conceptual, simplified model for demonstration.
    """
    G, I = y
    
    # Glucose input (Meal)
    # Simulate a meal as a Gaussian pulse of glucose
    peak_width = 1.0  # Width of the meal's effect
    D = meal_amount * np.exp(-((t - meal_time)**2) / (2 * peak_width**2))
    
    # dG/dt = Glucose Input - (Glucose clearance + Insulin-dependent clearance)
    dGdt = D - (k1 * G) - (k2 * G * I)
    
    # dI/dt = Insulin production (stimulated by G) - Insulin clearance
    # Use a simple threshold (k3 * max(0, G - threshold))
    glucose_threshold = 5.0 # Glucose level that triggers insulin
    dIdt = k3 * max(0, G - glucose_threshold) - (k4 * I)
    
    return [dGdt, dIdt]

# --- Streamlit UI (Sidebar) ---
st.sidebar.header("Model Parameters")

st.sidebar.markdown("### Physiology")
k1 = st.sidebar.slider("k1: Glucose Clearance Rate", 0.0, 1.0, 0.1, 0.01,
                       help="How fast glucose is cleared without insulin.")
k2 = st.sidebar.slider("k2: Insulin Sensitivity", 0.0, 1.0, 0.2, 0.01,
                       help="How effective insulin is at clearing glucose (k2*G*I).")
k3 = st.sidebar.slider("k3: Insulin Production Rate", 0.0, 1.0, 0.1, 0.01,
                       help="How fast insulin is produced in response to high glucose.")
k4 = st.sidebar.slider("k4: Insulin Clearance Rate", 0.0, 1.0, 0.1, 0.01,
                       help="How fast insulin degrades.")

st.sidebar.markdown("### Event")
meal_time = st.sidebar.slider("Meal Time (minutes)", 0, 120, 10)
meal_amount = st.sidebar.slider("Meal Size (Glucose Input)", 0.0, 20.0, 10.0, 0.5)

# --- Simulation Controls ---
sim_duration = 120  # Total simulation time in minutes
t_span = (0, sim_duration)
t_eval = np.linspace(0, sim_duration, 300) # Points to evaluate at

# Initial conditions (G=4.5, I=0.5)
y0 = [4.5, 0.5] 

# --- Run Simulation ---
with st.spinner("Running simulation..."):
    solution = solve_ivp(
        glucose_insulin_model, 
        t_span, 
        y0, 
        args=(k1, k2, k3, k4, meal_time, meal_amount),
        t_eval=t_eval
    )
    
    t = solution.t
    G = solution.y[0]
    I = solution.y[1]

# --- Plot Results ---
st.header("Simulation Results")

fig = go.Figure()

# Plot Glucose
fig.add_trace(go.Scatter(
    x=t, y=G, 
    mode='lines', 
    name='Glucose (G)',
    line=dict(color='blue')
))

# Plot Insulin
fig.add_trace(go.Scatter(
    x=t, y=I, 
    mode='lines', 
    name='Insulin (I)',
    line=dict(color='orange'),
    yaxis="y2" # Assign to secondary y-axis
))

# Add a marker for the meal
fig.add_vrect(
    x0=meal_time - 0.5, x1=meal_time + 0.5,
    fillcolor="green", opacity=0.25, 
    line_width=0,
    annotation_text="Meal", annotation_position="top left"
)

# Layout
fig.update_layout(
    title="Glucose and Insulin Dynamics Over Time",
    xaxis_title="Time (minutes)",
    yaxis=dict(title="Glucose (G)", color="blue"),
    yaxis2=dict(
        title="Insulin (I)", 
        color="orange",
        overlaying="y", 
        side="right",
        showgrid=False
    ),
    template="plotly_dark",
    legend=dict(x=0, y=1.1, orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("About this Model"):
    st.markdown("""
    This app solves a system of Ordinary Differential Equations (ODEs) to model a physiological system.
    
    - **Glucose (G):** Increases due to the "Meal" (D) and decreases due to natural clearance (k1*G) and insulin-driven clearance (k2*G*I).
    - **Insulin (I):** Increases when glucose is above a threshold (k3 * (G - threshold)) and decreases due to natural clearance (k4*I).
    
    **Try It:**
    - See what happens if you increase **Insulin Sensitivity (k2)**. Does glucose clear faster?
    - What if you decrease **Insulin Production (k3)**? Does the glucose spike stay high for longer? (Simulating Type 1 Diabetes).
    - What if you decrease **Insulin Sensitivity (k2)**? Does the body compensate? (Simulating Type 2 Diabetes / Insulin Resistance).
    """)
