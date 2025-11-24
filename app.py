import streamlit as st
import math
import pandas as pd

def f(t):
    return math.exp(-0.3*t) - 0.5

def fp(t):
    return -0.3 * math.exp(-0.3*t)

st.title("🔋 Newton's Method — Battery Decay")
st.markdown("### Find when battery reaches 50% capacity")
st.latex(r"f(t) = e^{-0.3t} - 0.5")
st.latex(r"f'(t) = -0.3e^{-0.3t}")

col1, col2 = st.columns(2)
with col1:
    t = st.number_input("Initial guess (t₀):", value=2.0, step=0.1)
with col2:
    tol = st.number_input("Tolerance level:", value=1e-5, format="%.2e", step=1e-6)

if st.button("Calculate", type="primary"):
    iterations = []
    t_current = t
    iteration = 0
    
    while True:
        t_new = t_current - f(t_current)/fp(t_current)
        iteration += 1
        iterations.append({
            "Iteration": iteration,
            "t": f"{t_new:.8f}",
            "f(t)": f"{f(t_new):.8f}",
            "|Error|": f"{abs(t_new - t_current):.8f}"
        })
        
        if abs(t_new - t_current) < tol:
            break
        t_current = t_new
    
    st.success(f"✅ Root found at **t ≈ {t_new:.8f}** in {iteration} iterations")
    
    st.markdown("### Iteration Details")
    df = pd.DataFrame(iterations)
    st.dataframe(df, use_container_width=True)
    
    st.info(f"Battery reaches 50% capacity after approximately **{t_new:.4f} time units**")
