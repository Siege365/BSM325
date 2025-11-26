import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    t = st.number_input("Initial guess (t₀):", value=2.00000, step=0.00001, format="%.5f")
with col2:
    tol = st.number_input("Tolerance level:", value=0.00001, step=0.00001, format="%.5f")

if st.button("Calculate", type="primary"):
    # Initialize tracking variables
    iterations = []
    t_current = t
    iteration = 0
    
    # Newton's Method iterative loop
    while True:
        # Calculate function and derivative values
        f_val = f(t_current)
        fp_val = fp(t_current)
        
        # Apply Newton's formula: t_new = t - f(t)/f'(t)
        t_new = t_current - f_val/fp_val
        
        # Calculate relative error: |ε| = (xₖ₊₁ - xₖ) / xₖ₊₁
        error = abs((t_new - t_current) / t_new) if t_new != 0 else 0
        
        # Store iteration data for display
        iterations.append({
            "k": iteration,
            "xₖ": f"{t_current:.5f}",
            "f(xₖ)": f"{f_val:.5f}",
            "f'(xₖ)": f"{fp_val:.5f}",
            "xₖ₊₁": f"{t_new:.5f}",
            "|εr|": f"{error:.5f}"
        })
        
        iteration += 1
        
        # Check convergence: stop if change is less than tolerance
        if error < tol:
            break
        
        # Update current value for next iteration
        t_current = t_new
    
    st.success(f"✅ Root found at **t ≈ {t_new:.8f}** in {iteration} iterations")
    
    # Create visualizations
    st.markdown("### 📊 Visualization")
    
    # Generate data for plotting
    t_range = np.linspace(0, max(t_new * 1.5, 5), 300)
    f_values = [math.exp(-0.3*t_val) - 0.5 for t_val in t_range]
    battery_percent = [math.exp(-0.3*t_val) * 100 for t_val in t_range]
    
    # Create two columns for graphs
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Graph 1: Function f(t) with root
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(t_range, f_values, 'b-', linewidth=2, label='f(t) = e^(-0.3t) - 0.5')
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.axvline(x=t_new, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.plot(t_new, 0, 'ro', markersize=10, label=f'Root at t={t_new:.4f}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time (t)', fontsize=12)
        ax1.set_ylabel('f(t)', fontsize=12)
        ax1.set_title('Function f(t) and Root Location', fontsize=14, fontweight='bold')
        ax1.legend()
        st.pyplot(fig1)
    
    with col_b:
        # Graph 2: Battery capacity over time
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(t_range, battery_percent, 'g-', linewidth=2, label='Battery Capacity')
        ax2.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='50% threshold')
        ax2.axvline(x=t_new, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.plot(t_new, 50, 'ro', markersize=10, label=f't={t_new:.4f}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time (t)', fontsize=12)
        ax2.set_ylabel('Battery Capacity (%)', fontsize=12)
        ax2.set_title('Battery Decay Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_ylim([0, 105])
        st.pyplot(fig2)
    
    # Graph 3: Convergence of iterations
    st.markdown("### 🎯 Convergence Analysis")
    iteration_nums = [item["k"] for item in iterations]
    t_values = [float(item["xₖ₊₁"]) for item in iterations]
    errors = [float(item["|εr|"]) for item in iterations]
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence to root
    ax3a.plot(iteration_nums, t_values, 'bo-', linewidth=2, markersize=8)
    ax3a.axhline(y=t_new, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Final root')
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlabel('Iteration', fontsize=12)
    ax3a.set_ylabel('t value', fontsize=12)
    ax3a.set_title('Convergence to Root', fontsize=14, fontweight='bold')
    ax3a.legend()
    
    # Error reduction
    ax3b.semilogy(iteration_nums, errors, 'mo-', linewidth=2, markersize=8)
    ax3b.grid(True, alpha=0.3)
    ax3b.set_xlabel('Iteration', fontsize=12)
    ax3b.set_ylabel('Error (log scale)', fontsize=12)
    ax3b.set_title('Error Reduction', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.markdown("### 📋 Iteration Details")
    df = pd.DataFrame(iterations)
    st.dataframe(df, use_container_width=True)
    
    st.info(f"Battery reaches 50% capacity after approximately **{t_new:.4f} time units**")
