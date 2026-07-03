"""
PFun Health - Difference Equations Integrator
This script symbolically defines the PFun CMA difference equations using SymPy 
and provides a numerical integration loop to generate physiological traces for the platform.
"""

import sympy as sp
import numpy as np

def define_difference_equations():
    """
    Defines the symbolic variables and the difference equations.
    Returns the updated symbolic expressions for G, I, and C.
    """
    # Define time-dependent state variables
    G_t, G_t_minus_1, I_t, C_t, M_t = sp.symbols('G_t G_t_minus_1 I_t C_t M_t')
    
    # Define physiological parameters
    G_base, beta, gamma = sp.symbols('G_base beta gamma')
    alpha_I, k_I = sp.symbols('alpha_I k_I')
    alpha_C, k_C = sp.symbols('alpha_C k_C')
    
    # 1. Glucose Difference Equation
    # Incorporates mean-centered sensitivity (beta) and gradient-centered sensitivity (gamma)
    G_next = G_t - beta * I_t * (G_t - G_base) - gamma * I_t * (G_t - G_t_minus_1) + C_t + M_t
    
    # 2. Insulin Difference Equation
    # Note: Max function is handled conditionally in numerical execution, 
    # but for symbolic representation we use a standard algebraic placeholder or Piecewise.
    I_next = I_t + alpha_I * sp.Piecewise((G_t - G_base, G_t > G_base), (0, True)) - k_I * I_t
    
    # 3. Counter-Regulatory (Cortisol/Glucagon) Difference Equation
    C_next = C_t + alpha_C * sp.Piecewise((G_base - G_t, G_t < G_base), (0, True)) - k_C * C_t

    return (G_next, I_next, C_next), (G_t, G_t_minus_1, I_t, C_t, M_t, G_base, beta, gamma, alpha_I, k_I, alpha_C, k_C)

def integrate_equations(timesteps=144, meal_times=None):
    """
    Numerically integrates the symbolic difference equations.
    Default timesteps=144 implies a 24-hour period mapped at 10-min intervals.
    """
    if meal_times is None:
        meal_times = {30: 40.0, 75: 60.0} # Example: Exogenous carb inputs at specific indices

    # Fetch symbolic equations
    equations, symbols = define_difference_equations()
    G_next_eq, I_next_eq, C_next_eq = equations
    G_t, G_t_minus_1, I_t, C_t, M_t, G_base, beta, gamma, alpha_I, k_I, alpha_C, k_C = symbols

    # Lambdify for fast numerical evaluation
    # This bridges SymPy's symbolic rigor with NumPy's execution speed
    eval_G = sp.lambdify(symbols, G_next_eq, modules='numpy')
    eval_I = sp.lambdify(symbols, I_next_eq, modules='numpy')
    eval_C = sp.lambdify(symbols, C_next_eq, modules='numpy')

    # Base parameter setup (Example values for a healthy phenotype)
    params = {
        'G_base': 90.0,
        'beta': 0.002,
        'gamma': 0.001,
        'alpha_I': 0.05,
        'k_I': 0.1,
        'alpha_C': 0.03,
        'k_C': 0.08
    }

    # Initialize State Arrays
    G_trace = np.zeros(timesteps)
    I_trace = np.zeros(timesteps)
    C_trace = np.zeros(timesteps)

    # Set initial conditions
    G_trace[0] = params['G_base']
    G_trace[1] = params['G_base']
    I_trace[0] = 5.0
    I_trace[1] = 5.0
    C_trace[0] = 2.0
    C_trace[1] = 2.0

    # Integration Loop
    for t in range(1, timesteps - 1):
        # Determine meal input for current timestep
        current_M = meal_times.get(t, 0.0)

        # Build execution arguments tuple
        args = (
            G_trace[t],          # G_t
            G_trace[t-1],        # G_t_minus_1
            I_trace[t],          # I_t
            C_trace[t],          # C_t
            current_M,           # M_t
            params['G_base'],
            params['beta'],
            params['gamma'],
            params['alpha_I'],
            params['k_I'],
            params['alpha_C'],
            params['k_C']
        )

        # Step forward
        G_trace[t+1] = eval_G(*args)
        I_trace[t+1] = eval_I(*args)
        C_trace[t+1] = eval_C(*args)

    return G_trace, I_trace, C_trace

if __name__ == "__main__":
    print("Initializing PFun Difference Equation Integrator...")
    glucose, insulin, cortisol = integrate_equations()
    
    print("\nIntegration Complete. 24-Hour Trace Samples (indices 30 to 40):")
    print("Time\tGlucose\t\tInsulin\t\tCortisol/Glucagon")
    for i in range(30, 40):
        print(f"{i}\t{glucose[i]:.2f}\t\t{insulin[i]:.2f}\t\t{cortisol[i]:.2f}")
    
    print("\nNext step: Connect this output trace to the IP Collaborative scenario generator.")