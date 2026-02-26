"""
Coupled stiffness–alignment–strategy model with discrete inner optimisation

Overview
--------
This script simulates a two-timescale adaptive system in which:

    - Substrate stiffness β(t) evolves continuously,
    - The alignment bias parameter θ₀(t) evolves continuously,
    - A discrete strategy (n_p, l_p) is chosen optimally at each time.

At every time step the model performs a fast discrete optimisation
(inner problem), and the resulting optimal choice feeds back into a
slow dynamical system (outer problem).

Model structure
---------------

Inner problem (fast, static optimisation):
    For the current state (β, θ₀), select

        (n_p*, l_p*) ∈ discrete grids

    to maximise the performance functional

        DTER(β; n_p, l_p, θ₀).

    All β-independent and θ₀-dependent integrals are precomputed
    on demand so that evaluating DTER reduces to a grid search.

Outer problem (slow, continuous dynamics):
    The continuous variables evolve according to

        dβ/dt   = ρ₀ − μ β − η (n_p* l_p*)
        dθ₀/dt  = − κ₀ (n_p* l_p*) θ₀

    where the optimal discrete product m = n_p* l_p* couples
    strategy back to both mechanical stiffness and alignment bias.

    - Larger optimal structures increase degradation of β.
    - The same structures accelerate decay of θ₀ toward zero.

Numerics
--------
- Time integration is performed with `scipy.integrate.solve_ivp`.
- The discrete optimum is recomputed during RHS evaluations.
- θ₀-dependent integral tables are cached and rebuilt only when
  θ₀ changes sufficiently.
- After integration, optimal trajectories
      n_p*(t), l_p*(t)
  are reconstructed along the solution path.

User-adjustable sections
------------------------
- Initial conditions: beta0, theta0_0
- Time integration: t_span, max_step
- ODE parameters: rho0, mu, eta, kappa0
- Discrete grids: n_p_values, l_p_values
- Alignment sharpness: gamma

Run
---
    python ODE.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad
from scipy.integrate import solve_ivp

############################### Plot formatting ################################

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
})

############################### User controls ##################################

# Time integration
beta0 = 1000.0
t_span = (0.0, 100.0)
max_step = 1.0

# Discrete search grids
n_p_values = list(range(1, 11))
l_p_values = list(range(1, 11))

# Alignment parameters 
gamma = 10.0

############################ Fixed model parameters ############################

alpha_a = 1.0
alpha_m = 1.0
d = 1.0
phi = 1.0
k = 500.0 * np.sqrt(2)
d_crit = 3.5

############################# Beta ODE parameters ##############################

# d beta / dt = rho0 - mu * beta - eta * (n_p* l_p*)
rho0 = 50
mu = 0.1
eta = 1

############################# Theta ODE parameters #############################

# d theta0 / dt = - kappa * (n_p* l_p*) * theta0
kappa0 = 1e-2   
theta0_0 = np.pi/2

############################### Helper functions ###############################

def g_of_beta(beta: float, phi: float, k: float) -> float:
    """Stiffness-dependent prefactor g(beta)."""
    x = beta / k
    return phi * x * np.exp(-x**2)

def p_min(theta: float, n_p: int) -> float:
    """Minimum-angle density for n_p independent samples (as defined in model)."""
    return n_p * (1.0 / np.pi) * (1.0 - theta / np.pi) ** (n_p - 1)

def a(theta: float, theta0: float = 0.0, gamma: float = 1.0) -> float:
    """
    Angular alignment density a(theta), normalized over [0, pi] with weight 1/pi.
    """
    base = 0.5 * (1.0 + np.cos(theta - theta0))  # in [0, 1]
    raw = base ** gamma

    norm, _ = quad(
        lambda th: (0.5 * (1.0 + np.cos(th - theta0))) ** gamma * (1.0 / np.pi),
        0.0, np.pi
    )
    return raw / norm

############################# Precomputation ###################################

def precompute_tables(
    n_p_values,
    l_p_values,
    d_crit,
    theta0,
    gamma,
    alpha_a,
    d,
):
    """
    Precompute beta-independent quantities so evaluating DTER(beta) is cheap.

    For each pair (n_p, l_p), compute:
        - A     : d * ∫ a(theta) * cos(theta) * p_min(theta, n_p) dtheta on [0, theta_crit]
        - d_avg : d * E[a_move] (beta-independent)
        - B     : n_p*l_p^2 + alpha_a*n_p^2*l_p^2
        - valid : mask for l_p >= d_crit

    Then:
        DTER(beta) = ( g(beta) * A ) / ( B + alpha_m * beta * g(beta) * d_avg )
    """
    nN = len(n_p_values)
    nL = len(l_p_values)

    A = np.zeros((nN, nL), dtype=float)
    d_avg = np.zeros((nN, nL), dtype=float)
    B = np.zeros((nN, nL), dtype=float)
    valid = np.zeros((nN, nL), dtype=bool)

    EA_rand, _ = quad(lambda th: a(th, theta0, gamma) * (1.0 / np.pi), 0.0, np.pi)

    for i, n_p in enumerate(n_p_values):
        for j, l_p in enumerate(l_p_values):

            # Structural cost term (beta-independent)
            B[i, j] = n_p * l_p**2 + alpha_a * (n_p**2) * (l_p**2)

            # Enforce detection threshold
            if l_p < d_crit:
                continue

            valid[i, j] = True
            theta_crit = np.arccos(d_crit / l_p)

            # Detection probability
            P_det = 1.0 - (1.0 - theta_crit / np.pi) ** n_p

            # Conditional expectation under detection
            if P_det > 0.0:
                EA_det, _ = quad(
                    lambda th: a(th, theta0, gamma) * p_min(th, n_p),
                    0.0,
                    theta_crit,
                )
                EA_det /= P_det
            else:
                EA_det = 0.0

            # Mixture of detected vs. random outcomes
            EA_move = P_det * EA_det + (1.0 - P_det) * EA_rand
            d_avg[i, j] = d * EA_move

            # Numerator integral
            num_int, _ = quad(
                lambda th: a(th, theta0, gamma) * np.cos(th) * p_min(th, n_p),
                0.0,
                theta_crit,
            )
            A[i, j] = d * num_int

    return A, d_avg, B, valid

############################ Inner optimisation ################################

def optimal_pair(
    beta,
    A,
    d_avg,
    B,
    valid,
    n_p_values,
    l_p_values,
    alpha_m,
    phi,
    k,
):
    """Return (n_p*, l_p*, DTER*) for the current beta."""
    g_val = g_of_beta(beta, phi, k)

    denom = B + alpha_m * beta * g_val * d_avg

    DTER = np.full_like(A, -np.inf, dtype=float)
    DTER[valid] = (g_val * A[valid]) / denom[valid]

    i, j = np.unravel_index(np.argmax(DTER), DTER.shape)
    return n_p_values[i], l_p_values[j], DTER[i, j]

############################### Simulation #####################################

def simulate(beta0, theta0_0, t_span, max_step):
    # Cache for theta0-dependent tables
    cache = {
        "theta0": None,
        "A": None,
        "d_avg": None,
        "B": None,
        "valid": None,
    }
    def build_tables(theta0_val):
        A, d_avg, B, valid = precompute_tables(
            n_p_values=n_p_values,
            l_p_values=l_p_values,
            d_crit=d_crit,
            theta0=theta0_val,
            gamma=gamma,
            alpha_a=alpha_a,
            d=d,
        )
        cache["theta0"] = theta0_val
        cache["A"] = A
        cache["d_avg"] = d_avg
        cache["B"] = B
        cache["valid"] = valid

    # Build initial tables
    build_tables(theta0_0)

    # Threshold for rebuilding tables
    dtheta_rebuild = 1e-3  

    def rhs(t, y):
        beta = float(y[0])
        theta0_val = float(y[1])

        # Keep theta0 in [0, pi] 
        theta0_val = np.clip(theta0_val, 0.0, np.pi)

        # Rebuild tables if theta0 has drifted enough
        if cache["theta0"] is None or abs(theta0_val - cache["theta0"]) > dtheta_rebuild:
            build_tables(theta0_val)

        A = cache["A"]; d_avg = cache["d_avg"]; B = cache["B"]; valid = cache["valid"]

        n_star, l_star, _ = optimal_pair(
            beta=max(beta, 0.0),
            A=A, d_avg=d_avg, B=B, valid=valid,
            n_p_values=n_p_values, l_p_values=l_p_values,
            alpha_m=alpha_m, phi=phi, k=k,
        )

        m = n_star * l_star

        dbeta = rho0 - mu * max(beta, 0.0) - eta * m

        # theta0' = -kappa(m) * theta0; choose kappa(m) = kappa0 * m
        dtheta0 = -(kappa0 * m) * theta0_val

        # Clamp beta at zero 
        if beta <= 0.0 and dbeta < 0.0:
            dbeta = 0.0

        return [dbeta, dtheta0]

    sol = solve_ivp(
        rhs,
        t_span,
        [beta0, theta0_0],
        max_step=max_step,
        dense_output=False,
    )

    t = sol.t
    beta_t = sol.y[0]
    theta0_t = sol.y[1]

    # Reconstruct optimal (n_p*, l_p*) along trajectory 
    n_star_t = np.zeros_like(beta_t, dtype=int)
    l_star_t = np.zeros_like(beta_t, dtype=int)

    for idx, (b, th0) in enumerate(zip(beta_t, theta0_t)):
        A, d_avg, B, valid = precompute_tables(
            n_p_values=n_p_values,
            l_p_values=l_p_values,
            d_crit=d_crit,
            theta0=float(np.clip(th0, 0.0, np.pi)),
            gamma=gamma,
            alpha_a=alpha_a,
            d=d,
        )
        n_s, l_s, _ = optimal_pair(
            beta=float(max(b, 0.0)),
            A=A, d_avg=d_avg, B=B, valid=valid,
            n_p_values=n_p_values, l_p_values=l_p_values,
            alpha_m=alpha_m, phi=phi, k=k,
        )
        n_star_t[idx] = n_s
        l_star_t[idx] = l_s

    return t, beta_t, theta0_t, n_star_t, l_star_t, sol

################################## Main ########################################

def main():
    t, beta_t, theta0_t, n_star_t, l_star_t, _ = simulate(beta0, theta0_0, t_span, max_step)

    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    ax[0].plot(t, beta_t)
    ax[0].set_ylabel(r"$\beta(t)$")

    ax[1].plot(t, theta0_t)
    ax[1].set_ylabel(r"$\theta_0(t)$")

    ax[2].step(t, n_star_t, where="post", label=r"$n_p^\star(t)$")
    ax[2].step(t, l_star_t, where="post", label=r"$l_p^\star(t)$")
    ax[2].set_ylabel("optimum")
    ax[2].legend()

    ax[3].step(t, n_star_t * l_star_t, where="post")
    ax[3].set_ylabel(r"$n_p^\star\, l_p^\star$")
    ax[3].set_xlabel("time")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
