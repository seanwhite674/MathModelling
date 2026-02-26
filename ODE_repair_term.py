"""
Couple substrate stiffness beta(t) to an optimal discrete choice (n_p, l_p).

Model structure
---------------
Inner problem (fast):
    For the current beta, select (n_p*, l_p*) on discrete grids
    to maximise DTER(beta; n_p, l_p).

Outer problem (slow):
    Evolve beta via an ODE with degradation proportional to the
    currently optimal product n_p* l_p*:

        d beta / dt = rho0 - mu * beta - eta * (n_p* l_p*)

Run
---
    python ODE.py

Edit
----
    - beta0, t_span, max_step
    - ODE parameters: rho0, mu, eta
    - discrete grids: n_p_values, l_p_values
    - alignment parameters: theta0, gamma
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
beta0 = 100000.0
t_span = (0.0, 1000.0)
max_step = 0.01 

# Discrete search grids
n_p_values = list(range(1, 11))
l_p_values = list(range(1, 11))

# Alignment parameters 
theta0 = 0.0
gamma = 0.0


############################ Fixed model parameters ############################

alpha_a = 1.0
alpha_m = 1.0
d = 1.0
phi = 1.0
k = 500.0 * np.sqrt(2)
d_crit = 3.5

############################# Beta ODE parameters ##############################

# d beta / dt = rho0 - mu * beta - eta * (n_p* l_p*)


k_rep = 0.1
k_deg = 0.01

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

    # Random-alignment expectation depends only on (theta0, gamma)
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

def simulate(beta0, t_span, max_step):
    A, d_avg, B, valid = precompute_tables(
        n_p_values=n_p_values,
        l_p_values=l_p_values,
        d_crit=d_crit,
        theta0=theta0,
        gamma=gamma,
        alpha_a=alpha_a,
        d=d,
    )

    # Store dbeta values for inspection
    dbeta_history = []
    time_history = []
    step_count = [0]  # Initialize step counter

    def rhs(t, y):
        beta = float(y[0])

        n_star, l_star, _ = optimal_pair(
            beta=max(beta, 0.0),
            A=A, d_avg=d_avg, B=B, valid=valid,
            n_p_values=n_p_values, l_p_values=l_p_values,
            alpha_m=alpha_m, phi=phi, k=k,
        )

        dbeta = k_rep * (beta0 - max(beta, 0.0)) - k_deg * (n_star * l_star)

        # Store for later inspection
        dbeta_history.append(dbeta)
        time_history.append(t)

        # Print only every 1000 steps
        step_count[0] += 1
        if step_count[0] % 1000 == 0:
            print(f"t={t:.2f}, beta={beta:.2f}, n*={n_star}, l*={l_star}, dbeta={dbeta:.6f}")

        # Clamp at zero (no flow into negatives)
        if beta <= 0.0 and dbeta < 0.0:
            dbeta = 0.0     
        return [dbeta]

    sol = solve_ivp(
        rhs,
        t_span,
        [beta0],
        max_step=max_step,
        dense_output=False,
    )

    t = sol.t
    beta_t = sol.y[0]

    # Reconstruct optimal (n_p*, l_p*) along the trajectory
    n_star_t = np.zeros_like(beta_t, dtype=int)
    l_star_t = np.zeros_like(beta_t, dtype=int)

    for idx, b in enumerate(beta_t):
        n_s, l_s, _ = optimal_pair(
            beta=float(max(b, 0.0)),
            A=A,
            d_avg=d_avg,
            B=B,
            valid=valid,
            n_p_values=n_p_values,
            l_p_values=l_p_values,
            alpha_m=alpha_m,
            phi=phi,
            k=k,
        )
        n_star_t[idx] = n_s
        l_star_t[idx] = l_s

    return t, beta_t, n_star_t, l_star_t, sol, np.array(dbeta_history), np.array(time_history)


def main():
    t, beta_t, n_star_t, l_star_t, sol, dbeta_hist, t_hist = simulate(beta0, t_span, max_step)

    # Limit to first 1000 values
    limit = min(1000, len(dbeta_hist))
    dbeta_hist_limited = dbeta_hist[:limit]
    t_hist_limited = t_hist[:limit]

    fig, ax = plt.subplots(4, 1, figsize=(9, 10))

    ax[0].plot(t, beta_t)
    ax[0].set_ylabel(r"$\beta(t)$")

    ax[1].step(t, n_star_t, where="post", label=r"$n_p^\star(t)$")
    ax[1].step(t, l_star_t, where="post", label=r"$l_p^\star(t)$")
    ax[1].set_ylabel("optimum")
    ax[1].legend()

    ax[2].step(t, n_star_t * l_star_t, where="post")
    ax[2].set_ylabel(r"$n_p^\star\, l_p^\star$")
    ax[2].set_xlabel("time")

    # Plot only first 1000 dbeta values
    ax[3].plot(t_hist_limited, dbeta_hist_limited, 'r-', linewidth=1.5)
    ax[3].set_ylabel(r"$\frac{d\beta}{dt}$")
    ax[3].set_xlabel("time")
    ax[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax[3].set_xlim([t_hist_limited[0], t_hist_limited[-1]])  # Focus on first 1000 steps

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()