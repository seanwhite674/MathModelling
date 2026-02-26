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
beta = 1000.0
t_span = (0.0, 1000.0)
max_step = 0.01 

# Discrete search grids
n_p_values = list(range(1, 11))
l_p_values = list(range(1, 11))

# Alignment parameters 
theta0init = np.pi/2
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
rho0 = 5
mu = 0.1
eta = 1

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



############################ Inner optimisation ################################

def optimal_pair(
    beta,
    theta0,
    n_p_values,
    l_p_values,
    alpha_m,
    phi,
    k,
):
    """Return (n_p*, l_p*, DTER*) for the current beta."""
    g_val = g_of_beta(beta, phi, k)

    ###   compute A and davg  
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

    denom = B + alpha_m * beta * g_val * d_avg

    DTER = np.full_like(A, -np.inf, dtype=float)
    DTER[valid] = (g_val * A[valid]) / denom[valid]

    i, j = np.unravel_index(np.argmax(DTER), DTER.shape)
    return n_p_values[i], l_p_values[j], DTER[i, j]


############################### Simulation #####################################

def simulate(theta0init, t_span, max_step):
    dtheta_history = []
    time_history = []
    step_count = [0]
    
    def rhs(t, y):
        theta0 = float(y[0])

        # Calculate A,B
        dtheta0 = -0.01


        dtheta_history.append(dtheta0)
        time_history.append(t)

        step_count[0] += 1
        if step_count[0] % 1000 == 0:
            print(f"t={t:.2f}, theta0={theta0:.2f}, dtheta0={dtheta0:.6f}")

        # Clamp at zero (no flow into negatives)
        if theta0 <= 0.0 and dtheta0 < 0.0:
            dtheta0 = 0.0
        
        return [dtheta0]

    sol = solve_ivp(
        rhs,
        t_span,
        [theta0init],
        max_step=max_step,
        dense_output=False,
    )

    t = sol.t
    theta_t = sol.y[0]

    # Reconstruct optimal (n_p*, l_p*) along the trajectory
    n_star_t = np.zeros_like(theta_t, dtype=int)
    l_star_t = np.zeros_like(theta_t, dtype=int)

    for idx, th in enumerate(theta_t):
        if idx % 1000 == 0:
            print("reconstruct idx", idx, "t", t[idx], "theta", th)
        n_s, l_s, _ = optimal_pair(
            beta=beta,
            theta0=float(max(th, 0.0)),
            n_p_values=n_p_values,
            l_p_values=l_p_values,
            alpha_m=alpha_m,
            phi=phi,
            k=k,
        )
        n_star_t[idx] = n_s
        l_star_t[idx] = l_s

    return t, theta_t, n_star_t, l_star_t, sol

################################## Main ########################################

def main():
    t, theta_t, n_star_t, l_star_t, _ = simulate(theta0init, t_span, max_step)

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    ax[0].plot(t, theta_t)
    ax[0].set_ylabel(r"$\theta(t)$")

    ax[1].step(t, n_star_t, where="post", label=r"$n_p^\star(t)$")
    ax[1].step(t, l_star_t, where="post", label=r"$l_p^\star(t)$")
    ax[1].set_ylabel("optimum")
    ax[1].legend()

    ax[2].step(t, n_star_t * l_star_t, where="post")
    ax[2].set_ylabel(r"$n_p^\star\, l_p^\star$")
    ax[2].set_xlabel("time")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
