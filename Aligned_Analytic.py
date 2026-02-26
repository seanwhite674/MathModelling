import numpy as np
from fractions import Fraction
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

######################### Compute DTER in aligned ECM ##########################

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
})

########################## Fixed model parameters ##############################

alpha_a = 1.0
alpha_m = 1.0
d = 1.0
phi = 1.0
beta = 500.0
k = 500 * np.sqrt(2)

############################# Helper functions #################################

def pi_fraction_label(x, max_den=8):
    frac = Fraction(x / np.pi).limit_denominator(max_den)

    if frac == 0:
        return r"0"
    elif frac == 1:
        return r"\pi"
    elif frac.numerator == 1:
        return rf"\pi/{frac.denominator}"
    else:
        return rf"{frac.numerator}\pi/{frac.denominator}"

def g(beta, phi, k): 
    return phi * (beta / k) * np.exp(-(beta / k)**2)

def p_min(theta, n_p):
    return n_p * (1/np.pi) * (1 - theta/np.pi)**(n_p - 1)

def a(theta, theta0=0.0, gamma=1.0):
    base = 0.5 * (1.0 + np.cos(theta - theta0))  # in [0,1]
    raw = base**gamma

    norm, _ = quad(
        lambda th: (0.5 * (1.0 + np.cos(th - theta0)))**gamma * (1/np.pi),
        0, np.pi
    )
    return raw / norm

def compute_DTER(n_p, l_p, g_val, d_crit, theta0=0.0, gamma=1.0):
    if l_p < d_crit:
        return 0.0

    theta_crit = np.arccos(d_crit / l_p)

    P_det = 1 - (1 - theta_crit/np.pi)**n_p

    if P_det > 0:
        EA_det, _ = quad(lambda th: a(th, theta0, gamma) * p_min(th, n_p), 0, 
                         theta_crit)
        EA_det = EA_det / P_det
    else:
        EA_det = 0.0

    EA_rand, _ = quad(lambda th: a(th, theta0, gamma) * (1/np.pi), 0, np.pi)

    EA_move = P_det * EA_det + (1 - P_det) * EA_rand
    d_avg = d * EA_move

    num_int, _ = quad(lambda th: a(th, theta0, gamma) * np.cos(th) * p_min(th, 
                      n_p), 0, theta_crit)

    numerator = g_val * d * num_int

    denominator = (
        n_p * l_p**2
        + alpha_a * n_p**2 * l_p**2
        + alpha_m * beta * g_val * d_avg
    )

    return numerator / denominator if denominator != 0 else 0.0

######################### Main parameter sweep #################################

def main():
    n_p_values = list(range(1, 11))
    l_p_values = list(range(1, 11))
    g_val = g(beta, phi, k)
    d_crit = 0.8

    theta0_values = np.linspace(0.0, np.pi/2, 5)
    gamma_values  = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    n_rows = len(gamma_values)
    n_cols = len(theta0_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        constrained_layout=True
    )

    vmin, vmax = 0.0, 0.0015
    last_im = None

    for r, gamma in enumerate(gamma_values):
        for c, theta0 in enumerate(theta0_values):

            ax = axes[r, c]

            DTER_matrix = np.zeros((len(n_p_values), len(l_p_values)))
            for i, n_p in enumerate(n_p_values):
                for j, l_p in enumerate(l_p_values):
                    DTER_matrix[i, j] = compute_DTER(
                        n_p, l_p, g_val, d_crit, theta0, gamma
                    )

            # Optimum
            max_idx = np.unravel_index(np.argmax(DTER_matrix), DTER_matrix.shape)
            opt_n, opt_l = max_idx

            last_im = sns.heatmap(
                DTER_matrix,
                ax=ax,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                xticklabels=l_p_values if r == n_rows - 1 else False,
                yticklabels=n_p_values if c == 0 else False,
                linewidths=0.3,
                linecolor="white"
            )

            ax.invert_yaxis()

            # Mark optimum
            ax.plot(
                opt_l + 0.5,
                opt_n + 0.5,
                marker='o',
                markersize=8,
                markerfacecolor='yellow',
                markeredgecolor='black'
            )

            if r == 0:
                ax.set_title(
                    rf"$\theta_0={pi_fraction_label(theta0)}$",
                    fontsize=14
                )


            # Row labels
            if c == 0:
                ax.set_ylabel(rf"$\gamma={gamma}$", fontsize=14)

            ax.set_xlabel("")
            ax.set_ylabel("")

    # Shared colorbar
    cbar = fig.colorbar(
        last_im.collections[0],
        ax=axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.01,
        format="%.6f"
    )
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("DTER", fontsize=16)

    fig.supxlabel(r"$l_p$", fontsize=20)
    fig.supylabel(r"$n_p$", fontsize=20)

    plt.savefig("DTER_grid_of_grids.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()