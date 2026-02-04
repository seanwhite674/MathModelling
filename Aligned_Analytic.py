import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

######################### Compute DTER in isotropic ECM ########################

# Enable LaTeX rendering for all text elements
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

def g(beta, phi, k): 
    return phi * (beta / k) * np.exp(-(beta / k)**2)

def p_min(theta, n_p):
    # Density of minimum of n_p i.i.d. U[0,pi]
    return n_p * (1/np.pi) * (1 - theta/np.pi)**(n_p - 1)

def a(theta):
    return (2/np.pi**2) * (np.pi - theta)

def compute_DTER(n_p, l_p, g_val, d_crit):
    if l_p < d_crit:
        return 0.0

    theta_crit = np.arccos(d_crit / l_p)

    # Detection probability for uniform angles
    P_det = 1 - (1 - theta_crit/np.pi)**n_p

    if P_det > 0:
        EA_det, _ = quad(lambda th: a(th) * p_min(th, n_p), 0, theta_crit)
        EA_det = EA_det / P_det
    else:
        EA_det = 0.0

    EA_rand, _ = quad(lambda th: a(th) * (1/np.pi), 0, np.pi)

    EA_move = P_det * EA_det + (1 - P_det) * EA_rand
    d_avg = d * EA_move

    # Numerator: E[d_x] = g * ∫_{0}^{theta_crit} d*a(theta)*cos(theta)*p_min(theta) dtheta
    num_int, _ = quad(lambda th: a(th) * np.cos(th) * p_min(th, n_p), 0, theta_crit)
    numerator = g_val * d * num_int

    denominator = (
        n_p * l_p**2
        + alpha_a * n_p**2 * l_p**2
        + alpha_m * beta * g_val * d_avg
    )

    return numerator / denominator if denominator != 0 else 0.0

######################## Detection threshold sweep #############################

def main():
    n_p_values = range(1, 11)
    l_p_values = range(1, 11)
    g_val = g(beta, phi, k)

    # Sweep d_crit from 0.5 to 9.5 by 1
    for d_crit in np.arange(0.5, 10.0, 1.0):
        DTER_matrix = np.zeros((len(n_p_values), len(l_p_values)))
        for i, n_p in enumerate(n_p_values):
            for j, l_p in enumerate(l_p_values):
                DTER_matrix[i, j] = compute_DTER(n_p, l_p, g_val, d_crit)

        # Find optimum
        max_DTER = np.max(DTER_matrix)
        max_indices = np.argwhere(DTER_matrix == max_DTER)
        optimal = max_indices[0]  
        opt_n, opt_l = n_p_values[optimal[0]], l_p_values[optimal[1]]

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            DTER_matrix,
            cmap="coolwarm",
            xticklabels=l_p_values,
            yticklabels=n_p_values,
            linewidths=0.5,
            linecolor='white',
            vmin=0,
            vmax=0.0015,
            cbar_kws={'format': '%.6f'}  
        )
        ax.invert_yaxis()

        # Mark optimum with a red dot
        plt.plot(
            opt_l - 0.5,
            opt_n - 0.5,
            marker='o',
            markersize=20,
            markerfacecolor='yellow',
            markeredgecolor='black'
        )

        # Increase axis and colorbar tick sizes; make y-ticks horizontal
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40, rotation=0)
        cax = ax.figure.axes[-1]  # colorbar axis
        cax.tick_params(labelsize=40)

        plt.xlabel(r"$l_p$", size=50)
        plt.ylabel(r"$n_p$", rotation=0, size=50, labelpad=50)
        plt.tight_layout()

        # Save to file and close
        filename = f"DTER_heatmap_dcrit_{d_crit:.1f}.png"
        plt.savefig(filename, dpi=1200)
        plt.close()

if __name__ == "__main__":
    main()
