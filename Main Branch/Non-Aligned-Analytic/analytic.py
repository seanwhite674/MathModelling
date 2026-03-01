import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

######################### Compute DTER in isotropic ECM ########################

# Enable LaTeX rendering for all text elements
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40
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

def integrand(theta, n_p):
    return (1 / np.pi) * np.cos(theta) * (1 - (theta / np.pi))**(n_p - 1)

def compute_DTER(n_p, l_p, g_val, d_crit):
    if l_p < d_crit:
        return 0.0
    theta_crit = np.arccos(d_crit / l_p)
    integral, _ = quad(integrand, 0, theta_crit, args=(n_p,))
    numerator = g_val * d * n_p * integral
    denominator = n_p * l_p**2 + alpha_a * n_p**2 * l_p**2 + \
                  beta * alpha_m * g_val * d
    return numerator / denominator if denominator != 0 else 0.0

######################## Detection threshold sweep #############################

def main():

    # Output directories
    base_outdir = "Non_Aligned_Analytic_Outputs"
    heatmap_dir = os.path.join(base_outdir, "Heatmaps")

    os.makedirs(heatmap_dir, exist_ok=True)
  

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

        out_heat_fn = os.path.join(
            heatmap_dir,
            f"DTER_Sim_heatmap_dcrit_{d_crit:.1f}.png")
        
        plt.savefig(out_heat_fn)
        plt.close()

if __name__ == "__main__":
    main()
