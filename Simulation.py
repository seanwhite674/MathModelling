#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

######################### Compute DTER in isotropic ECM ########################

# NOTE: enable TeX only if available on the cluster. Default False for portability.
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
})

dx = 0.0001
xs = np.arange(0, 10, dx)
x0 = 0
n_steps = 100

alpha_a = 1.0
alpha_m = 1.0
d = 1.0        
phi = 1.0
beta = 500.0
k = 500 * np.sqrt(2)

### np.savetxt


def g(beta, phi, k):
    return phi * (beta / k) * np.exp(-(beta / k)**2)


def chemotaxis_walk(x0, n_steps, dx, n_p, l_p, d_crit, d, L):

    # guard against domain errors in arccos
    ratio = float(d_crit) / float(l_p)
    if ratio >= 1.0:
        thetacrit = 0.0  # no detection possible
    elif ratio <= -1.0:
        thetacrit = np.pi
    else:
        thetacrit = np.arccos(ratio)

    x = float(x0)
    path = [x]
    nsteps = 0

    while nsteps < n_steps:
        # sample angles and choose smallest; ensure n_p >= 1
        thetarand = np.sort(np.random.uniform(0, np.pi, int(max(1, n_p))))
        theta = thetarand[0]

        if theta < thetacrit:
            x_new = x + d * np.cos(theta)
        else:
            theta = np.random.uniform(0, np.pi)
            x_new = x + d * np.cos(theta)

        x_grid = dx * np.round(x_new / dx)
        x_grid = np.clip(x_grid, 0, L)

        x = x_grid
        path.append(x)
        nsteps += 1

    return np.array(path)


def Edx(x0, n_steps, dx, n_p, l_p, d_crit, d, L):
    # Correct argument order (d_crit, d) as expected by chemotaxis_walk
    path = chemotaxis_walk(x0, n_steps, dx, n_p, l_p, d_crit, d, L)
    return path[-1] - path[0]


def Edx_mean(x0, n_steps, dx, n_p, l_p, d_crit, d, L, n_samples=10, seed=None):

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    displacements = np.zeros(int(n_samples))
    for i in range(int(n_samples)):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(sample_seed)
        displacements[i] = Edx(x0, n_steps, dx, n_p, l_p, d_crit, d, L)
    return np.mean(displacements)


def denom(n_p, l_p, g_val):
    return n_p * l_p**2 + alpha_a * n_p**2 * l_p**2 + beta * alpha_m * g_val * d


# def parse_args(argv):
#     p = argparse.ArgumentParser(
#         description="Takes in n_p, l_p and d_crit."
#     )
#     p.add_argument("--n-p", type=int, nargs=2, metavar=("MIN", "MAX"), required=True,
#                    help="n_p range: min max")
#     p.add_argument("--l-p", type=int, nargs=2, metavar=("MIN", "MAX"), required=True,
#                    help="l_p range: min max")
#     p.add_argument("--dcrit", type=float, nargs=2, metavar=("START", "STOP"), required=True,
#                    help="d_crit sweep: start stop")
#     return p.parse_args(argv[1:])


def main():

    n_p_values = range(1, 11)
    l_p_values = range(1, 11)
    g_val = g(beta, phi, k)

    n_samples = 10        # change to more samples
    seed = 42             # seed default
    L = 100.0

    for d_crit in np.arange(0.5, 10.0, 1.0):
        DTER_matrix = np.zeros((len(list(n_p_values)), len(list(l_p_values))))
        for i, n_p in enumerate(n_p_values):
            for j, l_p in enumerate(l_p_values):
                mean_Edx = Edx_mean(x0, n_steps, dx, n_p, l_p, d_crit, d, L,
                                    n_samples=n_samples, seed=(seed + i * 100 + j))
                DTER_matrix[i, j] = mean_Edx / denom(n_p, l_p, g_val)

        # Find optimum (use matrix indices)
        max_DTER = np.max(DTER_matrix)
        max_indices = np.argwhere(DTER_matrix == max_DTER)
        optimal = max_indices[0]
        opt_row, opt_col = optimal[0], optimal[1]
        opt_n = list(n_p_values)[opt_row]
        opt_l = list(l_p_values)[opt_col]

        # Plot heatmap 
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            DTER_matrix,
            cmap="coolwarm",
            xticklabels=list(l_p_values),
            yticklabels=list(n_p_values),
            linewidths=0.5,
            linecolor='white',
            #vmin=0,
            #vmax=0.0015,
            cbar_kws={'format': '%.6f'}
        )
        ax.invert_yaxis()

        # Mark optimum 
        plt.plot(
            opt_col + 0.5,
            opt_row + 0.5,
            marker='o',
            markersize=20,
            markerfacecolor='yellow',
            markeredgecolor='black')


        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40, rotation=0)
        cax = ax.figure.axes[-1]  # colorbar axis
        cax.tick_params(labelsize=40)

        plt.xlabel(r"$l_p$", size=50)
        plt.ylabel(r"$n_p$", rotation=0, size=50, labelpad=50)
        plt.tight_layout()
        plt.savefig(f"DTER_Sim_heatmap_dcrit_{d_crit:.3f}.png", dpi=50)   
        plt.close()


if __name__ == "__main__":
    main()
