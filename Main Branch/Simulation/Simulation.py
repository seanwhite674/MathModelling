#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import warnings

######################### Compute DTER in isotropic ECM ########################

# NOTE: enable TeX only if available on the cluster. Default False for portability.
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
})

alpha_a = 1.0
alpha_m = 1.0
d = 1.0
phi = 1.0
beta = 500.0
k = 500 * np.sqrt(2)


def g(beta, phi, k):
    return phi * (beta / k) * np.exp(-(beta / k) ** 2)


def chemotaxis_walk(x0, y0, n_steps, dx, n_p, l_p, d_crit, d, L, rng=None):
    """
    RNG-aware chemotaxis walk.

    Angles sampled in [0, pi] to match the paper symmetry.
    rng: np.random.Generator (if None, a new one is created)
    """
    if rng is None:
        rng = np.random.default_rng()

    if l_p <= 0:
        raise ValueError("l_p must be > 0")

    ratio = float(d_crit) / float(l_p)
    # Paper-consistent handling: if l_p < d_crit, gradient never detectable
    if ratio >= 1.0:
        warnings.warn(
            f"d_crit/l_p = {ratio:.6g} >= 1 -> no gradient detection possible; "
            "setting thetacrit = 0 (movement becomes random)."
        )
        thetacrit = 0.0
    elif ratio <= -1.0:
        warnings.warn(
            f"d_crit/l_p = {ratio:.6g} <= -1 -> setting thetacrit = pi (always biased)."
        )
        thetacrit = np.pi
    else:
        thetacrit = np.arccos(ratio)

    x, y = float(x0), float(y0)
    xpath = [x]
    xypath = [(x, y)]

    for _ in range(int(n_steps)):
        # sample candidate angles using the provided RNG and pick the best (minimum)
        thetarand = np.sort(rng.uniform(0.0, np.pi, int(max(1, n_p))))
        theta = float(thetarand[0])

        if theta < thetacrit:
            x_new = x + d * np.cos(theta)
            y_new = y + d * np.sin(theta)
        else:
            theta = float(rng.uniform(0.0, np.pi))
            x_new = x + d * np.cos(theta)
            y_new = y + d * np.sin(theta)

        # snap to grid
        x_grid = dx * np.round(x_new / dx)
        y_grid = dx * np.round(y_new / dx)

        x, y = x_grid, y_grid
        xpath.append(x)
        xypath.append((x, y))

    return np.array(xpath), np.array(xypath)


def Edx(x0, y0, n_steps, dx, n_p, l_p, d_crit, d, L, rng=None):
    xpath, _ = chemotaxis_walk(x0, y0, n_steps, dx, n_p, l_p, d_crit, d, L, rng=rng)
    return xpath[-1] - xpath[0]


def Edx_mean(x0, y0, n_steps, dx, n_p, l_p, d_crit, d, L, n_samples=10, seed=None,return_samples = False):
    """
    Monte Carlo mean of Edx using independent child RNGs.
    seed: integer used to construct master RNG (deterministic)
    """
    master_rng = np.random.default_rng(seed)
    displacements = np.zeros(int(n_samples))
    for i in range(int(n_samples)):
        child_seed = int(master_rng.integers(0, 2**31 - 1))
        child_rng = np.random.default_rng(child_seed)
        displacements[i] = Edx(x0, y0, n_steps, dx, n_p, l_p, d_crit, d, L, rng=child_rng)

    Displacements_mean = float(np.mean(displacements))

    if return_samples:
        return displacements, Displacements_mean
    else:
        return Displacements_mean


def Energy(n_p, l_p, g_val):
    return n_p * l_p ** 2 + alpha_a * n_p ** 2 * l_p ** 2 + beta * alpha_m * g_val * d

def main(): 
    # Output directories
    base_outdir = "Simulation_Outputs"
    heatmap_dir = os.path.join(base_outdir, "Heatmaps")
    traj_dir = os.path.join(base_outdir, "Trajectories")

    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    n_p_values = range(1, 11)
    l_p_values = range(1, 11)
    g_val = g(beta, phi, k)

    n_samples = 100       # change to more samples
    seed = 42             # seed default
    L = 10000.0
    dx = 0.0001
    x0 = 0.0
    y0 = 0.0
    n_steps = 40

    # top-level RNG for reproducible experiment-level randomness
    master_rng = np.random.default_rng(seed)

    for d_crit in np.arange(0.5, 10.0, 1.0):
        DTER_matrix = np.full(
            (len(n_p_values), len(l_p_values)),
            np.nan,
            dtype=float
        )

        for i, n_p in enumerate(n_p_values):
            for j, l_p in enumerate(l_p_values):
                # Paper-consistent: if l_p < d_crit, detection impossible
                if l_p < d_crit:
                    continue

                # deterministic seed per (i, j)
                per_case_seed = int(master_rng.integers(0, 2**31 - 1))
                mean_Edx = Edx_mean(
                    x0, y0, n_steps, dx,
                    n_p, l_p, d_crit, d, L,
                    n_samples=n_samples,
                    seed=per_case_seed
                )
                DTER_matrix[i, j] = mean_Edx / Energy(n_p, l_p, g_val)

        # Find optimum over valid cells
        if np.all(np.isnan(DTER_matrix)):
            print(f"No valid (n_p, l_p) for d_crit={d_crit:.3f}; skipping plotting.")
            continue

        max_DTER = np.nanmax(DTER_matrix)
        max_indices = np.argwhere(DTER_matrix == max_DTER)
        opt_row, opt_col = max_indices[0]  # take first if tie
        opt_n = list(n_p_values)[opt_row]
        opt_l = list(l_p_values)[opt_col]

        # Heatmap
        plt.figure(figsize=(8, 6), dpi=150)
        mask = np.isnan(DTER_matrix)
        ax = sns.heatmap(
            DTER_matrix,
            cmap="coolwarm",
            mask=mask,
            xticklabels=list(l_p_values),
            yticklabels=list(n_p_values),
            linewidths=1,
            linecolor="white",
            cbar_kws={"format": "%.6f"},
            square=False
        )
        plt.gcf().set_dpi(200)
        ax.invert_yaxis()

        # Mark optimum
        ax.scatter(
            opt_col + 0.5,
            opt_row + 0.5,
            s=200,
            c="yellow",
            edgecolor="k",
            marker="o"
        )

        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12, rotation=0)
        cax = ax.figure.axes[-1]
        cax.tick_params(labelsize=10)

        plt.xlabel(r"$l_p$", size=14)
        plt.ylabel(r"$n_p$", rotation=0, size=14, labelpad=20)
        plt.title(f"DTER / Energy (d_crit={d_crit:.1f})", size=14)
        plt.tight_layout()

        out_heat_fn = os.path.join(
            heatmap_dir,
            f"DTER_Sim_heatmap_dcrit_{d_crit:.1f}.png"
        )
        plt.savefig(out_heat_fn)
        plt.close()

        # Sample trajectory
        traj_seed = int(seed + opt_row * 100 + opt_col)
        traj_rng = np.random.default_rng(traj_seed)
        _, xypath = chemotaxis_walk(
            x0, y0, n_steps, dx,
            opt_n, opt_l, d_crit, d, L,
            rng=traj_rng
        )

        plt.figure(figsize=(8, 4), dpi=150)
        plt.plot(xypath[:, 0], xypath[:, 1], "-o", markersize=3)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title(
            f"Trajectory (d_crit={d_crit:.1f}, n_p={opt_n}, l_p={opt_l})"
        )
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out_traj_fn = os.path.join(
            traj_dir,
            f"Sample_trajectory_dcrit_{d_crit:.1f}_np_{opt_n}_lp_{opt_l}.png"
        )
        plt.savefig(out_traj_fn)
        plt.close()

        print(
            f"Saved heatmap {out_heat_fn} and trajectory {out_traj_fn} "
            f"(opt n={opt_n}, l={opt_l})"
        )


if __name__ == "__main__":
    main()
