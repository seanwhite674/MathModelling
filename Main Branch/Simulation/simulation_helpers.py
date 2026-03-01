# analysis_helpers.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from Simulation import Edx_mean, Energy, g, beta, phi, k

# ---------------------------------------------------------------------
# 1) Histogram + summary text for representative displacements array
# ---------------------------------------------------------------------
def plot_displacement_histogram(displacements, title=None, bins=60, show_kde=True, n_steps = 25, outpath = None):
    """ 
    Plot histogram + KDE and overlay summary stats.
    `displacements` should be produced by Edx_mean(..., return_samples=True) or collect_displacements(...).
    """
    mu = float(np.mean(displacements))
    med = float(np.median(displacements))
    sigma = float(np.std(displacements, ddof=1))
    skew = float(stats.skew(displacements))
    kurt = float(stats.kurtosis(displacements))
    effective_drift = mu / n_steps  # make sure n_steps is in scope or passed in


    plt.figure(figsize=(6, 4), dpi=150)
    sns.histplot(displacements, bins=bins, kde=show_kde, stat="probability")
    plt.axvline(mu, color="red", linestyle="--", label=f"mean={mu:.4f}")
    plt.axvline(med, color="green", linestyle=":", label=f"median={med:.4f}")

    # Legend: move it out of the data region
    plt.plot([], [], ' ', label=f"eff. drift = {effective_drift:.3f}")
    plt.legend(loc="upper left", frameon=True)
    plt.title(title or "Histogram of Δx samples")
    plt.xlabel("Δx")
    plt.ylabel("Probability")
    
    # Light gridlines (paper-friendly)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    
    # Stats box: keep top-right, but slightly inset
    plt.gca().text(
    0.97,
    0.95,
    f"n={len(displacements)}\n"
    f"mean={mu:.4f}\n"
    f"std={sigma:.4f}\n"
    f"skew={skew:.2f}\n"
    f"ex.kurt={kurt:.2f}",
    transform=plt.gca().transAxes,
    va="top",
    ha="right",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),)
    
    if outpath is not None:
        plt.savefig(outpath, dpi=200)
    
    plt.tight_layout()
    plt.show()

    
# ---------------------------------------------------------------------
# 2) Compute grid stats (mean, std, sem, snr, DTER) using Edx_mean(...) with return_samples
# ---------------------------------------------------------------------
def compute_grid_statistics(n_p_values, l_p_values, d_crit,
                            x0, y0, n_steps, dx, d, L,
                            n_samples=2000, master_seed=42, g_val=None):
    """
    Compute per-(n_p, l_p) matrices:
      - mean_mat: sample mean of Edx
      - std_mat: sample std (ddof=1)
      - sem_mat: std / sqrt(n_samples)
      - snr_mat: mean / std
      - DTER_mat: mean / Energy(n_p, l_p, g_val)

    Uses Edx_mean(..., seed=..., return_samples=True) to get raw samples.
    """
    if g_val is None:
        g_val = g(beta, phi, k)

    n_rows = len(n_p_values)
    n_cols = len(l_p_values)

    mean_mat = np.full((n_rows, n_cols), np.nan)
    std_mat = np.full((n_rows, n_cols), np.nan)
    sem_mat = np.full((n_rows, n_cols), np.nan)
    snr_mat = np.full((n_rows, n_cols), np.nan)
    DTER_mat = np.full((n_rows, n_cols), np.nan)

    grid_master = np.random.default_rng(master_seed)

    for i, n_p in enumerate(n_p_values):
        for j, l_p in enumerate(l_p_values):
            if l_p < d_crit:
                # detection impossible per your logic; leave NaN
                continue
            per_cell_seed = int(grid_master.integers(0, 2**31 - 1))
            # NOTE: use 'seed=' and request samples from Edx_mean
            displacements, mean_disp = Edx_mean(
                x0, y0, n_steps, dx, n_p, l_p, d_crit, d, L,
                n_samples=n_samples, seed=per_cell_seed, return_samples=True
            )

            sigma = float(np.std(displacements, ddof=1))
            sem = sigma / np.sqrt(len(displacements))
            mu = float(mean_disp)  # equal to np.mean(displacements) but use returned mean

            mean_mat[i, j] = mu
            std_mat[i, j] = sigma
            sem_mat[i, j] = sem
            snr_mat[i, j] = (mu / sigma) if (sigma > 0 and not np.isnan(sigma)) else np.nan
            DTER_mat[i, j] = mu / Energy(n_p, l_p, g_val)

    return {
        "mean": mean_mat,
        "std": std_mat,
        "sem": sem_mat,
        "snr": snr_mat,
        "DTER": DTER_mat
    }

# -----------------------
# 3) Parametric (z-based) 95% CI for the mean
# -----------------------
def ci_parametric(displacements, alpha=0.05):
    """
    Return (mean, sem, ci_low, ci_high) using normal approx:
      mean +/- z_{1-alpha/2} * sem
    Assumes displacements is a 1D numpy array.
    """
    x = np.asarray(displacements)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 samples for CI")
    mean = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    sem = sigma / np.sqrt(n)
    z = abs(np.round(np.sqrt(2) * np.erfcinv(alpha), 8)) if hasattr(np, "erfcinv") else 1.96
    # fallback to 1.96 if erfcinv not available
    if not np.isfinite(z):
        z = 1.96
    ci_low = mean - z * sem
    ci_high = mean + z * sem
    return mean, sem, ci_low, ci_high


# -----------------------
# 4) Bootstrap CI for the mean
# -----------------------
def bootstrap_ci_mean(displacements, n_boot=2000, alpha=0.05, rng_seed=12345, return_samples=False):
    """
    Nonparametric bootstrap CI for the sample mean.
    Returns (mean, ci_low, ci_high) by default, or (mean, ci_low, ci_high, boot_means) if return_samples.
    Uses simple percentile bootstrap.
    """
    x = np.asarray(displacements)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 samples for bootstrap")
    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    mean = float(x.mean())
    if return_samples:
        return mean, lo, hi, boot_means
    return mean, lo, hi


# -----------------------
# 5) Convergence plot (mean ± SEM vs sample size)
# -----------------------
def plot_convergence(displacements, n_steps, n_list=None, rng_seed=42, outpath=None, show=True):
    """
    Plot mean +/- SEM for increasing sample sizes.
    - displacements: 1D numpy array of available samples (should be >= max(n_list))
    - n_steps: number of simulation steps (used to compute effective drift if desired)
    - n_list: list of sample sizes to evaluate, default [50,100,200,500,1000] trimmed to available samples
    - outpath: if provided, save figure to this path (string or Pathlike)
    - show: if True, call plt.show()
    """
    x = np.asarray(displacements)
    N_available = x.size
    if n_list is None:
        n_list = [50, 100, 200, 500, 1000]
    # trim and filter
    n_list = [int(n) for n in n_list if int(n) <= N_available and int(n) > 1]
    if len(n_list) == 0:
        raise ValueError(f"No n_list entries <= available samples ({N_available})")

    rng = np.random.default_rng(rng_seed)
    means = []
    sems = []
    for n in n_list:
        # sample *without replacement* from the available displacements to simulate incremental draws
        subset = rng.choice(x, size=n, replace=False)
        mu = float(np.mean(subset))
        sigma = float(np.std(subset, ddof=1))
        sem = sigma / np.sqrt(n)
        means.append(mu)
        sems.append(sem)

    means = np.array(means)
    sems = np.array(sems)

    plt.figure(figsize=(6.5, 3.8), dpi=150)
    plt.errorbar(n_list, means, yerr=sems, fmt='-o', capsize=4, label='mean ± SEM')
    plt.xlabel("n_samples")
    plt.ylabel(r"$\langle \Delta x \rangle$")
    plt.title(f"Convergence of mean Δx (opt) — n_steps={n_steps}")
    plt.grid(True, linestyle='--', alpha=0.4)
    # also plot per-step drift if useful
    v_eff = float(x.mean()) / float(n_steps)
    plt.axhline(x.mean(), color='gray', linestyle=':', label=f"full-sample mean={x.mean():.3f}")
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    return n_list, means, sems, v_eff
