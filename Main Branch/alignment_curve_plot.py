#!/usr/bin/env python3
"""
plot_alignment_weights.py

Generate and save plots of the normalized alignment weight
a(theta; theta0, gamma) = base(theta)^gamma / Z(theta0, gamma)
with base(theta) = 0.5*(1 + cos(theta - theta0)), theta in [0, pi].

Saves per-theta0 figures and one combined figure.

Requirements:
    pip install numpy matplotlib
(Optionally: scipy for quad integration if you prefer.)
"""
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import os

plt.style.use("default")

plt.rcParams.update({
    # Background
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    # Text & ticks
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",

    # Axes lines
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,

    # Grid styling (dark dashed lines)
    "grid.color": "0.2",          # dark grey
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

# ---------- Utility / formatting ----------
def pi_fraction_label(x, max_den=8):
    """Return a latex-friendly pi-fraction label for angle x (radians)."""
    frac = Fraction(x / np.pi).limit_denominator(max_den)
    if frac == 0:
        return "0"
    elif frac == 1:
        return r"\pi"
    elif frac.numerator == 1:
        return rf"\pi/{frac.denominator}"
    else:
        return rf"{frac.numerator}\pi/{frac.denominator}"

# ---------- Alignment function ----------
def alignment_weight(theta, theta0=0.0, gamma=1.0, ngrid=4000):
    """
    Compute normalized alignment weight a(theta; theta0, gamma).

    theta : array-like (points to evaluate)
    theta0 : float (preferred orientation, radians)
    gamma : float (shape parameter)
    ngrid : int (grid size for numeric normalization)
    """
    theta = np.asarray(theta)
    # base in [0,1]
    base = 0.5 * (1.0 + np.cos(theta - theta0))
    raw = base**gamma

    # numeric normalization Z = (1/pi) * integral_0^pi base^gamma dtheta
    grid = np.linspace(0.0, np.pi, ngrid)
    grid_vals = (0.5 * (1.0 + np.cos(grid - theta0)))**gamma
    Z = (1.0 / np.pi) * np.trapz(grid_vals, grid)

    # avoid division by zero (gamma extremely large or base=0 everywhere)
    if Z <= 0:
        return np.zeros_like(raw)
    return raw / Z

# ---------- Plotting ----------
def plot_for_theta0(theta0, gamma_values, theta, out_dir, show_legend=True):
    plt.figure(figsize=(8, 5))
    for gamma in gamma_values:
        y = alignment_weight(theta, theta0=theta0, gamma=gamma, ngrid=6000)
        plt.plot(theta, y, label=rf"$\gamma={gamma}$", linewidth=1.6)

    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$a(\theta;\theta_0,\gamma)$")
    plt.title(rf"Alignment weight, $\theta_0={pi_fraction_label(theta0)}$")
    plt.xlim(0, np.pi)
    xticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    plt.grid(True,linestyle="--",linewidth=0.5,alpha=0.25)
    if show_legend:
        plt.legend(loc="upper right", ncol=2)
    fname = os.path.join(out_dir, f"alignment_theta0_{int(round(theta0*180/np.pi)):02d}_deg.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    return fname

def plot_combined(theta0_values, gamma_values, theta, out_dir):
    n = len(theta0_values)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.5*n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, th0 in zip(axes, theta0_values):
        for gamma in gamma_values:
            y = alignment_weight(theta, theta0=th0, gamma=gamma, ngrid=6000)
            ax.plot(theta, y, label=rf"$\gamma={gamma}$", linewidth=1.4)
        ax.set_xlim(0, np.pi)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_ylabel(r"$a(\theta)$")
        ax.set_title(rf"$\theta_0={pi_fraction_label(th0)}$")
        ax.grid(True,linestyle="--",linewidth=0.5,alpha=0.25)

    axes[-1].set_xlabel(r"$\theta$ (rad)")
    axes[0].legend(loc="upper right", ncol=2)
    combined_fname = os.path.join(out_dir, "alignment_functions_combined.png")
    fig.suptitle("Alignment weight curves for multiple $\gamma$ and $\theta_0$", fontsize=14, y=1.02)
    fig.savefig(combined_fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return combined_fname

# ---------- Main runnable ----------
def main():
    # Output directory (created if missing)
    out_dir = "alignment_plots"
    os.makedirs(out_dir, exist_ok=True)

    # Parameter choices (edit these as you like)
    gamma_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # shape parameters
    theta0_values = [0.0, np.pi/4, np.pi/2]  # preferred orientations
    theta = np.linspace(0, np.pi, 400)

    # Generate separate plots per theta0
    saved = []
    for th0 in theta0_values:
        fname = plot_for_theta0(th0, gamma_values, theta, out_dir, show_legend=True)
        saved.append(fname)

    # Combined figure
    combined = plot_combined(theta0_values, gamma_values, theta, out_dir)
    saved.append(combined)

    print("Saved plots:")
    for s in saved:
        print("  -", s)
    print("\nDone.")

if __name__ == "__main__":
    main()