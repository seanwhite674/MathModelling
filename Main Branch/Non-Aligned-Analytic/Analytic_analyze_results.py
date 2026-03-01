# analyze_analytic.py
"""
Top-level analytic grid analysis.
Assumes your analytic code is in analytic.py and exposes:
   - compute_DTER(n_p, l_p, g_val, d_crit)
   - g(beta, phi, k)
   - beta, phi, k (optional; g can be computed from sim if desired)

Optional: provide a Monte-Carlo CSV/NumPy file path via `mc_grid_path` to compare analytic -> MC.
MC grid file format assumed to be a CSV with shape (n_rows x n_cols) or a saved .npy array with same shape.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

import analytic         # your analytic file (must be importable)
import Analytic_helpers as ah

# settings (tweak here or via CLI)
DEFAULT_NP = list(range(1, 11))
DEFAULT_LP = list(range(1, 11))
DEFAULT_DCRIT = 2.0

OUTDIR = Path("Analytic_Outputs")


def compute_analytic_grid(n_p_values, l_p_values, d_crit):
    """
    Returns (DTER_matrix, n_p_values, l_p_values)
    """
    g_val = analytic.g(analytic.beta, analytic.phi, analytic.k) if hasattr(analytic, "g") else None
    n_rows = len(n_p_values)
    n_cols = len(l_p_values)
    mat = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, n_p in enumerate(n_p_values):
        for j, l_p in enumerate(l_p_values):
            try:
                mat[i, j] = analytic.compute_DTER(n_p, l_p, g_val, d_crit)
            except Exception as e:
                mat[i, j] = np.nan
                print(f"Warning: compute_DTER failed for n_p={n_p}, l_p={l_p}: {e}")
    return mat


def load_mc_grid(path, expected_shape=None):
    """
    Load Monte-Carlo grid from CSV or .npy
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.suffix.lower() in [".npy"]:
        arr = np.load(p)
    else:
        # try CSV via pandas (handles headerless numeric CSV)
        df = pd.read_csv(p, header=None)
        arr = df.values
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f"MC grid shape {arr.shape} != expected {expected_shape}")
    return arr


def main(n_p_values=DEFAULT_NP, l_p_values=DEFAULT_LP, d_crit=DEFAULT_DCRIT,
         outdir=OUTDIR, mc_grid_path=None, vmin=None, vmax=None):
    outdir = Path(outdir)
    ah.ensure_dir(outdir)
    heat_dir = outdir / "Heatmaps"
    ah.ensure_dir(heat_dir)

    print("Computing analytic grid...")
    DTER_analytic = compute_analytic_grid(n_p_values, l_p_values, d_crit)

    # Save numeric grid to CSV and npy
    np.save(outdir / f"dter_analytic_dcrit_{d_crit:.1f}.npy", DTER_analytic)
    pd.DataFrame(DTER_analytic).to_csv(outdir / f"dter_analytic_dcrit_{d_crit:.1f}.csv", index=False, header=False)

    # Heatmap of analytic DTER
    xticks = list(l_p_values)
    yticks = list(n_p_values)
    heat_fn = heat_dir / f"dter_analytic_heatmap_dcrit_{d_crit:.1f}.png"
    ah.save_heatmap(DTER_analytic, heat_fn,
                    title=f"Analytic DTER (d_crit={d_crit:.1f})",
                    xticklabels=xticks, yticklabels=yticks,
                    vmin=vmin, vmax=vmax)
    print(f"Saved analytic heatmap to {heat_fn}")

    # Find optima
    idx_list, vals = ah.find_optima(DTER_analytic, n_p_values=n_p_values, l_p_values=l_p_values)
    if len(idx_list) > 0:
        print(f"Analytic optimum: grid index {idx_list[0]}, (n_p,l_p) = {vals[0]}")
    else:
        print("No valid analytic cells (all NaN)")

    # Numeric summaries (self-consistency)
    print("Analytic grid summary statistics:")
    print(f"min = {np.nanmin(DTER_analytic):.6e}, max = {np.nanmax(DTER_analytic):.6e}, mean = {np.nanmean(DTER_analytic):.6e}")

    # Optionally compare to MC grid
    if mc_grid_path is not None:
        print("Loading Monte-Carlo grid for comparison...")
        DTER_mc = load_mc_grid(mc_grid_path, expected_shape=DTER_analytic.shape)
        mask = (~np.isnan(DTER_analytic)) & (~np.isnan(DTER_mc))
        rmse, mae, max_abs, n = ah.rmse_mae_max(DTER_mc, DTER_analytic, mask=mask)
        rho, pval = ah.spearman_correlation(DTER_mc, DTER_analytic, mask=mask)

        print(f"Comparison stats (N={n} cells): RMSE={rmse:.3e}, MAE={mae:.3e}, max_abs={max_abs:.3e}")
        print(f"Spearman rho={rho:.4f} (p={pval:.3g})")
        # percent within tolerance
        for tol in [0.05, 0.1, 0.2]:
            frac = ah.percent_within_tolerance(DTER_mc, DTER_analytic, tol=tol, mask=mask)
            print(f"Fraction within {int(tol*100)}% rel error = {frac:.3f}")

        # save difference heatmaps
        abs_diff = np.abs(DTER_mc - DTER_analytic)
        rel_diff = abs_diff / (np.abs(DTER_analytic) + 1e-20)
        ah.save_heatmap(abs_diff, heat_dir / f"absdiff_dcrit_{d_crit:.1f}.png",
                        title="Absolute difference |MC - Analytic|",
                        xticklabels=xticks, yticklabels=yticks)
        ah.save_heatmap(rel_diff, heat_dir / f"reldiff_dcrit_{d_crit:.1f}.png",
                        title="Relative diff |MC - Analytic|/|Analytic|",
                        xticklabels=xticks, yticklabels=yticks, vmin=0, vmax=1.0)

        # scatter
        mask_flat = mask.flatten()
        ah.scatter_compare(DTER_analytic.flatten()[mask_flat], DTER_mc.flatten()[mask_flat],
                           outpath=heat_dir / f"scatter_analytic_vs_mc_dcrit_{d_crit:.1f}.png",
                           xlabel="Analytic DTER", ylabel="MC DTER",
                           title=f"Analytic vs MC (d_crit={d_crit:.1f})")

        # compare optima
        mc_idx = np.unravel_index(np.nanargmax(DTER_mc), DTER_mc.shape)
        an_idx = idx_list[0] if len(idx_list) > 0 else None
        if an_idx is not None:
            dist = ah.manhattan_grid_distance(mc_idx, an_idx)
            print(f"MC optimum index = {mc_idx}; Analytic optimum index = {an_idx}; Manhattan dist = {dist}")

        # save comparison summary CSV
        rows = []
        for i, n_p in enumerate(n_p_values):
            for j, l_p in enumerate(l_p_values):
                rows.append({
                    "n_p": n_p,
                    "l_p": l_p,
                    "dter_analytic": float(DTER_analytic[i, j]) if not np.isnan(DTER_analytic[i, j]) else np.nan,
                    "dter_mc": float(DTER_mc[i, j]) if not np.isnan(DTER_mc[i, j]) else np.nan,
                    "abs_diff": float(abs_diff[i, j]) if not np.isnan(abs_diff[i, j]) else np.nan,
                    "rel_diff": float(rel_diff[i, j]) if not np.isnan(rel_diff[i, j]) else np.nan
                })
        pd.DataFrame(rows).to_csv(outdir / f"analytic_vs_mc_summary_dcrit_{d_crit:.1f}.csv", index=False)
        print(f"Saved analytic vs mc summary CSV to {outdir}")

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze analytic DTER grid (and optionally compare to MC).")
    parser.add_argument("--dcrit", type=float, default=DEFAULT_DCRIT, help="d_crit value to analyze")
    parser.add_argument("--outdir", type=str, default=str(OUTDIR), help="output directory")
    parser.add_argument("--mc", type=str, default=None, help="path to Monte-Carlo grid CSV or .npy to compare (optional)")
    args = parser.parse_args()

    main(d_crit=args.dcrit, outdir=Path(args.outdir), mc_grid_path=args.mc)
