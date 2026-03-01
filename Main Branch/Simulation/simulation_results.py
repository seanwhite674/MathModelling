# analyze_results.py
import numpy as np
import Simulation as sim            # <- your file is simulations.py
import simulation_helpers as ah        # <- helper functions you put in analysis_helpers.py
import pandas as pd
from simulation_helpers import ci_parametric, bootstrap_ci_mean, plot_convergence

# Optional: run the simulation main() if you want to re-generate heatmaps/trajectories
# Comment out if you already ran it and want a faster iterate.
# sim.main()

# Analysis parameters
d_crit_to_analyze = 2.0
n_p_values = list(range(1, 11))
l_p_values = list(range(1, 11))
steps = 60 

# Compute grid statistics (this can be a bit slow depending on n_samples)
grid_stats = ah.compute_grid_statistics(
    n_p_values, l_p_values, d_crit_to_analyze,
    x0=0.0, y0=0.0,
    n_steps = steps , dx=0.0001,
    d=1.0, L=10000.0,
    n_samples=200,          # use 200 for quick checks; increase to 2000 for final runs
    master_seed=42
)

mean_mat = grid_stats["mean"]
sem_mat  = grid_stats["sem"]
DTER_mat = grid_stats["DTER"]
std_mat  = grid_stats["std"]

rows = []
for i, n_p in enumerate(n_p_values):
    for j, l_p in enumerate(l_p_values):
        if np.isnan(mean_mat[i, j]):
            continue
        rows.append({
            "d_crit": d_crit_to_analyze,
            "n_p": n_p,
            "l_p": l_p,
            "mean_dx": mean_mat[i, j],
            "std_dx": grid_stats["std"][i, j],
            "sem_dx": sem_mat[i, j],
            "snr": mean_mat[i, j] / grid_stats["std"][i, j],
            "dter": DTER_mat[i, j],
        })

df = pd.DataFrame(rows)

from pathlib import Path
outdir = Path("Simulation_Outputs")
outdir.mkdir(exist_ok=True)
fname = outdir / f"grid_stats_dcrit_{d_crit_to_analyze:.1f}.csv"
df.to_csv(fname, index=False)
print(f"Saved grid stats to {fname}")

# Print grid stats (console)
print(f"\nGrid stats for d_crit={d_crit_to_analyze}\n")
for i, n_p in enumerate(n_p_values):
    for j, l_p in enumerate(l_p_values):
        if not np.isnan(mean_mat[i, j]):
            print(
                f"n_p={n_p}, l_p={l_p} | "
                f"mean={mean_mat[i,j]:.4f} | "
                f"SEM={sem_mat[i,j]:.4f} | "
                f"DTER={DTER_mat[i,j]:.5f}"
            )

# ------------------------
# Find the optimum (first max) robustly
# ------------------------
if np.all(np.isnan(DTER_mat)):
    raise RuntimeError("No valid DTER cells found for this d_crit.")

# flatten index of nanmax, then unravel to 2D
flat_argmax = np.nanargmax(DTER_mat)
i_opt, j_opt = np.unravel_index(int(flat_argmax), DTER_mat.shape)
opt_n = n_p_values[i_opt]
opt_l = l_p_values[j_opt]

print(f"\nOptimum found at grid indices (i={i_opt}, j={j_opt}) -> n_p={opt_n}, l_p={opt_l}")
print(f"Mean Δx at optimum = {mean_mat[i_opt, j_opt]:.4f}, std = {std_mat[i_opt, j_opt]:.4f}")

# ------------------------
# Request displacement samples for the optimum and plot histogram
# ------------------------
# Use the Edx_mean defined in simulations.py; return_samples=True gives (array, mean)
opt_displacements, opt_mean = sim.Edx_mean(
    x0=0.0, y0=0.0, n_steps= steps, dx=0.0001,
    n_p=opt_n, l_p=opt_l, d_crit=d_crit_to_analyze,
    d=1.0, L=10000.0,
    n_samples=2000,         # increase for final CI; use 200 for quick tests
    seed=12345,
    return_samples=True,
)

print(f"\nOptimum sample mean (from Edx_mean) = {opt_mean:.4f} (n_samples={len(opt_displacements)})")

ah.plot_displacement_histogram(
    opt_displacements, n_steps = steps, outpath="Simulation_Outputs/simulation_displacement_histogram.png", 
    title=f"Δx distribution (opt: n_p={opt_n}, l_p={opt_l}, d_crit={d_crit_to_analyze})"
)


# 1) quick parametric 95% CI
mean, sem, ci_low, ci_high = ci_parametric(opt_displacements)
print(f"Parametric 95% CI: mean={mean:.4f} ± 1.96*SEM -> [{ci_low:.4f}, {ci_high:.4f}] (n={len(opt_displacements)})")
 
# 2) bootstrap nonparametric CI (moderate cost)
mean_b, b_lo, b_hi = bootstrap_ci_mean(opt_displacements, n_boot=2000, rng_seed=12345)
print(f"Bootstrap 95% CI (n_boot=2000): mean={mean_b:.4f} -> [{b_lo:.4f}, {b_hi:.4f}]")

# 3) convergence plot (uses subsets of the samples)
n_list, means, sems, v_eff = plot_convergence(opt_displacements, n_steps= steps , n_list=[50,100,200,500,1000,1500,2000], outpath="Simulation_Outputs/simulation_convergence_opt.png")
print("Convergence v_eff (full-sample):", v_eff)