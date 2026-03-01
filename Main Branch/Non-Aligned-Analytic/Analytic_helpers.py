# analytic_helpers.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from math import sqrt
import os

sns.set(style="whitegrid")


def save_heatmap(mat, outpath, title=None, xticklabels=None, yticklabels=None,
                 cmap="coolwarm", vmin=None, vmax=None, annot=False, fmt=".3e"):
    """
    Save a heatmap image of `mat` (2D numpy array).
    xticklabels, yticklabels: iterables (or None)
    """
    plt.figure(figsize=(8, 6), dpi=150)
    ax = sns.heatmap(mat, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels,
                     linewidths=0.5, linecolor="white", vmin=vmin, vmax=vmax,
                     annot=annot, fmt=fmt, square=False)
    ax.invert_yaxis()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def scatter_compare(x, y, outpath, xlabel="Analytic", ylabel="Other", title=None, marker_size=30):
    """
    Scatter plot x vs y with identity line.
    x, y: 1D arrays of equal length (non-nan subset should be used).
    """
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(x, y, s=marker_size, alpha=0.85)
    mn = min(np.nanmin(x), np.nanmin(y))
    mx = max(np.nanmax(x), np.nanmax(y))
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def rmse_mae_max(a, b, mask=None):
    """
    Compute RMSE, MAE, and max abs difference between arrays a and b.
    mask: boolean array of same shape; True = include. If None uses all non-nan pairs.
    Returns (rmse, mae, max_abs, n_included)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if mask is None:
        mask = (~np.isnan(a)) & (~np.isnan(b))
    diff = a[mask] - b[mask]
    n = diff.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))
    return sqrt(mse), mae, max_abs, n


def spearman_correlation(a, b, mask=None):
    """
    Return spearman rho and pvalue for flattened arrays restricted by mask.
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    if mask is None:
        mask = (~np.isnan(a)) & (~np.isnan(b))
    if np.sum(mask) < 2:
        return np.nan, np.nan
    rho, p = spearmanr(a[mask], b[mask])
    return rho, p


def find_optima(mat, n_p_values=None, l_p_values=None):
    """
    Return index tuple(s) and corresponding (n_p, l_p) for argmax of mat.
    mat: 2D array (rows->n_p, cols->l_p)
    """
    if np.all(np.isnan(mat)):
        return [], []
    flat_argmax = np.nanargmax(mat)
    idx = np.unravel_index(int(flat_argmax), mat.shape)
    idx_list = [idx]
    vals = []
    if n_p_values is not None and l_p_values is not None:
        vals.append((n_p_values[idx[0]], l_p_values[idx[1]]))
    else:
        vals.append((idx[0], idx[1]))
    return idx_list, vals


def percent_within_tolerance(a, b, tol=0.1, mask=None):
    """
    Fraction of elements where |a-b|/(|b|+eps) <= tol.
    """

    a = np.asarray(a)
    b = np.asarray(b)
    eps = 1e-20
    if mask is None:
        mask = (~np.isnan(a)) & (~np.isnan(b))
    if np.sum(mask) == 0:
        return np.nan
    rel = np.abs(a[mask] - b[mask]) / (np.abs(b[mask]) + eps)
    return float(np.mean(rel <= tol))


def manhattan_grid_distance(idx1, idx2):
    """
    idx1, idx2: (row, col) index tuples
    """
    return abs(idx1[0] - idx2[0]) + abs(idx1[1] - idx2[1])


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
