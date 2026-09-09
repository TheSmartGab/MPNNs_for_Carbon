
import argparse
import json
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

import matplotlib
matplotlib.use("Agg")          # non-interactive backend  safe for parallel use

# Disable mathtext parsing for auto-formatted labels (tick labels, etc.)
# This prevents warnings when matplotlib generates scientific notation
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['axes.formatter.use_mathtext'] = False

# Suppress mathtext-related warnings from matplotlib
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Mathtext.*')
warnings.filterwarnings('ignore', message='.*ParseException.*')
warnings.filterwarnings('ignore', message='.*Glyph.*')

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import norm
from sklearn.metrics import r2_score

from statistics import compute_stats_quantiles

# unit-of-measure strings keyed by quantity name.
# ALL strings that contain backslash sequences MUST be fully wrapped in $...$
# so matplotlib's mathtext parser handles them; bare \AA outside $ causes
# ParseException in threaded rendering contexts.
UDM_MAP = {
    "E_per_atom":           r"$eV/atom$",
    "F_comp":               r"$eV/\AA$",
    "F_mag":                r"$eV/\AA$",
    "S_comp":               r"$eV/\AA^3$",
    "S_trace":              r"$eV/\AA^3$",
    "F_norm":               r"$eV/\AA$",
    "S_norm":               r"$eV/\AA^3$",
    "F_cos":                "a.u.",
    "S_cos":                "a.u.",
    # aggregated force quantities share the same units as their raw counterparts
    "aggregated_F_comp":    r"$eV/\AA$",
    "aggregated_F_mag":     r"$eV/\AA$",
    "aggregated_F_norm":    r"$eV/\AA$",
    "aggregated_F_cos":     "a.u.",
}

# Keep the legacy positional list for any code that still references it by index.
# It covers the original 9 quantities only (no aggregated entries).
udms = [
    UDM_MAP["E_per_atom"],
    UDM_MAP["F_comp"],
    UDM_MAP["F_mag"],
    UDM_MAP["S_comp"],
    UDM_MAP["S_trace"],
    UDM_MAP["F_norm"],
    UDM_MAP["S_norm"],
    UDM_MAP["F_cos"],
    UDM_MAP["S_cos"],
]


# Force-related quantities whose per-atom errors are spatially correlated.
# Error bars on these raw stats are misleading; they are suppressed in bar plots.
# Proper uncertainty is shown on the "aggregated_*" counterparts instead.
FORCE_QUANTITIES = frozenset({"F_comp", "F_mag", "F_norm", "F_cos"})

def tensor_metric(tensor: np.ndarray) -> float:
    """Frobenius-style metric for both vectors and matrices."""
    if tensor is None:
        return np.nan
    
    if tensor.ndim == 1:
        return float(np.sqrt(tensor @ tensor))
    return float(np.sqrt(np.trace(tensor.T @ tensor)))


def cosine(t1: np.ndarray, t2: np.ndarray) -> float:

    if t1 is None or t2 is None:
        return np.nan

    if t1.ndim == 1:
        return float((t1 @ t2) / (tensor_metric(t1) * tensor_metric(t2)))
    return float(np.trace(t1.T @ t2) / (tensor_metric(t1) * tensor_metric(t2)))


def voigt(stress: np.ndarray) -> np.ndarray:
    if stress is None:
        return None
    vs = np.empty(6)
    for i in range(3):
        vs[i] = stress[i, i]
    vs[3] = stress[1, 2]
    vs[4] = stress[0, 2]
    vs[5] = stress[0, 1]
    return vs


def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """Convert a 6-component Voigt vector to a symmetric 3x3 tensor.
    (Original had a bug: tensor[1,2] was set twice  fixed here.)
    """
    if v is None:
        return None
    t = np.zeros((3, 3))
    for i in range(3):
        t[i, i] = v[i]
    t[1, 2] = t[2, 1] = v[3]
    t[0, 2] = t[2, 0] = v[4]
    t[0, 1] = t[1, 0] = v[5]
    return t


# =============================================================================
# Error metrics
# =============================================================================

def mae_rmse(x: np.ndarray):
    mae = np.mean(np.abs(x))
    rmse = np.sqrt(np.mean(x ** 2))
    return mae, rmse


def apply_quantile_filter(data: np.ndarray, quantile) -> np.ndarray:
    if quantile is None or data.size == 0:
        return data
    q = np.percentile(np.abs(data), quantile)
    return data[np.abs(data) <= q]


def apply_cut(reference, model_data, ecut, fcut):
    print("[INFO] applying cut with ecut=", ecut, "fcut=", fcut)
    cut_ref, cut_model = [], []
    for ref, cfg in zip(reference, model_data):
        if cfg.get_potential_energy() > ecut:
            continue
        fs = cfg.get_forces()
        if np.any(np.einsum("ij,ij->i", fs, fs) > fcut ** 2):
            continue
        cut_ref.append(ref)
        cut_model.append(cfg)
    print("[INFO] Cutted configs:", len(reference) - len(cut_ref))
    return cut_ref, cut_model


def _get_output_path(base_path, image_format):
    """
    Convert a base path (without extension) to the full path with correct extension.
    Also returns the DPI to use for saving.
    """
    if image_format == 'pdf':
        return f"{base_path}.pdf", None  # PDF doesn't use DPI
    elif image_format == 'svg':
        return f"{base_path}.svg", None  # SVG is vector, doesn't use DPI
    elif image_format == 'png':
        return f"{base_path}.png", 600  # High-res PNG at 600 DPI
    else:
        raise ValueError(f"Unknown image format: {image_format}. Choose from: pdf, svg, png")


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_hist(data, title, xlabel, outfile_base, bins=100, image_format='pdf'):
    print("[INFO] Plotting histogram", outfile_base)
    mae, rmse = mae_rmse(data)
    data = np.asarray(data)
    data_clean = data[np.isfinite(data)]

    fig, ax = plt.subplots(figsize=(6, 4))
    counts, bin_edges, _ = ax.hist(data_clean, bins=bins, alpha=0.7, density=True, label="Data")
    mu, std = norm.fit(data_clean)
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    ax.plot(x, norm.pdf(x, mu, std), "r-", linewidth=2,
            label=f"Gaussian fit\nμ = {mu:.4e}\nσ = {std:.4e}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nMAE = {mae:.4e}, RMSE = {rmse:.4e}")
    ax.legend()
    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    _disable_mathtext_ticklabels(fig)
    outfile, dpi = _get_output_path(outfile_base, image_format)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def add_identity(ax, *args, **kwargs):
    identity = ax.plot([], [], *args, **kwargs)[0]

    def callback(event_ax):
        low  = max(*event_ax.get_xlim(), *event_ax.get_ylim())
        high = min(*event_ax.get_xlim()[::-1], *event_ax.get_ylim()[::-1])
        identity.set_data([low, high], [low, high])

    callback(ax)
    ax.callbacks.connect("xlim_changed", callback)
    ax.callbacks.connect("ylim_changed", callback)



def _disable_mathtext_ticklabels(fig):
    """Disable mathtext parsing for all tick labels in a figure."""
    def _plain_formatter(x, pos):
        """Format numbers as plain strings without mathtext."""
        return f"{x:.4g}"
    
    for ax in fig.get_axes():
        # Replace formatters with a simple function that doesn't invoke mathtext
        ax.xaxis.set_major_formatter(FuncFormatter(_plain_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_formatter))


def _udm_label(udm):
    """
    Return a udm string safe to embed *inside* a larger text label.
    When a udm like "$eV/\\AA^3$" is embedded in an f-string the whole
    composite becomes e.g. "Error [$ eV/\\AA^3$]" which matplotlib tries
    to parse as mathtext and raises ParseException.
    This function strips the outer $ and converts LaTeX commands to unicode
    so the result can be safely used inside f-string labels.
    Standalone axis labels should keep the original udm (with $).
    """
    s = udm.strip()
    if s.startswith("$") and s.endswith("$"):
        inner = s[1:-1]
        # Order matters: replace longer tokens before shorter ones
        inner = inner.replace("\\AA^3", "\u212b\u00b3")   # Å³
        inner = inner.replace("\\AA^2", "\u212b\u00b2")   # Å²
        inner = inner.replace("\\AA",   "\u212b")          # Å
        inner = inner.replace("\\Delta", "\u0394")         # Δ
        inner = inner.replace("\\sigma", "\u03c3")         # σ
        return inner
    return s


def parity_plot(DATA, udms, outdir, tag="", image_format='pdf'):
    properties = list(DATA["REF"].keys())
    for p, udm in zip(properties, udms):
        ref   = DATA["REF"][p]
        model = DATA["MODEL"][p]
        if ref.ndim != 1:
            continue
        r2 = r2_score(ref, model)
        fig, ax = plt.subplots()
        ax.scatter(ref, model, alpha=0.5)
        add_identity(ax, "--k", alpha=0.6)
        ax.set_xlabel(rf"Reference {p} [{_udm_label(udm)}]")
        ax.set_ylabel(rf"Model {p} [{_udm_label(udm)}]")
        ax.set_title(rf"$r^2 = {r2:.3f}$")
        ax.grid()
        try:
            fig.tight_layout()
        except (ValueError, RuntimeError):
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        _disable_mathtext_ticklabels(fig)
        outfile, dpi = _get_output_path(os.path.join(outdir, f"parity_{p}{tag}"), image_format)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Bar-plot helpers
# All three functions below now filter out NaN entries so no empty bars appear.
# ---------------------------------------------------------------------------

def _extract_bar_data(stats_dict, quantity, N_threshold=10):
    """
    Returns (labels, mae_vals, mae_err, rmse_vals, rmse_err, mare_vals)
    only for entries that have at least one non-NaN value among MAE/RMSE.

    Error bars are shown when quantiles are present and non-None.
    Raw force quantities have their quantiles nulled at source (_aggregate_stats),
    so they naturally produce zero error bars without any special flag here.
    """
    configs_all = list(stats_dict.keys())
    labels, mae_vals, mae_err_low, mae_err_up = [], [], [], []
    rmse_vals, rmse_err_low, rmse_err_up, mare_vals = [], [], [], []

    for cfg in configs_all:
        stats = stats_dict[cfg].get(quantity)
        if stats is None:
            continue

        mae_v = rmse_v = mare_v = np.nan
        m_el = m_eu = r_el = r_eu = 0.0

        mae_data = stats["MAE"]
        if mae_data["N"] is not None and mae_data["N"] >= N_threshold:
            mae_v = mae_data["value"]
            q = mae_data["quantiles"]
            if q:
                m_el = mae_v - q[0.025]
                m_eu = q[0.975] - mae_v

        rmse_data = stats["RMSE"]
        if rmse_data["N"] is not None and rmse_data["N"] >= N_threshold:
            rmse_v = rmse_data["value"]
            q = rmse_data["quantiles"]
            if q:
                r_el = rmse_v - q[0.025]
                r_eu = q[0.975] - rmse_v

        mare_data = stats["MARE"]
        # MARE has N=None by design; accept any finite value
        _mare_val = mare_data.get("value")
        if _mare_val is not None and np.isfinite(float(_mare_val)):
            mare_v = float(_mare_val)

        # skip completely empty rows
        if np.isnan(mae_v) and np.isnan(rmse_v) and np.isnan(mare_v):
            continue

        labels.append(cfg)
        mae_vals.append(mae_v);  mae_err_low.append(m_el); mae_err_up.append(m_eu)
        rmse_vals.append(rmse_v); rmse_err_low.append(r_el); rmse_err_up.append(r_eu)
        mare_vals.append(mare_v)

    return (labels, mae_vals, mae_err_low, mae_err_up,
            rmse_vals, rmse_err_low, rmse_err_up, mare_vals)


def plot_quantity_stats_by_config(
    all_stats, quantity, outdir, udm,
    discard_comp=False, logscale=False, N_threshold=10, image_format='pdf',
):
    if discard_comp and "comp" in quantity:
        return
    print("[INFO] plot_quantity_stats_by_config Plotting", quantity)

    (configs, mae_vals, mae_err_low, mae_err_up,
     rmse_vals, rmse_err_low, rmse_err_up, mare_vals) = _extract_bar_data(
        all_stats, quantity, N_threshold)

    if not configs:
        print(f"[WARNING] No data to plot for {quantity}  skipping.")
        return

    x = np.arange(len(configs))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(max(6, len(configs) * 0.8 + 2), 4))
    ax2 = ax1.twinx()

    _safe_bar(ax1, x - width / 2, mae_vals,  width, "MAE",
              mae_err_low,  mae_err_up)
    _safe_bar(ax1, x + width / 2, rmse_vals, width, "RMSE",
              rmse_err_low, rmse_err_up)
    ax2.bar(x + 3 * width / 2, mare_vals, width, label="MARE", alpha=0.7,
            color="C2")

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    ax1.set_ylabel(f"Error (MAE, RMSE) [{_udm_label(udm)}]")
    ax2.set_ylabel("Relative error (MARE)")
    if logscale:
        ax1.set_yscale("log"); ax2.set_yscale("log")
    ax1.set_title(f"{quantity}: statistics by configuration")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    _disable_mathtext_ticklabels(fig)
    outfile, dpi = _get_output_path(os.path.join(outdir, f"{quantity}_stats_by_config"), image_format)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def _safe_bar(ax, x, vals, width, label, err_low=None, err_up=None, **kwargs):
    """Bar plot with robust error bar handling.
    
    Guards against zero-amplitude artifacts and retains try/except fallback.
    """
    vals = np.asarray(vals)
    
    # 1. Initialize errors if they weren't passed
    err_low = np.zeros_like(vals) if err_low is None else np.asarray(err_low)
    err_up = np.zeros_like(vals) if err_up is None else np.asarray(err_up)
    
    # 2. Mask error bars where values themselves are NaN
    err_low = np.where(np.isnan(vals), np.nan, err_low)
    err_up = np.where(np.isnan(vals), np.nan, err_up)
    
    # 3. Check if there is *any* actual uncertainty data to plot
    has_err_low = np.any(~np.isnan(err_low) & (err_low != 0))
    has_err_up = np.any(~np.isnan(err_up) & (err_up != 0))
    
    # 4. Use your original try/except logic ONLY if there is data to plot
    if has_err_low or has_err_up:
        try:
            ax.bar(x, vals, width, label=label,
                   yerr=[err_low, err_up], capsize=3, **kwargs)
        except (ValueError, TypeError):
            # Your vital fallback logic is preserved here!
            ax.bar(x, vals, width, label=label, **kwargs)
    else:
        # If there is zero uncertainty, cleanly plot without touching yerr/capsize
        ax.bar(x, vals, width, label=label, **kwargs)


def plot_config_stats_summary(
    stats_cfg, cfg_name, outdir,
    discard_comp=False, logscale=False, N_threshold=10, image_format='pdf',
):
    quantities_all = list(stats_cfg.keys())
    if discard_comp:
        quantities_all = [q for q in quantities_all if "comp" not in q]

    # build data  filter empty
    quantities = []
    mae_vals, mae_err_low, mae_err_up = [], [], []
    rmse_vals, rmse_err_low, rmse_err_up = [], [], []
    mare_vals = []

    for q in quantities_all:
        stats = stats_cfg[q]
        mae_v = rmse_v = mare_v = np.nan
        m_el = m_eu = r_el = r_eu = 0.0

        md = stats["MAE"]
        if md["N"] is not None and md["N"] >= N_threshold:
            mae_v = md["value"]
            qt = md["quantiles"]
            if qt:
                m_el = mae_v - qt[0.025]; m_eu = qt[0.975] - mae_v

        rd = stats["RMSE"]
        if rd["N"] is not None and rd["N"] >= N_threshold:
            rmse_v = rd["value"]
            qt = rd["quantiles"]
            if qt:
                r_el = rmse_v - qt[0.025]; r_eu = qt[0.975] - rmse_v

        mrd = stats["MARE"]
        # MARE has N=None by design (and for aggregated_ quantities); accept any finite value
        _mare_val = mrd.get("value")
        if _mare_val is not None and np.isfinite(float(_mare_val)):
            mare_v = float(_mare_val)

        if np.isnan(mae_v) and np.isnan(rmse_v) and np.isnan(mare_v):
            continue

        quantities.append(q)
        mae_vals.append(mae_v);  mae_err_low.append(m_el); mae_err_up.append(m_eu)
        rmse_vals.append(rmse_v); rmse_err_low.append(r_el); rmse_err_up.append(r_eu)
        mare_vals.append(mare_v)

    if not quantities:
        print(f"[WARNING] No data to plot for summary of {cfg_name}  skipping.")
        return

    x = np.arange(len(quantities))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(max(7, len(quantities) * 0.9 + 2), 4))
    ax2 = ax1.twinx()

    _safe_bar(ax1, x - width / 2, mae_vals,  width, "MAE",  mae_err_low,  mae_err_up)
    _safe_bar(ax1, x + width / 2, rmse_vals, width, "RMSE", rmse_err_low, rmse_err_up)
    ax2.bar(x + 3 * width / 2, mare_vals, width, label="MARE", alpha=0.7, color="C2")

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(quantities, rotation=45, ha="right")
    ax1.set_ylabel("Error (MAE, RMSE)")
    ax2.set_ylabel("Relative error (MARE)")
    if logscale:
        ax1.set_yscale("log"); ax2.set_yscale("log")
    # ax1.set_title(f"Statistics summary — {cfg_name}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    _disable_mathtext_ticklabels(fig)
    outfile, dpi = _get_output_path(os.path.join(outdir, f"stats_summary_{cfg_name}"), image_format)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


import matplotlib.ticker as ticker

def plot_global(
    global_stats, quantity, outdir, udm,
    discard_comp=False, logscale=False, metric="RMSE", N_threshold=10, image_format='pdf',
):
    if discard_comp and "comp" in quantity:
        return
    print("[INFO] plot_global plotting", quantity)

    # Raw force quantities have their quantiles nulled at source (_aggregate_stats),
    # so no separate suppression flag is needed here — the `if q` guard below
    # naturally produces zero error bars when quantiles is None.
    labels  = list(global_stats.keys())
    configs = list(global_stats[labels[0]].keys())

    # Determine which configs have at least one finite value for this metric.
    # Some metrics (MARE, RMSRE, R2) store N=None by design; for those we only
    # require a finite value.  For metrics that do carry N we additionally
    # require N >= N_threshold.
    def _metric_ok(md, N_threshold):
        if md is None:
            return False
        val = md.get("value")
        if val is None or not np.isfinite(float(val)) if isinstance(val, (int, float)) else False:
            return False
        N = md.get("N")
        if N is not None and N < N_threshold:
            return False
        return True

    valid_configs = []
    for config in configs:
        for label in labels:
            data = global_stats[label].get(config, {}).get(quantity)
            if data is None:
                continue
            md = data.get(metric)
            if _metric_ok(md, N_threshold):
                valid_configs.append(config)
                break

    if not valid_configs:
        print(f"[WARNING] No data to plot for global {quantity} {metric}  skipping.")
        return

    n_labels = len(labels)
    width = 0.7 / n_labels
    x = np.arange(len(valid_configs))

    fig, ax1 = plt.subplots(figsize=(max(6, len(valid_configs) * n_labels * 0.4 + 2), 6))

    for shift_idx, label in enumerate(labels):
        vals, err_low, err_up = [], [], []
        for config in valid_configs:
            data = global_stats[label].get(config, {}).get(quantity)
            if data is None:
                vals.append(np.nan); err_low.append(0); err_up.append(0)
                continue
            md = data.get(metric)
            if _metric_ok(md, N_threshold):
                val = md["value"]
                vals.append(val)
                q = md.get("quantiles")
                if q:
                    err_low.append(val - q[0.025]); err_up.append(q[0.975] - val)
                else:
                    err_low.append(0); err_up.append(0)
            else:
                vals.append(np.nan); err_low.append(0); err_up.append(0)

        _safe_bar(ax1, x + shift_idx * width, vals, width, label, err_low, err_up)

    _disable_mathtext_ticklabels(fig)

    tick_positions = x + (n_labels - 1) * width / 2.0

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([str(c) for c in valid_configs])

    # Lock them in place
    ax1.xaxis.set_major_locator(plt.FixedLocator(tick_positions))
    ax1.xaxis.set_major_formatter(plt.FixedFormatter([str(c) for c in valid_configs]))

    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    # adjust metrics name
    if quantity.endswith("_norm"):
        if metric == "RMSE":
            metric_label = "RMSnE"
        if metric == "MARE":
            metric_label = "MRnE"
    else:
        metric_label = metric

    ax1.set_ylabel(f"{metric_label} [{_udm_label(udm)}]")
    
    if logscale:
        ax1.set_yscale("log")
        
        # Force a canvas draw so Matplotlib calculates bounds accurately
        fig.canvas.draw()
        ymin, ymax = ax1.get_ylim()
        major_locs = ax1.yaxis.get_major_locator().tick_values(ymin, ymax)
        
        # Check how many major ticks actually fall inside the visible plot view range
        visible_ticks = [t for t in major_locs if ymin <= t <= ymax]
        
        # If there's less than 2 labels, inject subs (1, 2, 5) to expand tick visibility 
        if len(visible_ticks) < 2:
            ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
            ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # ax1.set_title(f"{quantity}: statistics by configuration")
    ax1.legend()
    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
    outfile, dpi = _get_output_path(os.path.join(outdir, f"global_{quantity}_{metric}"), image_format)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)

# =============================================================================
# Statistics computation
# =============================================================================

def compute_statistics_flag(ref=None, model=None, err=None, flag="normal"):
    if flag == "normal":
        return compute_statistics(ref, model)
    if flag == "diff":
        return compute_statistics_diff(ref, err, model)
    if flag == "special":
        return compute_statistics_special(err)
    raise ValueError(f"Unknown flag: {flag}")


def compute_statistics(ref, model):
    mask = np.isfinite(ref) & np.isfinite(model)
    ref, model = ref[mask], model[mask]
    N = ref.size
    err = model - ref
    rel_err = err / ref
    rel_err = rel_err[np.isfinite(rel_err)]
    mae   = float(np.mean(np.abs(err)))
    mare  = float(np.mean(np.abs(rel_err)))
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    rmsre = float(np.sqrt(np.mean(rel_err ** 2)))
    mean_err = float(np.mean(err))
    var_err  = float(np.var(err, ddof=1))
    r2 = float(r2_score(ref, model))
    quantiles = compute_stats_quantiles(mean_err, var_err, N)
    return _pack_stats(mae, mare, rmse, rmsre, r2, mean_err, var_err, N, quantiles)


def compute_statistics_diff(ref, err, model):
    mask = np.isfinite(ref) & np.isfinite(model) & np.isfinite(err)
    ref, model, err = ref[mask], model[mask], err[mask]
    N = err.size
    rel_err = err / ref
    rel_err = rel_err[np.isfinite(rel_err)]
    mae   = float(np.mean(np.abs(err)))
    mare  = float(np.mean(np.abs(rel_err)))
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    rmsre = float(np.sqrt(np.mean(rel_err ** 2)))
    mean_err = float(np.mean(err))
    var_err  = float(np.var(err))
    r2 = float(r2_score(ref, model))
    quantiles = compute_stats_quantiles(mean_err, var_err, N)
    return _pack_stats(mae, mare, rmse, rmsre, r2, mean_err, var_err, N, quantiles)


def compute_statistics_special(values):
    mask   = np.isfinite(values)
    values = values[mask]
    N = values.size
    mae      = float(np.mean(np.abs(values)))
    rmse     = float(np.sqrt(np.mean(values ** 2)))
    mean_err = float(np.mean(values))
    var_err  = float(np.var(values))
    quantiles = compute_stats_quantiles(mean_err, var_err, N)
    return _pack_stats(mae, np.nan, rmse, np.nan, np.nan, mean_err, var_err, N, quantiles)


def _pack_stats(mae, mare, rmse, rmsre, r2, mean_err, var_err, N, quantiles):
    def _entry(val, q_key, N_val):
        return {"value": float(val) if np.isfinite(val) else val,
                "quantiles": quantiles[q_key] if quantiles else None,
                "N": N_val}

    return {
        "MAE":      _entry(mae,      "MAE",      N),
        "MARE":     {"value": float(mare) if np.isfinite(mare) else mare,
                     "quantiles": None, "N": None},
        "RMSE":     _entry(rmse,     "RMSE",     N),
        "RMSRE":    {"value": float(rmsre) if np.isfinite(rmsre) else rmsre,
                     "quantiles": None, "N": None},
        "R2":       {"value": float(r2) if np.isfinite(r2) else r2,
                     "quantiles": None, "N": N},
        "MEAN":     _entry(mean_err, "MEAN",     N),
        "VARIANCE": _entry(var_err,  "VARIANCE", N),
    }


# =============================================================================
# Force-specific uncertainty estimation via per-configuration aggregation
# =============================================================================
# Forces within one configuration are spatially correlated: treating all
# N_atoms * 3 components as independent samples over-counts the effective
# degrees of freedom and produces over-confident uncertainty intervals.
#
# The fix: for each force-related quantity, collapse each configuration to a
# *single* scalar (its per-config MAE, MSE, or RMSE).  Those N_configs scalars
# are genuinely independent (different physical structures), so they can be fed
# directly to the standard Bayesian posterior in compute_stats_quantiles.
#
# Specifically, for a set of per-configuration scalars v_i:
#   aggregated MAE  = mean(v_MAE_i)         where v_MAE_i  = mean_atoms |err_i|
#   aggregated MSE  = mean(v_MSE_i)         where v_MSE_i  = mean_atoms  err_i^2
#   aggregated RMSE = mean(sqrt(v_MSE_i))   = mean(v_RMSE_i)
#
# Uncertainty on the aggregated mean is estimated by treating the v_i as i.i.d.
# samples and sampling the Bayesian posterior for their mean and variance.

def _pack_aggregated_stats(
    per_cfg_mae: np.ndarray,
    per_cfg_mse: np.ndarray,
) -> dict:
    """
    Given 1-D arrays of per-configuration MAE and MSE values (length = N_configs),
    compute point estimates and Bayesian uncertainty intervals for the aggregated
    MAE and RMSE.

    Returns a dict with the same schema as _pack_stats so that it is transparent
    to all downstream plotting / JSON-dumping code.
    """
    mask = np.isfinite(per_cfg_mae) & np.isfinite(per_cfg_mse)
    per_cfg_mae = per_cfg_mae[mask]
    per_cfg_mse = per_cfg_mse[mask]
    N = per_cfg_mae.size

    # --- point estimates ---
    agg_mae  = float(np.mean(per_cfg_mae))
    agg_rmse = float(np.mean(np.sqrt(per_cfg_mse)))  # mean of per-cfg RMSEs

    # --- uncertainty via Bayesian posterior on the per-cfg MAE values ---
    #     (MAE uncertainty)
    mean_mae = float(np.mean(per_cfg_mae))
    var_mae  = float(np.var(per_cfg_mae, ddof=1)) if N > 1 else 0.0
    q_mae    = compute_stats_quantiles(mean_mae, var_mae, N)

    # --- uncertainty via Bayesian posterior on the per-cfg RMSE values ---
    per_cfg_rmse = np.sqrt(per_cfg_mse)
    mean_rmse = float(np.mean(per_cfg_rmse))
    var_rmse  = float(np.var(per_cfg_rmse, ddof=1)) if N > 1 else 0.0
    q_rmse    = compute_stats_quantiles(mean_rmse, var_rmse, N)

    return {
        # MAE / RMSE with proper per-config uncertainty.
        #
        # compute_stats_quantiles(mean_x, var_x, N) samples the Bayesian
        # posterior for a Gaussian population given N i.i.d. observations with
        # sample mean=mean_x and sample variance=var_x.  It returns quantiles
        # for several derived quantities of that population.
        #
        # Here we want the uncertainty on the *mean* of the N per-config values
        # (i.e. the aggregated MAE/RMSE point estimate itself), which is exactly
        # the "MEAN" key — the posterior quantiles of mu.  Reading "MAE" or
        # "RMSE" instead would give quantiles of E[|X|] / sqrt(E[X^2]) for a
        # draw from the population, not for its mean, producing far wider and
        # conceptually wrong intervals.
        "MAE":      {"value": agg_mae,  "quantiles": q_mae["MEAN"]  if q_mae  else None, "N": N},
        "RMSE":     {"value": agg_rmse, "quantiles": q_rmse["MEAN"] if q_rmse else None, "N": N},
        # Quantities that don't have a natural per-config analogue are left NaN
        "MARE":     {"value": np.nan, "quantiles": None, "N": None},
        "RMSRE":    {"value": np.nan, "quantiles": None, "N": None},
        "R2":       {"value": np.nan, "quantiles": None, "N": N},
        "MEAN":     {"value": np.nan, "quantiles": None, "N": N},
        "VARIANCE": {"value": np.nan, "quantiles": None, "N": N},
    }


def compute_statistics_aggregated_forces(results, quantity: str) -> dict:
    """
    Compute aggregated (per-configuration) statistics for a force-related
    quantity, avoiding the correlation problem of treating per-atom errors as
    independent.

    Parameters
    ----------
    results : list of per-frame dicts produced by _process_raw
    quantity : one of
        "F_comp"   – Cartesian force components  (f2 - f1), shape (N_atoms, 3)
        "F_mag"    – force magnitude differences  |f2|-|f1|, shape (N_atoms,)
        "F_norm"   – per-atom |f2-f1| norm,                 shape (N_atoms,)
        "F_cos"    – per-atom 1 - cosine similarity,        shape (N_atoms,)

    Returns
    -------
    dict with the same schema as _pack_stats
    """
    per_cfg_mae  = []
    per_cfg_mse  = []

    for r in results:
        if quantity == "F_comp":
            err = (r["f2"] - r["f1"]).reshape(-1)
        elif quantity == "F_mag":
            err = r["fmag2"] - r["fmag1"]
        elif quantity == "F_norm":
            err = r["f_norm_diff"]                 # per-atom |Δf| norms
        elif quantity == "F_cos":
            err = 1.0 - r["f_cos"]                 # per-atom 1 - cosine
        else:
            raise ValueError(f"Unknown aggregated force quantity: {quantity}")

        err = err[np.isfinite(err)]
        if err.size == 0:
            continue
        per_cfg_mae.append(float(np.mean(np.abs(err))))
        per_cfg_mse.append(float(np.mean(err ** 2)))

    if not per_cfg_mae:
        # nothing to work with — return all-NaN sentinel
        return _pack_aggregated_stats(np.array([np.nan]), np.array([np.nan]))

    return _pack_aggregated_stats(
        np.array(per_cfg_mae),
        np.array(per_cfg_mse),
    )


# =============================================================================
# Per-frame worker  operates on raw numpy tuples so it is fully picklable
# and runs efficiently inside a ProcessPoolExecutor.
# =============================================================================

def _extract_raw(frames1, frames2, E0):
    """
    Pull all needed data out of ASE Atoms objects into plain numpy tuples.
    Must be called in the main process (where ASE objects live).
    The returned list is fully picklable.
    """
    raw = []
    for a1, a2 in zip(frames1, frames2):
        n  = len(a1)
        e1 = float(a1.get_potential_energy()) / n
        e2 = float(a2.get_potential_energy()) / n
        if E0 is not None:
            e1 -= E0; e2 -= E0

        try:
            stress1 = a1.get_stress().copy()
        except Exception as e:
            stress1 = None
        try:
            stress2 = a2.get_stress().copy()
        except Exception as e:
            stress2 = None

        raw.append((e1, e2,
                    a1.get_forces().copy(),
                    a2.get_forces().copy(),
                    stress1,
                    stress2)
                    )
    return raw


def _process_raw(item):
    """
    Process a single (e1, e2, f1, f2, s1, s2) numpy tuple.
    No ASE objects  fully picklable for ProcessPoolExecutor.
    Uses vectorised einsum instead of Python loops over atoms.
    """
    e1, e2, f1, f2, s1, s2 = item
    t_s1 = voigt_to_tensor(s1)
    t_s2 = voigt_to_tensor(s2)

    fmag1 = np.linalg.norm(f1, axis=1)
    fmag2 = np.linalg.norm(f2, axis=1)

    # Vectorised per-atom cosine
    dot   = np.einsum("ij,ij->i", f1, f2)
    na    = np.linalg.norm(f1, axis=1)
    nb    = np.linalg.norm(f2, axis=1)
    denom = na * nb
    safe  = denom > 0
    f_cos_vals = np.where(safe, dot / np.where(safe, denom, 1.0), 0.0)

    s_cos_val = cosine(s1, s2)

    # Vectorised per-atom norms via einsum
    f_norm_ref  = np.sqrt(np.einsum("ij,ij->i", f1, f1))
    f_norm_mod  = np.sqrt(np.einsum("ij,ij->i", f2, f2))
    fdiff       = f2 - f1
    f_norm_diff = np.sqrt(np.einsum("ij,ij->i", fdiff, fdiff))

    s_norm_ref  = tensor_metric(t_s1)
    s_norm_mod  = tensor_metric(t_s2)

    try:
        s_norm_diff = tensor_metric(t_s2 - t_s1)
    except Exception as e:
        s_norm_diff = np.nan

    if s1 is None:
        s_trace_ref = np.nan
    else:
        s_trace_ref = float(np.sum(s1[:3]))

    if s2 is None:
        s_trace_mod = np.nan
    else:
        s_trace_mod = float(np.sum(s2[:3]))

    if s_cos_val is not None:
        s_cos_val = float(s_cos_val)
    else:
        s_cos_val = np.nan

    return {
        "e1": e1, "e2": e2,
        "f1": f1, "f2": f2,
        "fmag1": fmag1, "fmag2": fmag2,
        "s1": s1, "s2": s2,
        "s_trace_ref": s_trace_ref,
        "s_trace_mod": s_trace_mod,
        "f_cos": f_cos_vals, "s_cos": s_cos_val,
        "f_norm_ref": f_norm_ref, "f_norm_mod": f_norm_mod, "f_norm_diff": f_norm_diff,
        "s_norm_ref": s_norm_ref, "s_norm_mod": s_norm_mod, "s_norm_diff": s_norm_diff,
    }

# =============================================================================
# Core analysis
# =============================================================================

def _aggregate_stats(results, label, save, args):
    """
    Aggregate a list of per-frame result dicts into statistics.
    Separated from run_analysis so it can be called from worker processes.
    `args` is passed only for quantile / output_dir / save settings.
    """
    dEpa    = np.array([r["e2"] - r["e1"] for r in results])
    dF_comp = np.concatenate([(r["f2"] - r["f1"]).reshape(-1) for r in results])
    dF_mag  = np.concatenate([r["fmag2"] - r["fmag1"] for r in results])

    dS_comp = np.concatenate([
        r["s2"] - r["s1"]
        if r["s2"] is not None and r["s1"] is not None else np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for r in results
    ])
    dS_trace = np.array([
        (r["s_trace_mod"] - r["s_trace_ref"]) 
        if (r["s_trace_mod"] is not None and r["s_trace_ref"] is not None) 
        else None
        for r in results
    ])

    F_cos = apply_quantile_filter(
        np.concatenate([r["f_cos"] for r in results]), args.quantile)
    S_cos = apply_quantile_filter(
        np.array([r["s_cos"] for r in results]), args.quantile)

    dF_norm = apply_quantile_filter(
        np.concatenate([r["f_norm_diff"] for r in results]), args.quantile)
    dS_norm = apply_quantile_filter(
        np.array([r["s_norm_diff"] for r in results]), args.quantile)

    dEpa     = apply_quantile_filter(dEpa,     args.quantile)
    dF_comp  = apply_quantile_filter(dF_comp,  args.quantile)
    dF_mag   = apply_quantile_filter(dF_mag,   args.quantile)
    dS_comp  = apply_quantile_filter(dS_comp,  args.quantile)
    dS_trace = apply_quantile_filter(dS_trace, args.quantile)

    DATA = {
        "REF": {
            "E_per_atom": np.array([r["e1"] for r in results]),
            "F_comp":     np.concatenate([r["f1"].reshape(-1) for r in results]),
            "F_mag":      np.concatenate([r["fmag1"] for r in results]),
            "S_comp":     np.concatenate([
                r["s1"] 
                if r["s1"] is not None 
                else np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                for r in results
            ]),
            "S_trace":    np.array([r["s_trace_ref"] for r in results]),
        },
        "MODEL": {
            "E_per_atom": np.array([r["e2"] for r in results]),
            "F_comp":     np.concatenate([r["f2"].reshape(-1) for r in results]),
            "F_mag":      np.concatenate([r["fmag2"] for r in results]),
            "S_comp":     np.concatenate([
                r["s2"] 
                if r["s2"] is not None 
                else np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                for r in results
            ]),
            "S_trace":    np.array([r["s_trace_mod"] for r in results]),
        },
    }
    TENSOR_DATA = {
        "REF":  {
            "F_norm": np.concatenate([r["f_norm_ref"] for r in results]),
            "S_norm": np.array([r["s_norm_ref"] for r in results]),
        },
        "MODEL": {
            "F_norm": np.concatenate([r["f_norm_mod"] for r in results]),
            "S_norm": np.array([r["s_norm_mod"] for r in results]),
        },
        "DIFF": {
            "F_norm": dF_norm,
            "S_norm": dS_norm,
        },
    }

    if save:
        _save_plots(DATA, TENSOR_DATA, dEpa, dF_comp, dF_mag,
                    dS_comp, dS_trace, dF_norm, dS_norm,
                    F_cos, S_cos, args, label)

    stats = {}
    for p in DATA["REF"]:
        ref   = DATA["REF"][p]
        model = DATA["MODEL"][p]
        if ref.ndim != 1:
            continue
        stats[p] = compute_statistics(ref, model)

    for p in TENSOR_DATA["REF"]:
        ref   = TENSOR_DATA["REF"][p]
        diff  = TENSOR_DATA["DIFF"][p]
        model = TENSOR_DATA["MODEL"][p]
        stats[p] = compute_statistics_diff(ref, diff, model)

    stats["F_cos"] = compute_statistics_special(1 - F_cos)
    stats["S_cos"] = compute_statistics_special(1 - S_cos)

    # -------------------------------------------------------------------------
    # Null out quantiles on raw force quantities.
    # Per-atom errors within one configuration are spatially correlated, so the
    # Bayesian posterior computed over all N_atoms * N_configs values is
    # over-confident and statistically meaningless.  The correct uncertainty is
    # carried by the aggregated_* counterparts computed below.
    for fq in FORCE_QUANTITIES:
        if fq in stats:
            for metric_dict in stats[fq].values():
                if isinstance(metric_dict, dict):
                    metric_dict["quantiles"] = None

    # -------------------------------------------------------------------------
    # Aggregated force statistics (per-configuration aggregation)
    # Forces within one frame are spatially correlated; aggregating to per-config
    # MAE/RMSE before uncertainty estimation gives correct independent samples.
    # These are stored under "aggregated_<quantity>" keys.
    # -------------------------------------------------------------------------
    for fq in ("F_comp", "F_mag", "F_norm", "F_cos"):
        stats[f"aggregated_{fq}"] = compute_statistics_aggregated_forces(results, fq)

    return stats


def run_analysis(frames1, frames2, args, label="", save=True):
    if label:
        label = f"_{label}"

    # ASE extraction must happen in the calling process/thread
    raw_items = _extract_raw(frames1, frames2, args.E0)

    worker  = _process_raw
    results = _parallel_map(worker, raw_items)

    return _aggregate_stats(results, label, save, args)


def _parallel_map(fn, items, max_workers=None):
    """Run fn over items with a ProcessPoolExecutor; fall back to threads."""
    if not items:
        return []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(fn, items))
    except Exception as e:
        print(f"[WARNING] ProcessPoolExecutor failed ({e}), falling back to threads")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(fn, items))


def _save_plots(DATA, TENSOR_DATA, dEpa, dF_comp, dF_mag,
                dS_comp, dS_trace, dF_norm, dS_norm,
                F_cos, S_cos, args, label):
    """Save all histograms and parity plots; dispatched concurrently."""
    udms_p = [r"$eV/atom$", r"$eV/\AA$", r"$eV/\AA$", r"$eV/\AA^3$", r"$eV/\AA^3$"]
    tensor_udms = [r"$eV/\AA$", r"$eV/\AA^3$"]

    plot_tasks = [
        partial(parity_plot, DATA, udms_p, args.output_dir, label, image_format=args.image_format),
        partial(parity_plot, TENSOR_DATA, tensor_udms, args.output_dir, label, image_format=args.image_format),
        partial(plot_hist, 1.0 - F_cos,
                "1 - Cosine between the model and reference force", "a.u.",
                os.path.join(args.output_dir, f"F_cos{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, S_cos - 1.0,
                "1 - Cosine between the model and reference Stress tensor", "a.u.",
                os.path.join(args.output_dir, f"S_cos{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, dF_norm, "Force difference metric", r"$eV/\AA$",
                os.path.join(args.output_dir, f"F_metric{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, dS_norm, "Stress difference metric", r"$eV/\AA^3$",
                os.path.join(args.output_dir, f"S_metric{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, dEpa, "Energy per atom difference",
                r"$\Delta E$ (eV/atom)",
                os.path.join(args.output_dir, f"energy_per_atom_diff{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, dF_comp, "Force component difference",
                r"$\Delta F$ (eV/$\AA$)",
                os.path.join(args.output_dir, f"force_component_diff{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, dF_mag, "Force magnitude difference",
                r"$\Delta |F|$ (eV/$\AA$)",
                os.path.join(args.output_dir, f"force_magnitude_diff{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, dS_comp, "Stress component difference",
                r"$\Delta \sigma$ (eV/$\AA^3$)",
                os.path.join(args.output_dir, f"stress_component_diff{label}"), bins=100, image_format=args.image_format),
        partial(plot_hist, dS_trace, "Stress trace difference",
                r"$\Delta Tr(\sigma)$ (eV/$\AA^3$)",
                os.path.join(args.output_dir, f"stress_trace_diff{label}"), bins=100, image_format=args.image_format),
    ]

    # matplotlib is not thread-safe with the same figure in multiple threads,
    # but each task creates its own figure so ThreadPoolExecutor is safe here.
    with ThreadPoolExecutor() as ex:
        futs = [ex.submit(t) for t in plot_tasks]
        for f in as_completed(futs):
            exc = f.exception()
            if exc:
                print(f"[WARNING] Plot task raised: {exc}")
                traceback.print_exception(type(exc), exc, exc.__traceback__)



# =============================================================================
# Module-level config worker  (must be at module scope to be picklable)
# =============================================================================

def _run_config_job(job):
    """
    Worker that runs inside a ProcessPoolExecutor subprocess.
    Receives a (ct_key, raw_items) tuple of plain numpy data.
    Returns (ct_key, per_frame_results) — no matplotlib, no args object.
    """
    ct_key, raw_items = job
    worker  = _process_raw
    results = _parallel_map(worker, raw_items)
    return ct_key, results


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",    nargs="+", required=True)
    parser.add_argument("--model",  nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--ecut",  type=float, default=np.inf)
    parser.add_argument("--fcut",  type=float, default=np.inf)
    parser.add_argument("--config_type", type=str, default=None)
    parser.add_argument("--discard_comp", action="store_true")
    parser.add_argument("--logscale",     action="store_true")
    parser.add_argument("--E0", type=float, default=None)
    parser.add_argument("--save_hist_by_config", action="store_true")
    parser.add_argument("--image_format", type=str, default="svg",
                        choices=["pdf", "svg", "png"],
                        help="Output image format: pdf, svg, or png (600 dpi)")
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    from ase.io import read

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    GLOBAL_STATS = {}

    for label, ref_file, model_file in zip(args.labels, args.ref, args.model):
        print("=" * 60)
        print("[INFO] Processing label", label)

        print("[INFO] reading file", ref_file)
        frames_ref   = read(ref_file, ":")
        print("[INFO] reading file", model_file)
        frames_model = read(model_file, ":")

        frames_ref, frames_model = apply_cut(frames_ref, frames_model,
                                             args.ecut, args.fcut)

        all_stats = {}
        print("[INFO] Running global analysis")
        all_stats["total"] = run_analysis(frames_ref, frames_model, args, label=label)

        if args.config_type:
            print("[INFO] Running per-config-type analysis")
            config_map = {}
            for i, a in enumerate(frames_ref):
                ct = a.info.get(args.config_type)
                config_map.setdefault(ct, []).append(i)

            # Extract raw numpy data in the main process (ASE objects live here),
            # then dispatch the pure-numpy worker via ProcessPoolExecutor.
            # _aggregate_stats (which calls matplotlib) runs back in main process.
            config_jobs = []
            config_meta = {}  # ct_key -> (label_str, save_flag)
            for ct, idx in config_map.items():
                f1 = [frames_ref[i]   for i in idx]
                f2 = [frames_model[i] for i in idx]
                print(f"[INFO] {ct}: {len(f1)} configs")
                ct_key = str(ct)
                raw = _extract_raw(f1, f2, args.E0)
                config_jobs.append((ct_key, raw))
                config_meta[ct_key] = (label + "_" + ct_key,
                                       args.save_hist_by_config)

            with ProcessPoolExecutor() as ex:
                futs = {ex.submit(_run_config_job, job): job
                        for job in config_jobs}
                for fut in as_completed(futs):
                    try:
                        ct_key, frame_results = fut.result()
                        lbl, do_save = config_meta[ct_key]
                        # aggregate + plot in main process (matplotlib-safe)
                        all_stats[ct_key] = _aggregate_stats(
                            frame_results, lbl, do_save, args)
                    except Exception as e:
                        print(f"[WARNING] Config analysis failed: {e}")
                        traceback.print_exc()

        # save statistics JSON
        stats_file = os.path.join(args.output_dir, f"{label}_statistics.json")
        with open(stats_file, "w") as f:
            json.dump(all_stats, f, indent=2)
        print("[INFO] Statistics written to", stats_file)

        quantities = list(all_stats["total"].keys())

        # Split quantities into regular and aggregated for plotting purposes.
        # UDM_MAP is the authoritative source of units; unknown quantities fall
        # back to "a.u." so new entries never cause a KeyError.
        def _udm_for(q):
            return UDM_MAP.get(q, "a.u.")

        # A) per-quantity bar plots  dispatched concurrently
        print("[INFO] Plotting statistical summaries")
        with ThreadPoolExecutor() as ex:
            futs = []
            for q in quantities:
                futs.append(ex.submit(
                    plot_quantity_stats_by_config,
                    all_stats, q, args.output_dir, _udm_for(q),
                    args.discard_comp, args.logscale, image_format=args.image_format,
                ))
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc:
                    print(f"[WARNING] plot_quantity_stats_by_config raised: {exc}")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)

        # B) per-config summary plots
        if args.save_hist_by_config:
            by_cfg_dir = os.path.join(args.output_dir, "BY_CONFIG")
            os.makedirs(by_cfg_dir, exist_ok=True)
            with ThreadPoolExecutor() as ex:
                futs = {
                    ex.submit(
                        plot_config_stats_summary,
                        stats_cfg, cfg, by_cfg_dir,
                        args.discard_comp, args.logscale, image_format=args.image_format,
                    ): cfg
                    for cfg, stats_cfg in all_stats.items()
                }
                for fut in as_completed(futs):
                    exc = fut.exception()
                    if exc:
                        print(f"[WARNING] plot_config_stats_summary raised: {exc}")
                        traceback.print_exception(type(exc), exc, exc.__traceback__)

        GLOBAL_STATS[label] = all_stats

    # C) global comparison plots
    # Separate regular and aggregated quantities so we can route each to the
    # correct output filename prefix.
    all_quantities   = list(GLOBAL_STATS[args.labels[0]]["total"].keys())
    reg_quantities   = [q for q in all_quantities if not q.startswith("aggregated_")]
    agg_quantities   = [q for q in all_quantities if q.startswith("aggregated_")]

    def _udm_for(q):
        return UDM_MAP.get(q, "a.u.")

    with ThreadPoolExecutor() as ex:
        futs = []
        all_metrics = list(GLOBAL_STATS[args.labels[0]]["total"][all_quantities[0]].keys())

        # C1) regular global plots (existing behaviour, error bars suppressed for
        #     raw force quantities via the _no_err flag inside plot_global)
        for metric in all_metrics:
            for q in reg_quantities:
                udm   = _udm_for(q)
                _udm  = "a.u." if metric == "MARE" else udm
                _lscl = False if metric == "MEAN_ERROR" else args.logscale
                if "_cos" in q and metric in ("MARE", "RMSRE", "R2"):
                    continue
                futs.append(ex.submit(
                    plot_global,
                    GLOBAL_STATS, q, args.output_dir, _udm,
                    args.discard_comp, _lscl, metric, image_format=args.image_format,
                ))

        # C2) aggregated global plots — these carry proper per-config uncertainty
        #     intervals and are written with the "aggregated_global_" prefix via
        #     plot_global (which already prefixes the filename with the quantity
        #     name, which itself starts with "aggregated_").
        for metric in ("MAE", "RMSE"):        # only metrics that have meaning here
            for q in agg_quantities:
                udm  = _udm_for(q)
                _lscl = args.logscale
                futs.append(ex.submit(
                    plot_global,
                    GLOBAL_STATS, q, args.output_dir, udm,
                    args.discard_comp, _lscl, metric, image_format=args.image_format,
                ))

        for fut in as_completed(futs):
            exc = fut.exception()
            if exc:
                print(f"[WARNING] plot_global raised: {exc}")
                traceback.print_exception(type(exc), exc, exc.__traceback__)

if __name__ == "__main__":
    main()