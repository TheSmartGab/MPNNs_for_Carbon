import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict
from argparse import ArgumentParser

# Global matplotlib configuration
# plt.rcParams.update({
#     "font.size": 16,
#     "figure.figsize": (8, 6),
#     "axes.grid": True,
#     "grid.alpha": 0.3,
#     "lines.markersize": 8,
#     "lines.linewidth": 2,
#     "legend.fontsize": 16,
#     "legend.frameon": False
# })

def collect_data(root_dir, exclude_list):
    """Recursively find stats.json and organize by strain."""
    data = defaultdict(dict)
    pattern = re.compile(r"(\d+)K[\\/](\d+\.\d+)[\\/]RATE_OUT[\\/]stats\.json$")

    print(f"[INFO] Scanning {root_dir}...")
    for root, _, files in os.walk(root_dir):
        for f in files:
            full_path = os.path.join(root, f)
            match = pattern.search(full_path)
            if match:
                temp, strain = float(match.group(1)), match.group(2)
                if strain in exclude_list: continue
                try:
                    with open(full_path, "r") as fj:
                        data[strain][temp] = json.load(fj)
                except Exception as e:
                    print(f"[WARNING] Skipping {full_path}: {e}")
    return data

def linear_model(x, m, c):
    return m * x + c

def main():
    parser = ArgumentParser(description="Arrhenius analysis with strain-dependent trends.")
    parser.add_argument("root", type=str, help="Root directory")
    parser.add_argument("--kb", type=float, default=8.617333e-5, help="Boltzmann constant (eV/K)")
    parser.add_argument("--exclude_strain", nargs="+", default=[], help="Strains to exclude")
    args = parser.parse_args()

    out_dir = os.path.join(args.root, "ARRHENIUS")
    os.makedirs(out_dir, exist_ok=True)

    results = collect_data(args.root, args.exclude_strain)
    if not results:
        print("[ERROR] No valid data found."); return

    # --- Setup Independent Figures ---
    # 1. Arrhenius Plot
    fig_arr, ax_arr = plt.subplots(figsize=(8, 6))
    
    fit_summary = []
    sorted_strains = sorted(results.keys(), key=float)

    for strain in sorted_strains:
        temps = sorted(results[strain].keys())
        if len(temps) < 3: continue
        
        beta = np.array([1.0 / (args.kb * t) for t in temps])
        rates = np.array([results[strain][t]["q0_500"] for t in temps])
        y = -np.log(rates)
        y_sigma = np.array([results[strain][t]["std"] for t in temps]) / rates
        
        # 95% CI Error bars
        y_low = -np.log(np.array([results[strain][t]["q0_975"] for t in temps]))
        y_high = -np.log(np.array([results[strain][t]["q0_025"] for t in temps]))
        yerr = [y - y_low, y_high - y]

        try:
            popt, pcov = curve_fit(linear_model, beta, y, sigma=y_sigma, absolute_sigma=True)
            m, c = popt
            m_err, c_err = np.sqrt(np.diag(pcov))

            # Transform: y = beta*DeltaE - ln(gamma0) -> DeltaE = m, ln(gamma0) = -c
            gamma0 = np.exp(-c)
            gamma0_err = gamma0 * c_err 
            red_chi_sq = np.sum(((y - linear_model(beta, *popt)) / y_sigma)**2) / (len(y) - 2)

            # Plot on Arrhenius axes
            line = ax_arr.errorbar(beta, y, yerr=yerr, fmt='o', capsize=4, elinewidth=1)
            color = line[0].get_color()
            b_range = np.linspace(beta.min(), beta.max(), 100)
            label_str = (fr"$\epsilon$={strain}: $\Delta E$={m:.3f}$\pm${m_err:.0e} eV, "
                         fr"$\gamma_0$={gamma0:.1e}$\pm${gamma0_err:.0e} ps$^{{-1}}$")
            ax_arr.plot(b_range, linear_model(b_range, m, c), color=color, linestyle='--', alpha=0.6, label=label_str)

            fit_summary.append({
                "strain": float(strain), "delta_E": m, "delta_E_err": m_err,
                "gamma0": gamma0, "gamma0_err": gamma0_err, "red_chi_sq": red_chi_sq
            })
        except Exception as e:
            print(f"[WARNING] Fit failed for {strain}: {e}")

    # Format and save Arrhenius plot (Legend remains inside)
    ax_arr.set_xlabel(r"$\beta = 1/k_B T$ [eV$^{-1}$]", fontsize = 25)
    ax_arr.set_ylabel(r"$-\ln(\gamma)$", fontsize = 25)
    ax_arr.legend(loc = (1, 0.2), frameon=True) 
    fig_arr.savefig(os.path.join(out_dir, "MF_arrhenius_fits.pdf"), format="pdf", bbox_inches='tight')
    plt.close(fig_arr)

    # --- Trend Plots & Calculations ---
    if fit_summary:
        df = pd.DataFrame(fit_summary).sort_values("strain")
        
        # 2. Energy Barrier Plot (Cleaned up: Only raw data, no trendline)
        fig_en, ax_en = plt.subplots(figsize=(7, 5))
        ax_en.errorbar(df["strain"], df["delta_E"], yerr=df["delta_E_err"], fmt='s-', color='darkred', capsize=5)
        ax_en.set_xlabel("Strain", fontsize = 25)
        ax_en.set_ylabel(r"$\Delta E$ [eV]", fontsize = 25)
        fig_en.savefig(os.path.join(out_dir, "MF_energy_barrier_vs_strain.pdf"), format="pdf", bbox_inches='tight')
        plt.close(fig_en)

        # 3. Knocking Frequency Plot
        fig_gam, ax_gam = plt.subplots(figsize=(7, 5))
        ax_gam.errorbar(df["strain"], df["gamma0"], yerr=df["gamma0_err"], fmt='d-', color='darkblue', capsize=5)
        ax_gam.set_xlabel("Strain", fontsize = 25)
        ax_gam.set_ylabel(r"$\gamma_0$ [ps$^{-1}$]", fontsize =25)
        ax_gam.set_yscale('log')
        fig_gam.savefig(os.path.join(out_dir, "MF_knocking_frequency_vs_strain.pdf"), format="pdf", bbox_inches='tight')
        plt.close(fig_gam)

        # Calculate and record extrapolation metrics silently into a text file
        try:
            # Perform linear fit: DeltaE = A * strain + B
            popt_en, pcov_en = curve_fit(linear_model, df["strain"], df["delta_E"], sigma=df["delta_E_err"], absolute_sigma=True)
            A, B = popt_en
            var_A, var_B = np.diag(pcov_en)
            cov_AB = pcov_en[0, 1]

            # X-intercept (where DeltaE = 0): epsilon_0 = -B / A
            x_int = -B / A
            
            # Uncertainty via Delta Method: 
            # sigma_x^2 = (1/A)^2 * var_B + (B/A^2)^2 * var_A - 2*(B/A^3)*cov_AB
            x_int_err = np.sqrt((1/A)**2 * var_B + (B/(A**2))**2 * var_A - 2 * (B/(A**3)) * cov_AB)

            # Write results to a standalone text report
            report_path = os.path.join(out_dir, "extrapolation_report.txt")
            with open(report_path, "w") as f_rep:
                f_rep.write("==================================================\n")
                f_rep.write("ENERGY BARRIER STRAIN EXTRAPOLATION REPORT\n")
                f_rep.write("==================================================\n\n")
                f_rep.write("Linear Model: Delta E = A * strain + B\n\n")
                f_rep.write(f"Slope (A):      {A:+.6e} ± {np.sqrt(var_A):.6e} eV\n")
                f_rep.write(f"Intercept (B):  {B:+.6e} ± {np.sqrt(var_B):.6e} eV\n")
                f_rep.write(f"Covariance(AB): {cov_AB:+.6e}\n\n")
                f_rep.write("--------------------------------------------------\n")
                f_rep.write("Extrapolated X-Intercept (Delta E = 0):\n")
                f_rep.write(f"Critical Strain (epsilon_0): {x_int:.6f} ± {x_int_err:.6f}\n")
                f_rep.write("--------------------------------------------------\n")
                
        except Exception as e:
            print(f"[WARNING] Extrapolation math or file writing failed: {e}")

        # Save data summary
        df.to_csv(os.path.join(out_dir, "fit_results.csv"), index=False)
        
    print(f"\n[SUCCESS] Independent plots, fit CSV, and extrapolation text report saved in {out_dir}")

if __name__ == "__main__":
    main()