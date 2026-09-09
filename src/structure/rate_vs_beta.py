import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Constants
KB = 8.617333262145e-5  # eV/K

def load_data(base_path):
    betas = []
    means = []
    stds = []
    raw_temps = []
    raw_means = []
    raw_stds = []

    if not os.path.isdir(base_path):
        print(f"Error: The directory '{base_path}' does not exist.")
        return None

    dirs = [d for d in os.listdir(base_path) if d.isdigit()]
    dirs.sort(key=int)

    for d in dirs:
        json_path = os.path.join(base_path, d, 'stats.json')
        if not os.path.exists(json_path):
            continue
        
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
                T = float(d)
                beta = 1 / (KB * T)
                
                mu_rate = data['mean']
                sigma_rate = data['std']
                
                # Transform to -log space
                val = -np.log(mu_rate)
                val_err = sigma_rate / mu_rate 
                
                betas.append(beta)
                means.append(val)
                stds.append(val_err)
                
                # Keep track of raw numbers for separate individual plots
                raw_temps.append(T)
                raw_means.append(mu_rate)
                raw_stds.append(sigma_rate)
                
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error parsing {json_path}: {e}")

    return np.array(betas), np.array(means), np.array(stds), raw_temps, raw_means, raw_stds


def plot_individual_temperatures(temps, means, stds, output_dir):
    """Generates and saves a separate SVG plot for each individual temperature."""
    for T, mean, std in zip(temps, means, stds):
        plt.figure(figsize=(5, 5))
        
        # Plotting the individual rate value with its standard deviation
        plt.errorbar(
            [T], [mean], yerr=[std],
            fmt='o', color='#e74c3c', ecolor='#2c3e50',
            elinewidth=2, capsize=5, markersize=8, label=f'T = {T} K'
        )
        
        plt.xlabel('Temperature (K)')
        plt.ylabel('Rate')
        plt.title(f'Rate Distribution at {T} K')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        
        # Text box displaying precise values
        stats_text = f"Mean: {mean:.4e}\nStd Dev: {std:.4e}"
        plt.gca().text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        sep_svg_name = os.path.join(output_dir, f"temp_{int(T)}_stats.svg")
        plt.savefig(sep_svg_name, format='svg', dpi=300)
        plt.close()
        print(f"Separate plot saved as: {sep_svg_name}")


def plot_and_fit(betas, means, stds, output_name):
    # Perform Linear Regression: y = mx + c
    res = linregress(betas, means)
    
    ea = res.slope
    ea_err = res.stderr  
    
    tau = np.exp(res.intercept)
    tau_err = tau * res.intercept_stderr
    
    fit_line = res.slope * betas + res.intercept

    plt.figure(figsize=(9, 7))
    
    # Plot data points
    plt.errorbar(
        betas, means, yerr=stds, 
        fmt='o', color='#2c3e50', ecolor='#e74c3c', 
        elinewidth=1, capsize=3, label='Data (Mean $\pm$ Std)'
    )
    
    # Plot Regression Line
    plt.plot(betas, fit_line, '--', color='#2980b9', 
             label=f'Fit: $R^2={res.rvalue**2:.4f}$')
    
    # Text Box for results
    stats_text = (
        f"$E_a$ (slope): {ea:.4f} $\pm$ {ea_err:.4f} eV\n"
        f"Intercept: {res.intercept:.4f} $\pm$ {res.intercept_stderr:.4f}\n"
        f"$\\tau$: {tau:.3e} $\pm$ {tau_err:.3e}"
    )
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.xlabel(r'$\beta$ ($eV^{-1}$)')
    plt.ylabel(r'$-\ln(\text{rate})$')
    plt.title('Arrhenius Analysis with Linear Regression')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_name, format='svg', dpi=300)
    plt.close() # Close figure to avoid memory leaks
    
    print(f"\n--- Global Fit Results ---")
    print(f"Activation Energy (Ea): {ea:.6f} +/- {ea_err:.6f} eV")
    print(f"Tau: {tau:.6e} +/- {tau_err:.6e}")
    print(f"Main plot saved as: {output_name}")


def main():
    parser = argparse.ArgumentParser(description="Linear regression on -log(rate) vs Beta.")
    parser.add_argument("directory", type=str, help="Path to data directory.")
    parser.add_argument("-o", "--output", type=str, default="arrhenius_fit.svg")

    args = parser.parse_args()
    result = load_data(args.directory)
    
    if result and len(result[0]) > 0:
        betas, means, stds, raw_temps, raw_means, raw_stds = result
        
        # 1. Generate the master Arrhenius regression plot
        plot_and_fit(betas, means, stds, args.output)
        
        # 2. Generate isolated SVG plots for each distinct temperature directory
        plot_individual_temperatures(raw_temps, raw_means, raw_stds, os.path.dirname(args.output) or '.')
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()