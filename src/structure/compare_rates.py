import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Constants
K_B_EV = 8.617333262e-5  # Boltzmann constant in eV/K

import glob
import json
from pathlib import Path

def parse_data(target_dir):
    data_list = []
    
    # Convert input string to a Path object for easier manipulation
    base_path = Path(target_dir)
    
    # Define the pattern: target_dir/*/RATE_OUT/stats.json
    # rglob is recursive, but since we have a specific structure, 
    # we use glob on the path object.
    path_pattern = str(base_path / "*" / "RATE_OUT" / "stats.json")
    files = glob.glob(path_pattern)
    
    if not files:
        print(f"No stats.json files found in {path_pattern}")
        return None

    for file_path in files:
        p = Path(file_path)
        
        # The strain is the name of the folder inside target_dir
        # Structure: target_dir / <strain_folder> / RATE_OUT / stats.json
        # p.parents[1] gives the <strain_folder> path
        try:
            strain_folder_name = p.parents[1].name
            strain = float(strain_folder_name)
        except (ValueError, IndexError):
            print(f"Skipping {file_path}: Could not parse strain from folder name.")
            continue
            
        try:
            with open(file_path, 'r') as f:
                stats = json.load(f)
                
                # Handling the nested list structure: [[val], [val]]
                samples = [s[0] for s in stats['samples'] if isinstance(s, list) and len(s) > 0]
                
                data_list.append({
                    'strain': strain,
                    'q025': stats['q0_025'],
                    'q500': stats['q0_500'],
                    'q975': stats['q0_975'],
                    'mean': stats['mean'],
                    'samples': samples
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Sort by strain value (0.20, 0.201, 0.202, etc.)
    return sorted(data_list, key=lambda x: x['strain'])

# Usage:
# results = parse_data("./thesis_data")

def plot_results(data, temp):
    strains = [d['strain'] for d in data]
    q500 = np.array([d['q500'] for d in data])
    q025 = np.array([d['q025'] for d in data])
    q975 = np.array([d['q975'] for d in data])
    
    # Constants for conversion
    kbT = K_B_EV * temp
    
    # --- Data for Plot 1 (Rates) ---
    yerr_rate = [q500 - q025, q975 - q500]

    # --- Data for Plot 3 (Energy Barrier in eV) ---
    # Formula: E = -kbT * ln(rate)
    energy_median = -kbT * np.log(q500)
    energy_low = -kbT * np.log(q975)  # High rate = Low energy
    energy_high = -kbT * np.log(q025) # Low rate = High energy
    
    yerr_energy = [energy_median - energy_low, energy_high - energy_median]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # --- Plot 1: Strain vs Breaking Rate ---
    axes[0].errorbar(strains, q500, yerr=yerr_rate, fmt='o', color='#2c3e50', 
                     ecolor='#3498db', elinewidth=2, capsize=4, label='Median + 95% CI')
    
    # for x, y, s in zip(strains, q500, strains):
    #     axes[0].annotate(f'ε={s:.3f}', (x, y), textcoords="offset points", 
    #                      xytext=(0,10), ha='center', fontsize=8, fontweight='bold')

    axes[0].set_title('Strain vs. Breaking Rate', fontweight='bold')
    axes[0].set_xlabel('Strain [a.u.]')
    axes[0].set_ylabel('Breaking Rate [1/ps]')
    axes[0].legend()

    # --- Plot 2: Overlaid Histograms ---
    colors = sns.color_palette("viridis", len(data))
    for i, d in enumerate(data):
        sns.kdeplot(d['samples'], ax=axes[1], fill=True, color=colors[i], label=f"ε: {d['strain']}")
    axes[1].set_title('Rate Distribution', fontweight='bold')
    axes[1].set_xlabel('Breaking Rate [1/ps]')
    axes[1].set_ylabel('Density')
    if len(data) <= 20:
        axes[1].legend(fontsize='x-small', bbox_to_anchor=(1.02, 1), loc='upper left')

    # --- Plot 3: -kbT * log(Rate) vs Strain with Error Bars ---
    axes[2].errorbar(strains, energy_median, yerr=yerr_energy, fmt='s', color='#e74c3c', 
                     ecolor='#c0392b', elinewidth=1.5, capsize=4, linestyle='--', 
                     alpha=0.8, label=r'$-k_B T\, \ln(\gamma)$')
    axes[1].set_xscale("log")
    
    # for x, y, s in zip(strains, energy_median, strains):
    #     axes[2].annotate(f'ε={s:.3f}', (x, y), textcoords="offset points", 
    #                      xytext=(0,10), ha='center', fontsize=8, color='#c0392b')

    axes[2].set_title(f'Energy Barrier Approximation ({temp}K)', fontweight='bold')
    axes[2].set_xlabel('Strain')
    axes[2].set_ylabel(r'Energy Barrier [eV]')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("RateComparison.pdf")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process breaking rate stats.')
    parser.add_argument('dir', type=str, help="root directory with rate subdirectories")
    parser.add_argument('temp', type=float, help='Temperature in Kelvin')
    args = parser.parse_args()

    processed_data = parse_data(args.dir)
    if processed_data:
        plot_results(processed_data, args.temp)


if __name__ == "__main__":
    main()