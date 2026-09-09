import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Configuration & Data Extraction
target_pattern = re.compile(r"model_T(\d+)_b8f_l(\d+)_pT_f16_rd1_rw16")
aggregated_data = {}

for item in os.listdir('.'):
    match = target_pattern.match(item)
    if match and os.path.isdir(item):
        T_val = int(match.group(1))
        l_val = int(match.group(2))
        
        csv_path = os.path.join(item, "logger", "version_1", "metrics.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(item, "logger", "version_0", "metrics.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(item, "metrics.csv")
            
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                val_force_col = "val0_epoch/forces_rmse"
                val_energy_col = "val0_epoch/per_atom_energy_rmse"
                
                if val_force_col in df.columns and val_energy_col in df.columns:
                    df_clean = df.dropna(subset=[val_force_col, val_energy_col])
                    
                    if not df_clean.empty:
                        best_row_idx = df_clean[val_force_col].idxmin()
                        best_row = df_clean.loc[best_row_idx]
                        
                        f_rmse = float(best_row[val_force_col])
                        e_rmse = float(best_row[val_energy_col])
                        
                        aggregated_data[item] = {
                            "T": T_val,
                            "l": l_val,
                            "E_per_atom_RMSE": e_rmse,
                            "aggregated_F_comp_RMSE": f_rmse
                        }
                    else:
                        print(f"[WARNING] CSV empty after removing NaNs in {csv_path}")
                else:
                    print(f"[WARNING] Validation targets missing from columns in {csv_path}")
                    
            except Exception as e:
                print(f"Error parsing {csv_path}: {e}")
        else:
            print(f"[INFO] No metrics.csv found for directory {item}")

output_json = "summary_metrics_b8f_rd1_rw16.json"
with open(output_json, 'w') as f:
    json.dump(aggregated_data, f, indent=4)
print(f"Aggregated data saved cleanly to {output_json}")


# 2. Reading and Preparing Data for Plotting
with open(output_json, 'r') as f:
    plot_data = json.load(f)

series_by_l = {}
for model, metrics in plot_data.items():
    l_val = metrics["l"]
    if l_val not in series_by_l:
        series_by_l[l_val] = {"T": [], "E_val": [], "F_val": []}
    
    series_by_l[l_val]["T"].append(metrics["T"])
    series_by_l[l_val]["E_val"].append(metrics["E_per_atom_RMSE"])
    series_by_l[l_val]["F_val"].append(metrics["aggregated_F_comp_RMSE"])

for l_val in series_by_l:
    idx = np.argsort(series_by_l[l_val]["T"])
    series_by_l[l_val]["T"] = np.array(series_by_l[l_val]["T"])[idx]
    series_by_l[l_val]["E_val"] = np.array(series_by_l[l_val]["E_val"])[idx]
    series_by_l[l_val]["F_val"] = np.array(series_by_l[l_val]["F_val"])[idx]


# 3. Generating Plots
# plt.rcParams.update({
#     "font.size": 11,
#     "axes.labelsize": 12,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 11,
# })

# 3. Generating Plots

# Calculate the number of columns dynamically based on your data
num_cols = len(series_by_l)

# --- Plot 1: Energy per Atom RMSE vs T ---
plt.figure(figsize=(7, 5.2)) # Slightly increased height to accommodate the top legend
for l_val, data in sorted(series_by_l.items()):
    # The label is just the number to fit your requested "lmax  0  1  2" look
    plt.plot(data["T"], data["E_val"], '-o', label=f"{l_val}", linewidth=1.5, markersize=7)

plt.xlabel(r"$\mathrm{Number\ of\ Layers\ (T)}$")
plt.ylabel(r"$E/\mathrm{atom\ RMSE}\ [\mathrm{eV/atom}]$")
plt.xticks(sorted(list(set(T for d in series_by_l.values() for T in d["T"]))))
plt.grid(True, linestyle='--', alpha=0.6)

# Horizontal tabular legend configuration
plt.legend(
    title=r"$l_{\mathrm{max}}$:", 
    ncol=num_cols, 
    loc="upper left",      # Places it centered at the top
    #bbox_to_anchor=(0.5, 1.15), # Pushes it slightly above the plot area so it doesn't overlap data
    columnspacing=1.5,       # Controls spacing between columns
    handletextpad=0.5,       # Controls spacing between the line and its label
    frameon=True
)

plt.tight_layout()
plt.savefig("NequipGrapheneEnergyRmse.pdf", format="pdf")
plt.show()


# --- Plot 2: Aggregated Force Component RMSE vs T ---
plt.figure(figsize=(7, 5.2))
for l_val, data in sorted(series_by_l.items()):
    plt.plot(data["T"], data["F_val"], '-o', label=f"{l_val}", linewidth=1.5, markersize=7)

plt.xlabel(r"$\mathrm{Number\ of\ Layers\ (T)}$")
plt.ylabel(r"$F_{\mathrm{comp}}\ \mathrm{RMSE}\ [\mathrm{eV/\AA}]$")
plt.xticks(sorted(list(set(T for d in series_by_l.values() for T in d["T"]))))
plt.grid(True, linestyle='--', alpha=0.6)

# Horizontal tabular legend configuration
plt.legend(
    title=r"$l_{\mathrm{max}}$:", 
    ncol=num_cols, 
    loc="upper left", 
    #bbox_to_anchor=(0.5, 1.15), 
    columnspacing=1.5, 
    handletextpad=0.5, 
    frameon=True
)

plt.tight_layout()
plt.savefig("NequipGrapheneForceRmse.pdf", format="pdf")
plt.show()
