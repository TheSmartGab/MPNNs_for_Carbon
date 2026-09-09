import os
import glob
import re  # Added for matching the folder names
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

DEG2RAD = np.pi / 180.

def parse_args():
    parser = ArgumentParser(description="Traverses directories, extracts rotation angles, and plots combined boundaries.")
    parser.add_argument("--root_dir", "-r", default=".", help="Root directory containing R0, R1... subfolders")
    parser.add_argument("--pattern", "-p", default="*summary*.csv", help="Glob pattern to identify target summary CSVs")
    parser.add_argument("--output_name", "-o", default="all_boundaries_combined.png", help="Filename for the combined plot output")
    return parser.parse_args()

def extract_rotation_angle(folder_name):
    """
    Looks for patterns like 'R0', 'R15', 'R180' in the folder name.
    Returns the integer angle if found, otherwise returns None.
    """
    match = re.search(r'R(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None

def collect_and_plot_all_boundaries(root_dir, pattern, output_name):
    search_path = os.path.join(root_dir, "**", pattern)
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{pattern}' inside '{root_dir}'.")
        return

    # 1. First pass: Collect valid files and parse their rotation angles
    valid_datasets = []
    for file_path in csv_files:
        folder_name = os.path.basename(os.path.dirname(file_path)) or ""
        angle = extract_rotation_angle(folder_name)
        
        if angle is not None:
            valid_datasets.append({
                'file_path': file_path,
                'angle': angle,
                'label': f"Rotation: {angle}°"
            })
        else:
            # Fallback if a directory doesn't fit the RX naming style
            valid_datasets.append({
                'file_path': file_path,
                'angle': 999,  # Push unsorted/unknown elements to the back
                'label': folder_name or "Unknown Setup"
            })

    # 2. Sort datasets numerically by their actual rotation angle 
    valid_datasets = sorted(valid_datasets, key=lambda x: x['angle'])
    
    print(f"Found and sorted {len(valid_datasets)} rotation data paths.")
    
    # Setup master figure
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # We will use a standard color cycle/map for the trajectory lines to keep them distinct
    line_cmap = plt.cm.get_cmap('tab10', len(valid_datasets))
    
    # Global tracking variables for the stress colorbar mapping
    all_stresses = []
    sc_objects = []
    plotted_any = False

    # 3. Plotting loop using our sorted tracking arrays
    for idx, item in enumerate(valid_datasets):
        file_path = item['file_path']
        dir_label = item['label']
        
        try:
            df = pd.read_csv(file_path)
            
            if 'boundary_found' not in df.columns or 'critical_amplitude' not in df.columns:
                continue
                
            df_valid = df[df['boundary_found'] == True].dropna(subset=['critical_amplitude']).copy()
            if df_valid.empty:
                continue
            
            # Compute physical parameters
            xs = df_valid['critical_amplitude'] * np.cos(DEG2RAD * df_valid['angle_deg'])
            y_vals = df_valid['critical_amplitude'] * np.sin(DEG2RAD * df_valid['angle_deg'])
            
            # Replicate your clean stress_norm mapping logic from original file
            stress_norm = np.sqrt(
                df_valid['boundary_stress_xx']**2 +
                df_valid['boundary_stress_yy']**2 +
                2 * (df_valid['boundary_stress_xy']**2)
            ) * 100
            
            sort_idx = np.argsort(df_valid['angle_deg'].values)
            xs_sorted = xs.values[sort_idx]
            ys_sorted = y_vals.values[sort_idx]
            stress_sorted = stress_norm.values[sort_idx]
            
            if np.abs(df_valid['angle_deg'].max() - df_valid['angle_deg'].min()) > 300:
                xs_sorted = np.append(xs_sorted, xs_sorted[0])
                ys_sorted = np.append(ys_sorted, ys_sorted[0])
                stress_sorted = np.append(stress_sorted, stress_sorted[0])
            
            # Draw structural framework connection line
            line_color = line_cmap(idx)
            ax.plot(xs_sorted, ys_sorted, linestyle='-', linewidth=1, color=line_color, label=dir_label, alpha=0.7, zorder = 4)
            
            # Scatter structural points tracked to the continuous scientific colormap 
            sc = ax.scatter(
                xs_sorted, ys_sorted, 
                c=stress_sorted, cmap='viridis', 
                edgecolors='k', s=35, zorder=3
            )
            sc_objects.append(sc)
            all_stresses.extend(stress_sorted)
            
            plotted_any = True
            
        except Exception as e:
            print(f"Skipping file due to reading error: {file_path}. Details: {e}")

    if not plotted_any:
        print("No valid datasets containing complete boundaries could be built.")
        plt.close(fig)
        return

    # Normalize colormaps globally across ALL sub-directories cleanly
    min_stress, max_stress = min(all_stresses), max(all_stresses)
    for sc in sc_objects:
        sc.set_clim(min_stress, max_stress)

    # Add Colorbar scaled to our stress metrics global spectrum
    cbar = fig.colorbar(sc_objects[0], ax=ax, pad=0.02)
    cbar.set_label(r'planar stress l2 norm [$eV/\AA^2$] ')

    # Aesthetics and styling (keeping your exact bounds updates)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, zorder=1)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, zorder=1)
    ax.set_xlabel('$\epsilon_{xx}$')
    ax.set_ylabel('$\epsilon_{yy}$')
    
    ticks = np.linspace(0, 0.25, 6)
    labels = [f"{t:.2f}" for t in ticks]

    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)

    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Forces absolute uniform spatial scaling matching standard unit geometry expectations
    ax.set_aspect('equal', adjustable='box')
    
    # Internal Legend position override configuration
    ax.legend(loc="lower left", title="Rotation Angle", frameon=True, facecolor='white', framealpha=0.9, ncol=2, fontsize = 13)

    # Keeping your customized constraints explicitly intact
    ax.set_xlim(0, 0.26)
    ax.set_ylim(0, 0.26)
    
    output_path = os.path.join(root_dir, output_name)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nMaster aggregate figure successfully saved to: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    collect_and_plot_all_boundaries(args.root_dir, args.pattern, args.output_name)