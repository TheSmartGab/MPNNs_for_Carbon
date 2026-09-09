import argparse
import json
import collections
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# MACE color palette matching the native script
colors = [
    "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

def parse_path(path):
    name_re = re.compile(r"(?P<name>.+)_run-(?P<seed>\d+)_train.txt")
    match = name_re.match(os.path.basename(path))
    if not match:
        return os.path.splitext(os.path.basename(path))[0]
    return match.group("name")

def parse_and_plot(file_path, min_epoch=0, start_swa=None, linear=False, batch_size=8):
    # Structures to hold aggregated metrics per epoch
    train_epochs = []
    train_metrics = collections.defaultdict(list)
    val_epochs = []
    val_metrics = collections.defaultdict(list)

    current_epoch = None
    batch_metrics = collections.defaultdict(list)

    # Comprehensive target keys list from MACE
    target_keys = [
        'loss', 'mae_e', 'mae_e_per_atom', 'rmse_e', 'rmse_e_per_atom', 'q95_e',
        'mae_f', 'rel_mae_f', 'rmse_f', 'rel_rmse_f', 'q95_f',
        'mae_stress', 'rmse_stress', 'q95_stress', 'rmse_virials_per_atom',
        'mae_virials', 'rmse_mu_per_atom'
    ]

    def flush_training_epoch():
        """Averages the accumulated batch metrics for the completed training epoch."""
        if current_epoch is not None and current_epoch > min_epoch:
            train_epochs.append(current_epoch)
            for key in target_keys:
                if batch_metrics[key]:
                    if key == 'loss':
                        # Clean batch-size normalization scaling
                        normalized_losses = [l / batch_size for l in batch_metrics[key]]
                        avg_val = sum(normalized_losses) / len(normalized_losses)
                    else:
                        avg_val = sum(batch_metrics[key]) / len(batch_metrics[key])
                    train_metrics[key].append(avg_val)
                else:
                    train_metrics[key].append(None)
            batch_metrics.clear()

    total_lines = sum(1 for _ in open(file_path, 'r')) if os.path.exists(file_path) else 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Parsing MACE Log"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = data.get("mode")
            epoch = data.get("epoch")
            actual_epoch = epoch if epoch is not None else 0

            if actual_epoch <= min_epoch:
                continue

            if mode in ["train", "opt"]:
                if actual_epoch != current_epoch:
                    flush_training_epoch()
                    current_epoch = actual_epoch
                for key in target_keys:
                    if key in data:
                        batch_metrics[key].append(data[key])

            elif mode == "eval":
                val_epochs.append(actual_epoch)
                for key in target_keys:
                    val_metrics[key].append(data.get(key))

        flush_training_epoch()

    def filter_valid(epochs, metrics):
        valid_pairs = [(e, m) for e, m in zip(epochs, metrics) if m is not None]
        if not valid_pairs:
            return [], []
        return zip(*valid_pairs)

    # Dictionary representing clean UI label conversion matrices from MACE
    labels_lookup = {
        "loss": ("Total Loss", "Loss Value", "loss"),
        "mae_e": ("Energy MAE", "MAE E [meV]", "energy_mae"),
        "mae_e_per_atom": ("Energy MAE per Atom", "MAE E/atom [meV/atom]", "energy_mae_per_atom"),
        "rmse_e": ("Energy RMSE", "RMSE E [meV]", "energy_rmse"),
        "rmse_e_per_atom": ("Energy RMSE per Atom", "RMSE E/atom [meV/atom]", "energy_rmse_per_atom"),
        "q95_e": ("Energy Q95", "Q95 E [meV]", "energy_q95"),
        "mae_f": ("Force MAE", "MAE F [meV / A]", "force_mae"),
        "rel_mae_f": ("Relative Force MAE", "Relative MAE F [meV / A]", "force_rel_mae"),
        "rmse_f": ("Force RMSE", "RMSE F [meV / A]", "force_rmse"),
        "rel_rmse_f": ("Relative Force RMSE", "Relative RMSE F [meV / A]", "force_rel_rmse"),
        "q95_f": ("Force Q95", "Q95 F [meV / A]", "force_q95"),
        "mae_stress": ("Stress MAE", "MAE Stress", "stress_mae"),
        "rmse_stress": (r"Stress RMSE", "RMSE Stress [meV / $\AA^3$]", "stress_rmse"),
        "q95_stress": ("Stress Q95", "Q95 Stress", "stress_q95"),
        "rmse_virials_per_atom": ("Virials RMSE per Atom", "RMSE virials/atom [meV]", "virials_rmse_per_atom"),
        "mae_virials": ("Virials MAE", "MAE Virials [meV]", "virials_mae"),
        "rmse_mu_per_atom": ("MU RMSE per Atom", "RMSE MU/atom [mDebye]", "mu_rmse_per_atom"),
    }

    base_name = parse_path(file_path)
    dirname = os.path.dirname(file_path)

    # Generate contiguous sequence lookup dictionaries to hide index jumps
    all_unique_epochs = sorted(list(set(train_epochs + val_epochs)))
    epoch_to_visual = {raw: i for i, raw in enumerate(all_unique_epochs)}
    
    jump_visual_idx = None
    if start_swa is not None and start_swa in epoch_to_visual:
        jump_visual_idx = epoch_to_visual[start_swa]
    else:
        # Fallback to locate the jump if an explicit epoch parameter wasn't supplied
        for i in range(1, len(all_unique_epochs)):
            if all_unique_epochs[i] - all_unique_epochs[i-1] > 5:
                jump_visual_idx = i
                break

    for key in tqdm(target_keys, total=len(target_keys), desc="plotting keys"):
        t_ep, t_val = filter_valid(train_epochs, train_metrics[key])
        v_ep, v_val = filter_valid(val_epochs, val_metrics[key])

        if not t_ep and not v_ep:
            continue

        category, y_label, filename_suffix = labels_lookup.get(key, (key, key, key))
        fig, ax = plt.subplots(figsize=(7, 5))

        # Re-map standard epochs to contiguous array index keys
        t_ep_visual = [epoch_to_visual[e] for e in t_ep]
        v_ep_visual = [epoch_to_visual[e] for e in v_ep]

        has_data = False
        if t_ep:
            scale = 1e3 if key != 'loss' else 1.0
            scaled_t_val = [v * scale for v in t_val]
            ax.plot(t_ep_visual, scaled_t_val, label='Train', marker='o', color=colors[1], linewidth=1, markersize=3)
            has_data = True

        if v_ep:
            scale = 1e3 if key != 'loss' else 1.0
            scaled_v_val = [v * scale for v in v_val]
            ax.plot(v_ep_visual, scaled_v_val, label='Val', marker='s', linestyle='--', color=colors[0], linewidth=1, markersize=3)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        # Draw stage delimiter safely without text or legend collision
        if jump_visual_idx is not None:
            line_pos = jump_visual_idx - 0.5 
            ax.axvline(line_pos, color="black", linestyle="dashed", linewidth=1.2, alpha=0.8, label = "Start stage 2")
            
            # Text uses your managed thesis font scaling seamlessly now
            # ax.text(line_pos, 0.05, "Start Stage 2 ", color="black",
            #         horizontalalignment='right', verticalalignment='bottom',
            #         transform=ax.get_xaxis_transform(),
            #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

        # Clean X-axis ticking logic to prevent digit strings overlapping
        num_ticks = min(6, len(all_unique_epochs))
        tick_indices = np.linspace(0, len(all_unique_epochs) - 1, num_ticks, dtype=int)
        
        if jump_visual_idx is not None and jump_visual_idx not in tick_indices:
            closest_idx = np.abs(tick_indices - jump_visual_idx).argmin()
            tick_indices[closest_idx] = jump_visual_idx
            tick_indices = np.sort(tick_indices)

        ax.set_xticks(tick_indices)
        # Displaying sequential indices directly (0, 1, 2...) so that no numerical jumps appear
        ax.set_xticklabels([str(idx) for idx in tick_indices], rotation=45, ha='right')

        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        
        if not linear:
            ax.set_yscale('log')
            
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        
        # Dynamic location finding keeps the legend box away from active data lines and annotation boundaries
        ax.legend(loc='best')
        
        plt.tight_layout()
        out_filename = os.path.join(dirname, f"{base_name}_{filename_suffix}.pdf")
        plt.savefig(out_filename, format='pdf', bbox_inches='tight')
        plt.close(fig)

    print(f"\n[Success] Complete parsing execution. Isolated metric plots saved as individual PDFs.")

def main():
    parser = argparse.ArgumentParser(description="Plot isolated MACE metrics cleanly directly to unique PDFs.")
    parser.add_argument('--path', type=str, required=True, help='Path to log file (*_train.txt)')
    parser.add_argument('--min_epoch', default=0, type=int, help='Minimum epoch to parse')
    parser.add_argument('--start_stage_two', '--start_swa', default=None, type=int, dest="start_swa", help='Epoch SWA begins')
    parser.add_argument('--linear', action='store_true', help='Plot on linear scale instead of log')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to normalize training loss tracking')
    args = parser.parse_args()
    
    parse_and_plot(
        file_path=args.path, 
        min_epoch=args.min_epoch, 
        start_swa=args.start_swa, 
        linear=args.linear, 
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()