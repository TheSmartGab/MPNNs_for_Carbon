import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
# use np.correlate(x, x, mode = 'same')

from ase.io import read
from ase import Atoms

import re

def parse_energy_stress_from_file(file_path) -> tuple[np.array, np.array, np.array, np.array]:

    etot, ekin, epot = [], [], []
    stress = []
    traj = read(file_path, index=':')    
    for frame in traj:
        etot.append(frame.get_total_energy())
        ekin.append(frame.get_kinetic_energy())
        epot.append(frame.get_potential_energy())
        stress.append(frame.get_stress(voigt = False))  # full tensor   

    return np.array(etot), np.array(ekin), np.array(epot), np.array(stress)

def power_spectrum(corr: np.array, dt: float = 1.0) -> tuple[np.array, np.array]:
    """
    Compute normalized one-sided power spectrum from an autocorrelation function.
    """
    N = len(corr)
    # FFT and normalization (Parseval's theorem)
    spectrum = np.fft.fft(corr)
    ps = (np.abs(spectrum)**2) / N

    # Frequency axis (in 1/dt units)
    freqs = np.fft.fftfreq(N, d=dt)

    # Keep only positive frequencies
    mask = freqs >= 0
    return freqs[mask], ps[mask]

def plot_frame(etot: np.array, ekin: np.array, epot: np.array, stress: np.array, outputdir: str, filename: str):
    
    time = np.arange(len(etot))

    # === Energy plot with dual y-axis ===
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_etot = 'tab:blue'
    color_epot = 'tab:green'
    color_ekin = 'tab:orange'

    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Energy (eV)', color=color_etot)
    ax1.plot(time, etot, label='Total Energy', color=color_etot)
    ax1.plot(time, epot, label='Potential Energy', color=color_epot)
    ax1.tick_params(axis='y', labelcolor=color_etot)
    ax1.grid()

    # Secondary y-axis for kinetic energy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Kinetic Energy (eV)', color=color_ekin)
    ax2.plot(time, ekin, label='Kinetic Energy', color=color_ekin, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_ekin)

    # Combined legend from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Energy vs Frame')
    fig.tight_layout()
    plt.savefig(os.path.join(outputdir, f"{filename}_energies.pdf"))
    plt.close()

    # === Stress components ===
    plt.figure(figsize=(10, 6))
    for i in range(3):
        for j in range(3):
            plt.plot(time, stress[:, i, j], label=f'Stress {i}{j}')
    plt.xlabel('Frame')
    plt.ylabel('Stress (eV/Å³)')
    plt.title('Stress Components vs Frame')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f"{filename}_stress.pdf"))
    plt.close()


def main():

    parser = argparse.ArgumentParser(description='Plot self-correlation and power spectrum of energies from .xyz files')
    parser.add_argument('input_dir', type=str, help='Path to the input directory with xyz files')
    parser.add_argument('output_dir', type=str, help='Directory to save output figures of single files')
    parser.add_argument('--min_frames', type=int, default=1, help='Minimum number of frames required to process a file')

    args = parser.parse_args()
    args.input_dir = os.path.normpath(args.input_dir)
    base_dir = '/'.join(args.input_dir.split('/')[:-1])
    info_dir = os.path.join(base_dir, 'info')

    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if f.endswith('.xyz') or f.endswith('.extxyz') or f.endswith('.traj')]

    # === Process each file one by one ===
    for file in files:
        file_path = os.path.join(args.input_dir, file)
        etot, ekin, epot, stress  = parse_energy_stress_from_file(file_path)

        if len(etot) < args.min_frames:
            print(f"[WARNING] Skipping {file} with only {len(etot)} frames")
            continue
        print(f"[INFO] Processing {file} with {len(etot)} frames")
        name, ext = os.path.splitext(file)
        plot_frame(etot, ekin, epot, stress, args.output_dir, name)

if __name__ == '__main__':
    main()