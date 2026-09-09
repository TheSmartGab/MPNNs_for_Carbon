import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
# use np.correlate(x, x, mode = 'same')

import re

def parse_energy_from_file(file_path) -> np.array:

    energies = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r'energy=([-+]?\d*[\.]?\d*|\d*)', line)
            if match:
                energy = float(match.group(1))
                energies.append(energy)

    return np.array(energies)

def self_correlation(x: np.array, max_lag: int = None) -> np.array:

    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]  # take the second half
    result = result[:max_lag + 1]  # limit to max_lag
    result /= np.arange(n, n - max_lag - 1, -1)  # normalize by number of overlapping points
    result /= result[0]  # normalize by zero-lag value

    return result

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


def main():

    parser = argparse.ArgumentParser(description='Plot self-correlation and power spectrum of energies from .xyz files')
    parser.add_argument('input_dir', type=str, help='Path to the input directory with xyz files')
    parser.add_argument('output_dir', type=str, help='Directory to save output figures of single files')
    parser.add_argument('--max_lag', type=int, default=None, help='Maximum lag for self-correlation')
    parser.add_argument('--min_frames', type=int, default=100, help='Minimum number of frames required to process a file')

    args = parser.parse_args()
    args.input_dir = os.path.normpath(args.input_dir)
    base_dir = '/'.join(args.input_dir.split('/')[:-1])
    info_dir = os.path.join(base_dir, 'info')

    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if f.endswith('.xyz')]

    # Combined summary plots
    fig_corr = plt.figure(figsize=(10, 6))
    ax_corr = fig_corr.add_subplot(1, 1, 1)
    ax_corr.set_xlabel('Lag')
    ax_corr.set_ylabel('Self-correlation')
    ax_corr.set_title('Self-correlation of Energies')
    ax_corr.grid(True)

    fig_spectrum = plt.figure(figsize=(10, 6))
    ax_spectrum = fig_spectrum.add_subplot(1, 1, 1)
    ax_spectrum.set_xlabel('Frequency')
    ax_spectrum.set_ylabel('Power Spectrum')
    ax_spectrum.set_title('Power Spectrum of Energies')
    ax_spectrum.set_xscale('log')
    ax_spectrum.set_yscale('log')
    ax_spectrum.grid(True)

    # === Process each file one by one ===
    for file in files:
        file_path = os.path.join(args.input_dir, file)
        energies = parse_energy_from_file(file_path)

        if len(energies) < args.min_frames:
            print(f"[WARNING] Not enough frames ({len(energies)}) in {file_path}, skipping.")
            continue

        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]

        corr = self_correlation(energies, max_lag=args.max_lag)
        freqs, spectrum = power_spectrum(corr)
        spectrum /= np.max(spectrum)  # normalize shape for comparison

        # Update summary figures
        lags = np.arange(len(corr))
        ax_corr.plot(lags, corr, alpha=0.3)
        ax_spectrum.plot(freqs, spectrum, alpha=0.3)

        # === Create and save per-file figure ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(lags, corr, color='C0')
        ax1.set_title(f"Autocorrelation\n{file}")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("Correlation")
        ax1.grid(True, alpha=0.3)

        ax2.plot(freqs, spectrum, color='C1')
        ax2.set_title("Power Spectrum")
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Normalized Power")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save immediately and close
        output_path = os.path.join(args.output_dir, f"{file_id}.pdf")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)  # free memory


    # === Save summary plots ===
    corr_path = os.path.join(info_dir, 'self_correlation_summary.pdf')
    spectrum_path = os.path.join(info_dir, 'power_spectrum_summary.pdf')

    fig_corr.savefig(corr_path)
    fig_spectrum.savefig(spectrum_path)
    plt.close('all')





if __name__ == '__main__':
    main()