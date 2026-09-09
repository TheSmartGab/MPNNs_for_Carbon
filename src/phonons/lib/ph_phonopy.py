"functions to plot phonopy style data (band.yaml)"
import numpy as np

UNITS = {"kpoints": "frac", "frequencies": "THz"}

# keep fractional coordinates
def convert_kpoints(kpoints):
    return kpoints

# convert to THz
def convert_frequencies(frequencies):
    return frequencies


def get_kpoints_frequencies(data):
    phonons = data["phonon"]
    nbands = len(phonons[0]["band"])

    kpoints = []
    freqs = []

    for ph in phonons:
        kpoints.append(ph['q-position'])
        fs = [entry['frequency'] for entry in ph['band']]
        freqs.append(fs)

    kpoints = np.vstack(kpoints)
    freqs = np.vstack(freqs)
    freqs = freqs.reshape(1, len(kpoints), nbands)

    kpoints = convert_kpoints(kpoints)
    freqs = convert_frequencies(freqs)

    return kpoints, freqs