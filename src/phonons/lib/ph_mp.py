"functions to plot mp (materials project) style data"

import numpy as np


UNITS = {"kpoints": "frac", "frequencies": "eV"}


# keep fractional coordinates
def convert_kpoints(kpoints):
    return kpoints


# convert to THz and adjust shape
def convert_frequencies(frequencies):
    frequencies=np.array(frequencies)
    print(frequencies)
    print(frequencies.shape)
    frequencies = np.transpose(frequencies, (1, 0))[None, :, :]

    return frequencies

def get_kpoints(data):

    kpoints = data.get("qpoints", [])
    kpoints = convert_kpoints(kpoints)

    return np.array(kpoints)


def get_frequencies(data):
    frequencies = data.get("frequencies")
    frequencies = convert_frequencies(frequencies)
    return frequencies

def get_kpoints_frequencies(data):
    return get_kpoints(data), get_frequencies(data)
