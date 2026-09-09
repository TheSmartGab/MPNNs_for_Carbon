"functions to plot ase style data"

import conversions

import numpy as np


UNITS = {"kpoints": "frac", "frequencies": "eV"}


# keep fractional coordinates
def convert_kpoints(kpoints):
    return kpoints


# convert to THz
def convert_frequencies(frequencies):
    return frequencies * conversions.EV_TO_THZ

def extract_ndarray(obj):
    v = obj[2]
    shape = obj[0]
    dtype = obj[1]

    return np.reshape(v, shape=shape).astype(dtype)


def get_kpoints(data):

    ndarray = data["path"]["kpts"]["__ndarray__"]
    shape = ndarray[0]
    dtype = ndarray[1]
    kpoints = extract_ndarray(ndarray)

    kpoints = convert_kpoints(kpoints)

    return kpoints


def get_frequencies(data):
    frequencies = extract_ndarray(data['energies']['__ndarray__'])
    frequencies = convert_frequencies(frequencies)

    return frequencies

def get_kpoints_frequencies(data):
    return get_kpoints(data), get_frequencies(data)