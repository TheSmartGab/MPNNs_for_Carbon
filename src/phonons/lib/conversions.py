from scipy.constants import Planck, e

# units to use, this is not really used for now, each format file should specify its units, and declare conversion coefficients
# usefull for plotting labels and avoid confusion
UNITS = {
    "kpoints" : "frac", # frac for fractional coordinates in term of reciprocal lattice vectors
    "frequencies" : "THz"
}

# frequencies in THz
EV_TO_THZ = e/Planck * 1e-12