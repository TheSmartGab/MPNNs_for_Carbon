
from lib import ph_mp, ph_ase, ph_phonopy

# interface to other libraries, each format must have its method implemented in its dedicated file
FORMATS_STRUCTURE = {
    "ase" : {
        "get_kpoints_frequencies": ph_ase.get_kpoints_frequencies,
    },
    "mp" : {
        "get_kpoints_frequencies" : ph_mp.get_kpoints_frequencies,
    },
    "phonopy" : {
        "get_kpoints_frequencies" : ph_phonopy.get_kpoints_frequencies
    }
}