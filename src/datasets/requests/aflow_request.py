import os, sys, requests, json
from pprint import pprint
from pymatgen.core import Composition, Lattice 

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_OUTPUT = '../../C/aflow/all/'
os.makedirs(BASE_OUTPUT, exist_ok=False)
FILES_OUTPUT = os.path.join(BASE_OUTPUT, 'files/')
os.makedirs(FILES_OUTPUT, exist_ok=False)
INFO_OUTPUT = os.path.join(BASE_OUTPUT, 'info/')
os.makedirs(INFO_OUTPUT, exist_ok=False)

VALID_ENTRIES_FILE = open(os.path.join(INFO_OUTPUT, 'ValidEntries.txt'), 'w')
VALID_ENTRIES_FILE.write('AUID\t\t\t\t\tDFT_type\tChemical_Hill\tN_Atoms\tEnergy(eV)\n')
COUNT_FILE = open(os.path.join(INFO_OUTPUT, 'counts.txt'), 'w')
counter = 0
NOT_FOUND_FILE = open(os.path.join(INFO_OUTPUT, 'NotFound.txt'), 'w')

# Path to your summon URL file
summons_path = './aflux_summons.txt'
with open(summons_path, 'r') as f:
    summon = f.read().strip()
print(f'[INFO]: Summon: {summon}')
SUMMON_LOG = os.path.join(INFO_OUTPUT, 'summon.txt')
with open(SUMMON_LOG, 'w') as f:
    f.write(summon + '\n')

def aflow_request(summon):
    try:
        response = requests.get(summon)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f'[ERROR]: {e}')
        sys.exit(1)

from pymatgen.core import Lattice

def lattice_to_extxyz_string(geometry):
    """
    Convert AFLOW 'geometry' list [a, b, c, alpha, beta, gamma]
    into a flattened extxyz-compatible Lattice string.
    """
    if not geometry or len(geometry) != 6:
        raise ValueError(f"Invalid geometry: {geometry}")

    a, b, c, alpha, beta, gamma = geometry
    # create pymatgen lattice
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    # flatten to a single line "a1x a1y a1z a2x a2y a2z a3x a3y a3z"
    lattice_flat = ' '.join(f"{v:.8f}" for vec in lattice.matrix for v in vec)
    return lattice_flat

def expand_species(species_list, composition):
    """
    Expand species by composition counts.
    e.g. species=['C','O'], composition=[2,1] → ['C','C','O']
    """
    expanded = []
    for el, n in zip(species_list, composition):
        expanded.extend([el] * int(n))
    return expanded

def make_aflow_json_url(aurl: str) -> str:
    if aurl.startswith("aflowlib.duke.edu:"):
        # Replace ONLY the first colon after the hostname
        path = aurl.replace("aflowlib.duke.edu:", "aflowlib.duke.edu/", 1)
        return f"https://{path}/aflowlib.json"
    elif aurl.startswith("http"):
        # Already a URL
        return f"{aurl.rstrip('/')}/aflowlib.json"
    else:
        # Fallback — assume user gave path-like string
        return f"https://aflowlib.duke.edu/{aurl.strip('/')}/aflowlib.json"

def process_aurl_info(aurl):
    global counter, VALID_ENTRIES_FILE, COUNT_FILE
    print
    # Ensure proper https URL
    # full_url = f"https://{'/'.join(aurl.split(':'))}/aflowlib.json"
    full_url = make_aflow_json_url(aurl)

    try:
        r = requests.get(full_url, headers={"Accept":"application/json"}, verify = False)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(f'[ERROR]: {e} for {full_url}')
        NOT_FOUND_FILE.write(f"{aurl}, {e}\n")
        return

    # Extract useful info
    auid = data.get('auid')
    compound = data.get('compound')
    chemical_hill = Composition(compound).hill_formula
    species_set = data.get('species', [])
    stoichiometry = data.get('composition', [])
    species_list = expand_species(species_set, stoichiometry)
    positions = data.get('positions_cartesian', [])
    forces = data.get('forces', [])
    energy = data.get('energy_cell', data.get('enthalpy_cell', 0.0))
    dft = data.get('dft_type', 'None')
    lattice = data.get('geometry', [])  # a,b,c,alpha,beta,gamma

    if not positions or not lattice:
        print(f'[WARNING]: Missing positions or lattice for {auid}')
        return

    try:
        lattice_flat = lattice_to_extxyz_string(data.get('geometry'))
    except ValueError as e:
        print(f"[WARNING]: {e}")
        return

    # Build extxyz content
    n_atoms = data.get('natoms', len(positions))
    header = f'{n_atoms}\nLattice="{lattice_flat}" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.8f} pbc="T T T"\n'
    lines = []
    for element, pos, force in zip(species_list, positions, forces):
        line = f"{element} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} {force[0]:.8f} {force[1]:.8f} {force[2]:.8f}"
        lines.append(line)

    content = header + '\n'.join(lines)

    # Write to file
    filename = f"{auid.replace(':','_')}.xyz"
    with open(os.path.join(FILES_OUTPUT, filename), 'w') as f:
        f.write(content)
        counter += 1

    VALID_ENTRIES_FILE.write(f"{auid}\t{dft}\t{chemical_hill}\t\t\t{n_atoms}\t\t{energy:.8f}\n")
    COUNT_FILE.write(f"counts : {counter}\n")

def main():
    data = aflow_request(summon)
    for entry in data:
        aurl = entry.get('aurl')
        if aurl:
            process_aurl_info(aurl)

    COUNT_FILE.close()
    VALID_ENTRIES_FILE.close()
    NOT_FOUND_FILE.close()

if __name__ == '__main__':
    main()
