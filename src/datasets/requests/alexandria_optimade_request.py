from optimade.client import OptimadeClient
from pprint import pprint
import json
import numpy as np
import pandas as pd
import os
from pymatgen.core import Composition

PREFIX = "_alexandria_"
# non optimade standard keys
keys = {
    'energy': 'energy_corrected',
    'forces': 'forces',
}
for key in list(keys.keys()):
    keys[key] = PREFIX + keys[key]

# additional database specific keys 
info_keys = {
    'functional': 'xc_functional'
}
for ikey in list(info_keys.keys()):
    info_keys[ikey] = PREFIX + info_keys[ikey]

BASE_OUTPUT = "../../C/alexandria/all/"
OUT_DIR = os.path.join(BASE_OUTPUT, "files")
INFO_DIR = os.path.join(BASE_OUTPUT, "info")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INFO_DIR, exist_ok=True)

providers = ['alexandria']
filter_str = 'elements HAS ONLY "C"'

# ========================
# GLOBAL DATA
# ========================
df = pd.DataFrame(columns=['id', 'formula', 'energy', 'provider'])
provider_counts = {p: 0 for p in providers}

# ========================
# HELPER FUNCTIONS
# ========================

def write_xyz(filename, lattice_vectors, positions, species, forces, energy, pbc=(True, True, True)):
    """Write structure data to an .xyz file in extxyz format."""
    natoms = len(species)
    lattice_str = " ".join([f"{v:.8f}" for v in np.array(lattice_vectors).flatten()])
    pbc_str = " ".join(['T' if b else 'F' for b in pbc])

    header = (
        f"{natoms}\n"
        f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy:.8f} pbc="{pbc_str}"\n'
    )

    lines = []
    for sp, pos, f in zip(species, positions, forces):
        pos_str = " ".join([f"{x:.8f}" for x in pos])
        f_str = " ".join([f"{x:.8f}" for x in f])
        lines.append(f"{sp} {pos_str} {f_str}")

    with open(filename, "w") as f:
        f.write(header + "\n".join(lines) + "\n")


def add_info_to_df(struct_id: str, info: dict):
    """Add or update an entry in the global dataframe. Dynamically adds new columns for new info keys."""
    global df

    # Add missing columns dynamically
    for k in info.keys():
        if k not in df.columns:
            df[k] = None

    entry = {'id': struct_id}
    entry.update(info)
    df.loc[len(df)] = entry


# ========================
# PARSER CALLBACK
# ========================

def parse_data(url, response):
    provider = response['meta']['provider']['prefix']
    data = response['data']

    # Count entries for this provider
    global provider_counts
    provider_counts.setdefault(provider, 0)
    provider_counts[provider] += len(data)

    for entry in data:
        struct_id = entry['id']
        attributes = entry['attributes']

        comp = Composition(attributes['chemical_formula_descriptive'])
        hill_comp = comp.hill_formula
        lattice_vectors = np.array(attributes['lattice_vectors'])
        positions = np.array(attributes['cartesian_site_positions'])
        species = attributes['species_at_sites']
        periodicity = np.array(attributes['dimension_types'])
        pbc = tuple(bool(x) for x in periodicity)

        forces = np.array(attributes.get(keys['forces'], np.zeros_like(positions)))
        energy = attributes.get(keys['energy'], np.nan)

        info = {
            'formula': hill_comp,
            'energy': energy,
            'provider': provider
        }

        for ik, k in info_keys.items():
            info[ik] = attributes.get(k, None)

        add_info_to_df(struct_id, info)

        filename = os.path.join(OUT_DIR, f"{provider}_{struct_id}.xyz")
        write_xyz(filename, lattice_vectors, positions, species, forces, energy, pbc)

    return {'next': response.get('links', {}).get('next')}


# ========================
# RUN CLIENT
# ========================
client = OptimadeClient(
    include_providers=providers,
    callbacks=[parse_data],
    max_results_per_provider=None,
)
client.get(filter=filter_str)

# ========================
# SAVE SUMMARY DATAFRAME
# ========================
summary_path = os.path.join(INFO_DIR, "summary.csv")
df.to_csv(summary_path, index=False)
print(df.head())

# ========================
# WRITE QUERY INFO
# ========================
query_file = os.path.join(INFO_DIR, "query.txt")
with open(query_file, "w") as f:
    f.write("Providers:\n")
    for provider in providers:
        f.write(f"  - {provider}\n")
    f.write(f"\nFilter:\n  {filter_str}\n")
    f.write("\nFetched entries per provider:\n")
    for provider, count in provider_counts.items():
        f.write(f"  {provider}: {count}\n")

