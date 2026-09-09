import os
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.constants import e as elem_charge
import math
import json
import sys
import time
from pymatgen.core.composition import Composition

# output unbuffered to monitor progress in real time
os.environ["PYTHONUNBUFFERED"] = "1"

# counters
tot_frames = 0
n_structures = 0
n_local_descriptors = 0
force_type = {"total": 0, "free": 0, "U": 0}  # free  # total  # unknown


#
def frexp10(x):
    """Return (mantissa, exponent) such that x = mantissa * 10**exponent,
    with mantissa in [1, 10)."""
    if x == 0:
        return 0.0, 0
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10**exponent)
    return mantissa, exponent


BASE_URL = "https://nomad-lab.eu/prod/v1/api/v1"
MAX_RETRIES = 3
TIMEOUT = 60  # seconds

# OUTPUT DIRS
# CHANGE BASE_OUT_DIR
BASE_OUT_DIR = "../../C/nomad/DAC/"
os.makedirs(BASE_OUT_DIR, exist_ok=False)
OUT_DIR_FILES = os.path.join(BASE_OUT_DIR, "files")
os.makedirs(OUT_DIR_FILES, exist_ok=False)
OUT_DIR_INFO = os.path.join(BASE_OUT_DIR, "info")
os.makedirs(OUT_DIR_INFO, exist_ok=False)

from wget import download
from ase.io import Trajectory, write, read
import tempfile, shutil

READ_TRAJ = True
BASE_TRAJ_URL = "https://nomad-lab.eu/prod/v1/api/v1/uploads/"  # uploaid/"raw"/mainfile
TRAJ_FORMAT = "xml"  # ase traj format
OUT_TRAJ_DIR = os.path.join(BASE_OUT_DIR, "traj")
if READ_TRAJ:
    os.makedirs(OUT_TRAJ_DIR, exist_ok=False)

NO_FORCE_FILENAME = "NoForce.txt"
NO_FORCE_FILEPATH = os.path.join(OUT_DIR_INFO, NO_FORCE_FILENAME)
VALID_ENTRIES_FILENAME = "ValidEntries.txt"
VALID_ENTRIES_FILEPATH = os.path.join(OUT_DIR_INFO, VALID_ENTRIES_FILENAME)
with open(VALID_ENTRIES_FILEPATH, "w") as f:
    f.write(
        "ID\t\t\t\t\t\t\t\tMATERIAL_ID\t\t\tDIMENSIONALITY\tCHEMICAL_FORMULA_HILL\tWORKFLOW\tENSAMBLE\tFRAMES\n"
    )

IN_QUERY_BODY_FILEPATH = "./nomad_query.json"
with open(IN_QUERY_BODY_FILEPATH, "r") as f:
    query_body = json.load(f)

with open(os.path.join(OUT_DIR_INFO, "query.json"), "w") as f:
    json.dump(query_body, f, indent=4)


#  NOMAD API helpers
def fetch_entries(cursor=None):

    global query_body

    if cursor:
        query_body["pagination"]["page_after_value"] = cursor

    resp = requests.post(f"{BASE_URL}/entries/query", json=query_body)
    resp.raise_for_status()
    j = resp.json()
    return j.get("data", []), j.get("pagination", {}).get("next_page_after_value")


def fetch_archive(entry_id):
    """Fetch full archive with retries and timeout."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{BASE_URL}/entries/{entry_id}/archive/query",
                json={"required": "*"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("data", {}).get("archive")
        except Exception as e:
            print(f"[INFO] {entry_id}: attempt {attempt+1} failed: {e}")
            time.sleep(2**attempt)
    print(f"[WARNING] Failed to fetch archive for {entry_id}")
    return None


def write_frame_to_file(run_filename, xyz_content):
    """Write a single frame to file, appending to run file."""
    with open(run_filename, "a") as f:
        f.write(xyz_content + "\n")


#  Formatting helpers
def periodic_to_extxyz(periodic):
    return " ".join("T" if p == 1 else "F" for p in periodic)


def format_extxyz(positions, species, periodic, forces, energy, lattice_vectors):
    lines = [str(len(species))]
    lattice_str = " ".join(" ".join(f"{x:.8f}" for x in vec) for vec in lattice_vectors)
    lines.append(
        f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.11f} pbc="{periodic}"'
    )
    for s, p, f in zip(species, positions, forces):
        lines.append(
            f"{s} "
            + " ".join(f"{x:.8f}" for x in p)
            + " "
            + " ".join(f"{x:.8f}" for x in f)
        )
    return "\n".join(lines)


def to_angstrom(positions_obj):
    """Convert NOMAD positions object to Å consistently."""
    if isinstance(positions_obj, dict):
        positions = positions_obj["value"]
        unit = positions_obj.get("unit", "meter")
    else:
        positions = positions_obj
        unit = "meter"  # fallback: SI

    positions = np.array(positions)

    if unit == "meter":
        return positions * 1e10
    elif unit.lower().startswith("ang"):
        return positions
    else:
        raise ValueError(f"Unknown unit for positions: {unit}")


def append_NO_FORCE(entry_id):
    with open(NO_FORCE_FILEPATH, "a") as f:
        f.write(str(entry_id) + "\n")

    return


def append_VALID_ENTRY(
    entry_id,
    material_id,
    dimensionality,
    chemical_formula_hill,
    workflow,
    ensamble,
    frames,
):
    with open(VALID_ENTRIES_FILEPATH, "a") as f:
        # Convert frames list to a single string separated by spaces
        frames_str = " ".join(str(frame) for frame in frames)
        # Create the line to write, separating fields by commas (or choose your own delimiter)
        line = f"{entry_id},\t{material_id},\t{dimensionality},\t{chemical_formula_hill},\t{workflow},\t{ensamble},\t{frames_str}\n"
        f.write(line)

    return


#  parse + write directly
def parse_and_write(entry_id, archive):
    """
    Parse an archive entry and write frames to disk in extended XYZ format.
    Memory-conservative: streams frames directly to file without building
    large intermediate strings/lists.
    """
    # global counters to upate
    global tot_frames
    global n_local_descriptors
    global n_structures

    # flag to check that at least one frame is good
    good = False

    # try to collect metadata to keep
    results = archive.get("results", {})

    chemical_formula_hill = results.get("material", {}).get(
        "chemical_formula_hill", None
    )
    if chemical_formula_hill is None:
        chemical_formula_hill = (
            archive.get("metadata", {})
            .get("optimade", {})
            .get("chemical_formula_hill", None)
        )
    if chemical_formula_hill is None:
        chemical_formula_hill = (
            archive.get("run", [])[0]
            .get("system", [{}])[0]
            .get("chemical_composition_hill", None)
        )
    # DAC carbon
    # from pprint import pprint 
    # pprint(archive)
    try:
        if chemical_formula_hill is None:
            atom_parameters = (
                archive.get("run", [])[0].get("method", [])[0].get("atom_parameters", [])
            )
            comp_string = ""
            for ap in atom_parameters:
                comp_string += ap.get("label") + str(ap.get("atom_number"))

            try:
                comp_obj = Composition(comp_string)
                chemical_formula_hill = comp_obj.hill_formula

            # catch error if formula was not parsed correctely from data
            except ValueError as e:
                print(
                    f"[ERROR]: error {e} while parsing chemical formula hill, dataset may be organized differently than previous cases"
                )
                print(f"ID: {entry_id}")
                print(f"COMP_STRING: {comp_string}")  


    except IndexError or TypeError or KeyError as e:
        print(
            f"[ERROR]: error {e} while parsing chemical formula hill, dataset may be organized differently than previous cases"
        )
        print(f"ID: {entry_id}")
        print(f"COMP_STRING: {comp_string}")

    dimensionality = results.get("material", {}).get("dimensionality", None)
    material_id = results.get("material", {}).get("material_id", None)
    workflow = results.get("method", {}).get("workflow_name", None)

    ensamble = (
        results.get("properties", {})
        .get("thermodynamic", {})
        .get("provenance", {})
        .get("ensemble_type", None)
    )
    if ensamble is None:
        ensamble = (
            archive.get("workflow2", {})
            .get("method", {})
            .get("thermodynamic_ensemble", None)
        )

    # read trajectory file directely
    if READ_TRAJ:
        mainfile = archive.get("metadata", {}).get("mainfile", None)
        upload_id = archive.get("metadata", {}).get("upload_id", None)
        if mainfile and upload_id:
            mainfile_url = f"{BASE_TRAJ_URL}{upload_id}/raw/{mainfile}"
            if not mainfile_url.endswith(TRAJ_FORMAT):
                print(f"[WARNING]: mainfile {mainfile_url} does not end with {TRAJ_FORMAT} extension, skip")
                return 0
            traj_filepath = os.path.join(OUT_TRAJ_DIR, f"{entry_id}.{TRAJ_FORMAT}")
            try:
                traj_filepath = download(mainfile_url, out=traj_filepath)
                if TRAJ_FORMAT == "traj" or TRAJ_FORMAT == "xml":
                    if TRAJ_FORMAT == "xml":
                        format = "vasp-xml"
                    elif TRAJ_FORMAT == "traj":
                        format = "traj"
                    else:
                        print("[WARNING]: unrecognised format")
                    traj = read(traj_filepath, index = ":", format=format)
                    n_frames = len(traj)

                    if n_frames == 0:
                        print(
                            f"[WARNING] {entry_id} with url {mainfile_url}: trajectory has zero frames, skipping"
                        )
                        return 0

                    # check if at least one frame of one run is good for the structure
                    good = True
                    n_local_descriptors += sum(len(atoms) for atoms in traj)
                    tot_frames += n_frames

                    write(
                        os.path.join(
                            OUT_DIR_FILES,
                            (
                                f"{entry_id}_{ensamble}_{chemical_formula_hill}.xyz"
                                if ensamble
                                else f"{entry_id}_{chemical_formula_hill}.xyz"
                            ),
                        ),
                        traj,
                        format="extxyz",
                    )

                    frames = [n_frames]  # single run with all frames
                else:
                    print(f"[ERROR] Unsupported TRAJ_FORMAT {TRAJ_FORMAT}, exit")
                    exit(1)
            except Exception as e:
                print(
                    f"[WARNING] {entry_id} with url {mainfile_url}: failed to read trajectory: {e}"
                )
                return 0

    # extract trajectory from processed data
    if not READ_TRAJ:
        frames = []

        for run_i, run in enumerate(archive.get("run", [])):
            systems = run.get("system", [])
            run_filename = None
            run_frames = 0

            # Determine force type once per run
            try:
                forces_keys = run.get("calculation", [])[0].get("forces", {}).keys()
            except IndexError:
                print(
                    f"[WARNING] run {run_i} of entry {entry_id} had no calculations, skipping"
                )
                with open(NO_FORCE_FILEPATH, "a") as f:
                    f.write(f"{entry_id}\n")
                continue

            frc_type = (
                "total"
                if "total" in forces_keys
                else "free" if "free" in forces_keys else None
            )
            if frc_type is None:
                frc_type = "U"
                force_type[frc_type] += 1
                with open(NO_FORCE_FILEPATH, "a") as f:
                    f.write(f"{entry_id}\n")
                break
            force_type[frc_type] += 1

            # Process each calculation in this run
            for calc_i, calc in enumerate(run.get("calculation", [])):

                try:
                    # --- System info ---
                    positions = species = lattice = periodic = None
                    system_ref = calc.get("system_ref")

                    if systems:
                        if system_ref:
                            try:
                                idx = int(system_ref.split("/")[-1])
                                atoms = systems[idx].get("atoms", {})
                                positions = atoms.get("positions")
                                species = atoms.get("labels")
                                lattice = atoms.get("lattice_vectors", np.eye(3))
                                periodic = atoms.get("periodic", [0, 0, 0])
                            except Exception:
                                print(
                                    f"[WARNING] {entry_id} run {run_i} calc {calc_i}: invalid system_ref, skipping"
                                )
                                continue
                        else:
                            # use calc_i as fallback system index
                            try:
                                atoms = systems[calc_i].get("atoms", {})
                                positions = atoms.get("positions")
                                species = atoms.get("labels")
                                lattice = atoms.get("lattice_vectors", np.eye(3))
                                periodic = atoms.get("periodic", [0, 0, 0])

                            except Exception:
                                print(
                                    f"[WARNING] {entry_id} run {run_i} calc {calc_i}: no system_ref and invalid system index, skipping"
                                )
                                continue
                    else:
                        try:
                            structure = archive["results"]["properties"]["structures"][
                                "structure_original"
                            ]
                        except KeyError:
                            structure = archive.get("metadata", {}).get("optimade", {})

                        positions = structure.get("cartesian_site_positions")
                        species = structure.get("species_at_sites")
                        lattice = structure.get("lattice_vectors")
                        periodic = structure.get("dimension_types", [0, 0, 0])
                    if (
                        positions is None
                        or species is None
                        or lattice is None
                        or periodic is None
                    ):
                        print(
                            f"[WARNING] {entry_id} run {run_i} calc {calc_i}: missing system info, skipping"
                        )
                        continue

                    # --- Unit conversion ---
                    try:
                        pos = to_angstrom(positions)
                        lv = to_angstrom(lattice)

                        # pos = positions
                        # lv = lattice

                    except Exception as e:
                        print(
                            f"[WARNING] {entry_id} run {run_i} calc {calc_i}: unit conversion failed: {e}"
                        )
                        continue

                    # --- Forces & energy ---
                    forces = (
                        calc.get("forces", {}).get(frc_type.lower(), {}).get("value")
                    )
                    if forces is None:
                        print(
                            f"[WARNING] {entry_id} run {run_i} calc {calc_i}: no forces, skipping"
                        )
                        continue

                    try:
                        energy = calc["energy"]["total"]["value"]
                    except Exception:
                        print(
                            f"[WARNING] {entry_id} run {run_i} calc {calc_i}: no energy, skipping"
                        )
                        continue

                    frc = (
                        np.asarray(forces, dtype="float64") / (elem_charge / 1e-10)
                    ).tolist()
                    energy_ev = float(energy) / elem_charge

                    # --- Open run file once ---
                    if run_filename is None:
                        run_filename = os.path.join(
                            OUT_DIR_FILES,
                            (
                                f"{entry_id}_{ensamble}_{chemical_formula_hill}.xyz"
                                if ensamble
                                else f"{entry_id}_{chemical_formula_hill}.xyz"
                            ),
                        )

                    # check if at least one frame of one run is good for the structure
                    if good == False:
                        good = True

                    # increase frames if frame is good
                    run_frames += 1
                    with open(run_filename, "a") as f_out:
                        # Header
                        f_out.write(f"{len(species)}\n")
                        lv_str = " ".join(
                            " ".join(f"{x:.8f}" for x in vec) for vec in lv
                        )
                        periodic_str = " ".join(
                            "T" if p == 1 else "F" for p in periodic
                        )
                        f_out.write(
                            f'Lattice="{lv_str}" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy_ev:.11f} pbc="{periodic_str}"\n'
                        )
                        # Atom lines
                        for s, p, f in zip(species, pos, frc):
                            f_out.write(
                                f"{s} "
                                + " ".join(f"{x:.8f}" for x in p)
                                + " "
                                + " ".join(f"{x:.8f}" for x in f)
                                + "\n"
                            )

                    # --- Stats update ---
                    n_local_descriptors += len(species)
                    tot_frames += 1

                except Exception as e:
                    print(
                        f"[INFO] {entry_id} run {run_i} calc {calc_i} skipped due to error: {e}"
                    )
                    continue
            frames.append(run_frames)

    if good:
        n_structures += 1
        append_VALID_ENTRY(
            entry_id,
            material_id,
            dimensionality,
            chemical_formula_hill,
            workflow,
            ensamble,
            frames,
        )

    return 0


#  Main loop with concurrency
def main():
    cursor = None
    page = 0

    while True:
        print(f"Fetching page {page}...")
        entries, cursor = fetch_entries(cursor)
        if not entries:
            break

        # Fetch archives in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {
                executor.submit(fetch_archive, e["entry_id"]): e["entry_id"]
                for e in entries
            }
            for future in as_completed(future_map):
                entry_id = future_map[future]
                archive = future.result()
                if archive:
                    parsed = parse_and_write(entry_id, archive)
        # for e in entries:
        #     entry_id = e["entry_id"]
        #     archive = fetch_archive(entry_id)
        #     if archive:
        #         parsed = parse_and_write(entry_id, archive)

        # Save progress once per page
        if cursor:
            with open("last_cursor.txt", "w") as f:
                f.write(cursor)
        with open(os.path.join(OUT_DIR_INFO, "counts.txt"), "w") as f:
            f.write(f"n structures: {n_structures}\n")
            f.write(f"local descriptors: {n_local_descriptors}\n")
            f.write(f"frames: {tot_frames}\n")
            f.write("forces types:\n")
            json.dump(force_type, f, indent=4)
            f.write("\n")

        if cursor is None:
            break
        page += 1

    print(f"query completed, data stored in {OUT_DIR_FILES}")

    return 0


if __name__ == "__main__":
    main()
