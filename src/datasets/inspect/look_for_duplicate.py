import os, sys
import argparse

from ase.io import read
from ase import Atoms

verbose = bool(False)
verboseprint = print if verbose else lambda *a, **k: None

def retrieve_frame(file_path, frame):
    try:
        atoms = read(file_path, index=frame)
        return atoms
    except Exception as e:
        print(f"Error reading frame {frame} from {file_path}: {e}")
        return None
    
def locate_0momenta_frame(traj):
    """geometry optimization start with 0 momenta configurations. locate last frame with zero momenta"""
    for i, atoms in enumerate(traj):
        if (atoms.get_momenta() == 0).all():
            return_index = i
        else:
            return return_index
        
    verboseprint("No frame with zero momenta found")
    return None, None

def check_file_frame(file_path, target_frame):
    traj = read(file_path, index=':')
    
    for frame in traj:
        if compare_frames(frame, target_frame):
            return True
    return False

    
def compare_frames(atoms1 : Atoms, atoms2 : Atoms):
    if len(atoms1) != len(atoms2):
        return False
    # if not (atoms1.symbols == atoms2.symbols).all():
    #     verboseprint("Different symbols")
    #     return False

    # if not (atoms1.cell == atoms2.cell).all():
    #     verboseprint("Different cell")
    #     return False
    
    # if not (atoms1.pbc == atoms2.pbc).all():
    #     verboseprint("Different pbc")
    #     return False
    
    if not (atoms1.get_positions() == atoms2.get_positions()).all():
        verboseprint("Different positions")
        return False
    
    # if not (atoms1.get_momenta() == atoms2.get_momenta()).all():
    #     verboseprint("Different momenta")
    #     return False
    
    # if not (atoms1.get_forces() == atoms2.get_forces()).all():
    #     verboseprint("Different forces")
    #     return False
    
    # if not (atoms1.get_stress() == atoms2.get_stress()).all():
    #     verboseprint("Different stress")
    #     return False
    
    # if not atoms1.info == atoms2.info:
    #     verboseprint("Different info")
    #     return False    
    
    # if not atoms1.get_total_energy() == atoms2.get_total_energy():
    #     verboseprint("Different total energy")
    #     return False    
    # if not atoms1.get_kinetic_energy() == atoms2.get_kinetic_energy():
    #     verboseprint("Different kinetic energy")
    #     return False  
    # if not atoms1.get_potential_energy() == atoms2.get_potential_energy():
    #     verboseprint("Different potential energy")
    #     return False      

    return True

def main():

    parser = argparse.ArgumentParser("look for a duplicate frame. GRAPHENE_MLIP has suspicious relaxation simulation, which I suspected containing the molacular dynamics as well. In this script I look for duplicate frames.")
    parser.add_argument("file_path", type=str, help="path to the file")
    parser.add_argument("frame", type=int, help="frame to look for")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity", default=False)

    global verbose

    verbose = parser.parse_args().verbose
    verboseprint = print if verbose else lambda *a, **k: None

    file_path = parser.parse_args().file_path
    frame = parser.parse_args().frame


    if not os.path.exists(file_path):
        print(f"{file_path} does not exist")
        return 1
    if not os.path.isfile(file_path):
        print(f"{file_path} is not a file")
        return 1
    
    dir = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    name, ext = os.path.splitext(base)
    composition = name.split("_")[-1]


    target_traj = read(file_path, index=':')
    initial_frame = locate_0momenta_frame(target_traj)
    if initial_frame is not None:
        target_frame = initial_frame + frame +1
    else:
        target_frame = frame

    print("looking for target frame:", target_frame)
    print("comparing to frame inddexes:", frame)

    atoms_to_find = target_traj[target_frame]
    if atoms_to_find is None:
        print(f"Could not retrieve frame {frame} from {file_path}")
        return 1
    
    for file in os.listdir(dir):
        base_file = os.path.basename(file)
        name_file, ext_file = os.path.splitext(base_file)
        composition_file = name_file.split("_")[-1]
        if composition_file != composition:
            continue

        if not file.endswith(ext):
            continue
        if file == base:
            continue
        
        file_path_to_check = os.path.join(dir, file)
        match = check_file_frame(file_path_to_check, atoms_to_find)
        if not match:
            print(f"No match in {file_path_to_check}")
        else:
            print(f"Found duplicate frame in {file_path_to_check}")
            return 0
        # try:
        #     frame_to_check = read(file_path_to_check, index = str(frame))
        # except Exception as e:
        #     print(f"Could not read frame {frame} from {file_path_to_check}: {e}")
        #     continue
       
        # if compare_frames(atoms_to_find, frame_to_check):
        #     print(f"Found duplicate frame in {file_path_to_check}")
        #     return 0
        
        # else:
        #     print(f"No match in {file_path_to_check}")

    
    return 0


if __name__ == "__main__":
    main()