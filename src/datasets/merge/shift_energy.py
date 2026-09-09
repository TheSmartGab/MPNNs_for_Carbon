import argparse
import json
import os
from ase.io import read, write

def shift_energies(input_path, isolated_energies):
    """
    Subtracts the sum of isolated atomic energies from the total energy of 
    each configuration in a file.
    """
    # Load the configurations
    try:
        configs = read(input_path, index=':')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    shifted_configs = []

    for atoms in configs:
        # Get total potential energy
        try:
            total_energy = atoms.get_potential_energy()
        except RuntimeError:
            print("Skipping a configuration: No potential energy found.")
            continue

        # Calculate the reference energy for the specific composition
        # Reference = sum(count_i * isolated_energy_i)
        composition = atoms.get_atomic_numbers()
        ref_energy = sum(isolated_energies.get(str(z)) for z in composition)

        # Shift the energy
        new_energy = total_energy - ref_energy
        
        # Create a copy and update the results dictionary
        new_atoms = atoms.copy()
        new_atoms.calc = atoms.calc
        new_atoms.calc.results['energy'] = new_energy
        pe = new_atoms.get_potential_energy()
        fs = new_atoms.get_forces()

        try:
            stress = new_atoms.get_stress()
        except Exception as e:
            print("[WARNING] encountered exception", e, "stress for this configuration is not available.")

        shifted_configs.append(new_atoms)

    # Determine output path
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_shifted{ext}"

    # Write output
    write(output_path, shifted_configs)
    print(f"Successfully wrote {len(shifted_configs)} configurations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Shift ASE configuration energies by isolated atom values.")
    
    parser.add_argument("filepath", type=str, help="Path to the input geometry file.")
    parser.add_argument("energies", type=str, 
                        help="JSON string of isolated energies, e.g., '{\"6\": -1.2, \"1\": -0.5}'")

    args = parser.parse_args()

    # Parse the energy dictionary
    try:
        iso_en_dict = json.loads(args.energies)
        shift_energies(args.filepath, iso_en_dict)
    except json.JSONDecodeError:
        print("Error: The energy dictionary must be a valid JSON string.")


    return 0

if __name__ == "__main__":
    main()