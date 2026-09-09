from argparse import ArgumentParser
import os
import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read

from nequip.ase import NequIPCalculator

# split names, change if your directories/files are named differently
SPLIT_KEYS = ["training", "validation", "test"]


def gather_ref(args):
    """Plot reference training / validation / test diatomic energies."""
    print("[INFO] Gathering reference data")

    REF_SPLIT_DATA = {k: [] for k in SPLIT_KEYS}

    FILES = {
        k: os.path.join(args.ref_split, k, f"{k}.{args.ref_extension}")
        for k in SPLIT_KEYS
    }

    for k in SPLIT_KEYS:
        penergy_force_file = os.path.join(args.ref_split, k, "penergy_force.dat")

        if not (os.path.isfile(penergy_force_file) and os.path.isfile(penergy_force_file)):
            images = read(FILES[k], index=":", format=args.ref_format)

            for idx, atoms in enumerate(images):
                if len(atoms) != 2:
                    print(
                        f"[ERROR] reference split {k} at index {idx} "
                        "does not contain 2 atoms. exit."
                    )
                    exit(-1)

                p0, p1 = atoms.get_positions()
                d = np.linalg.norm(p0 - p1)
                e = atoms.get_potential_energy()
                f = atoms.get_forces()[0][0] # assumes force is only directed aling x axis
                REF_SPLIT_DATA[k].append([d, e, f])

            REF_SPLIT_DATA[k] = np.array(REF_SPLIT_DATA[k])
            order = np.argsort(REF_SPLIT_DATA[k][:, 0])
            REF_SPLIT_DATA[k] = REF_SPLIT_DATA[k][order]

            np.savetxt(penergy_force_file, REF_SPLIT_DATA[k])
        else:
            REF_SPLIT_DATA[k] = np.loadtxt(penergy_force_file)

        # return rather than plotting
        # plt.scatter(
        #     REF_SPLIT_DATA[k][:, 0],
        #     REF_SPLIT_DATA[k][:, 1],
        #     s=2,
        #     label=k,
        # )
    return REF_SPLIT_DATA


def main():
    parser = ArgumentParser(
        "Compute and plot the 2-atom potential energy of a NequIP model."
    )

    # model
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to deployed NequIP model (e.g. deployed.pth)",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs=2,
        required=True,
        help="chemical symbols of the diatomic molecule",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to use: cpu or cuda",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory where results will be stored",
    )

    # distance scan
    parser.add_argument("--min_d", type=float, default=0.5)
    parser.add_argument("--max_d", type=float, default=5.0)
    parser.add_argument("--num", type=int, default=100)

    # reference data
    parser.add_argument("--ref_split", type=str, default=None)
    parser.add_argument("--ref_extension", type=str, default="extxyz")
    parser.add_argument("--ref_format", type=str, default="extxyz")

    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print("[ERROR] NequIP model not found. exit.")
        exit(-1)

    if not os.path.isdir(args.output_dir):
        print("[INFO] Creating directory", args.output_dir)
        os.makedirs(args.output_dir)

    if args.ref_split:
        REF_SPLIT_DATA = gather_ref(args)

    # distance grid
    ds = np.linspace(args.min_d, args.max_d, args.num)
    energies = []
    forces = []

    # NequIP calculator
    calculator = NequIPCalculator.from_compiled_model(
        compile_path=args.model_path,
        device=args.device,
        chemical_species_to_atom_type_map=True,
    )

    atoms = Atoms(args.elements)
    atoms.calc = calculator

    for d in ds:
        atoms.positions = [[0.0, 0.0, 0.0], [d, 0.0, 0.0]]
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces()[0][0]) # assumes force is only directed along x axis

    energies = np.array(energies)
    forces = np.array(forces)

    min_idx = np.argmin(energies)
    min_e = energies[min_idx]
    min_d = ds[min_idx]
    last_e = energies[-1]

    # save model curve
    plot_points_path = os.path.join(args.output_dir, "model_penergy_forces.out")
    np.savetxt(plot_points_path, np.column_stack([ds, energies, forces]))

    ##################################################################
    # plotting

    # energy
    plt.plot(ds, energies, label="NequIP potential energy")
    plt.scatter(
        min_d,
        min_e,
        marker="x",
        color="orange",
        label=f"model min: ({min_d:.2f}, {min_e:.2f})",
    )

    if args.ref_split:
        for k in SPLIT_KEYS:
            plt.scatter(
                REF_SPLIT_DATA[k][:, 0],
                REF_SPLIT_DATA[k][:, 1],
                s=2,
                label=k,
            )

    plt.xlabel(r"distance [$\AA$]")
    plt.ylabel(r"potential energy [eV]")

    plt.ylim((min_e - 2.0, min_e + 40.0))
    plt.grid()
    plt.legend()
    plt.tight_layout()

    out_plot = os.path.join(args.output_dir, "potential_energy.pdf")
    print("[INFO] saving plot to", out_plot)
    plt.savefig(out_plot)
    plt.close()

    # force 
    plt.plot(ds, forces, label="NequIP force")

    if args.ref_split:
        for k in SPLIT_KEYS:
            plt.scatter(
                REF_SPLIT_DATA[k][:, 0],
                REF_SPLIT_DATA[k][:, 2],
                s=2,
                label=k,
            )

    ymin = np.clip(np.min(forces) - 2., -100., None)
    ymax = np.clip(np.max(forces) + 2., None, 100.)

    plt.ylim((ymin, ymax))    
    plt.xlabel(r"distance [$\AA$]")
    plt.ylabel(r"force [$eV/\AA$]")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    out_plot = os.path.join(args.output_dir, "force.pdf")
    print("[INFO] saving plot to", out_plot)
    plt.savefig(out_plot)
    plt.close()

    #####################################################
    # some info
    # binding energy
    be = last_e - min_e

    info_path = os.path.join(args.output_dir, "info.out")
    print("[INFO] Writing info to", info_path)
    with open(info_path, "w") as f:
        f.write(f"binding energy:\t{be}\n")
        f.write(f"distance with minimum energy:\t{min_d}\n")

    return 0


if __name__ == "__main__":
    main()
