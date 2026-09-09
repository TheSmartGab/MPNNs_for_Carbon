from argparse import ArgumentParser
import os
from mace.calculators import MACECalculator
from ase import Atoms
from ase.io import read, write
from ase.build import bulk
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(
        "This scripts computes and plot the 2-atoms potential energy of a mace model. Requires mace (and ase, but it is installed with mace)"
    )

    # model
    parser.add_argument("--model_path", type=str, help="path/to/model.model")
    parser.add_argument(
        "--name",
        type=str,
        help="name argument for ase.build.bulk",
    )
    parser.add_argument(
        "--crystalstructure", type=str, help="crystalstructure for ase.build.bulk"
    )

    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="device to use. use gpu for nvidia/AMD gpu, and mps for aplle silicon",
    )
    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory where to store results. if it does not exists, it will be created.",
    )

    # some computing and plotting parameters
    parser.add_argument(
        "--min_l", required=False, default=1, type=float, help="minimum lattice parameter"
    )
    parser.add_argument(
        "--max_l", required=False, default=10, type=float, help="maximum lattice parameter"
    )
    parser.add_argument(
        "--num",
        required=False,
        default=1000,
        type=int,
        help="number of points to evaluate",
    )

    args = parser.parse_args()

    # some sanity checking
    if not args.model_path.endswith(".model"):
        print("[ERROR] model should be in .model format. exit")
        exit(-1)

    if not os.path.isdir(args.output_dir):
        print("[INFO] Creating directory", args.output_dir)
        os.makedirs(args.output_dir)

    ls = np.linspace(args.min_l, args.max_l, args.num)
    energies = []
    calculator = MACECalculator(model_path=args.model_path, device=args.device)

    images=[]

    for l in ls:
        atoms = bulk(name=args.name, crystalstructure=args.crystalstructure, a=l)
        atoms.calc=calculator
        e=atoms.get_potential_energy()
        energies.append(e)
        images.append(atoms)
        print(l, e, "\n")

    write(os.path.join(args.output_dir, "configs.extxyz"), images)
    min_idx = np.argmin(energies)
    min_e = energies[min_idx]
    min_l = ls[min_idx]
    last_e = energies[
        -1
    ]  # assuming the energy actually converges to some constant value at last

    # save plot points
    plot_points_path = os.path.join(args.output_dir, "model_penergy.out")
    with open(plot_points_path, "w") as f:
        for d, e in zip(ls, energies):
            f.write(f"{d} {e}\n")

    plt.plot(ls, energies, label="model potential energy")
    plt.scatter(
        min_l, min_e, marker="x", label="minimum of potential energy", color="orange"
    )
    plt.xlabel(r"distance [$\AA$]", fontsize = 14)
    plt.ylabel(r"potential energy [eV]", fontsize = 14)


    # ylim for improved readability
    plt.ylim((min_e - 2.0, min_e + 40.0))  # can adjust this at will

    plt.grid()
    plt.legend()
    plt.tight_layout()
    print(
        "[INFO] saving plot to", os.path.join(args.output_dir, "potential_energy.svg")
    )
    plt.savefig(os.path.join(args.output_dir, "potential_energy.svg"))

    be = last_e - min_e

    print("[INFO] Writing info to", os.path.join(args.output_dir, "info.out"))
    with open(os.path.join(args.output_dir, "info.out"), "w") as f:
        f.write(f"binding energy:\t{be}\n")
        f.write(f"lattice parameter with minimum energy:\t{min_l}\n")

    return 0


if __name__ == "__main__":
    main()
