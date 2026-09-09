import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('fullerene_potential.out')
distances = data[:,0]
energies = data[:,1]

min_idx = np.argmin(energies)
emin = energies[min_idx]
dmin = distances[min_idx]
be = energies[-1] - emin

plt.plot(distances, energies, label="potential energy")
plt.ylabel("potential energy [eV]")
plt.xlabel(r"center-center distance [$\AA$]")

plt.ylim(emin-0.5, emin+be+1)
plt.xlim(9, 12.5)

plt.legend()
plt.grid()

plt.savefig("penergy.pdf")
plt.show()

info = {
            'dmin': dmin,
            'emin':emin,
            'be':be
        }

import json
with open("info.out", "w") as f:
    json.dump(info, f, indent=4)
