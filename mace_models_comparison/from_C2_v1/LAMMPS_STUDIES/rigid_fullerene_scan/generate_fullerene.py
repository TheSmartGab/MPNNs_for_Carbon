from ase.build import molecule
from ase.io import write
import numpy as np

atoms = molecule('C60')
# cell = np.array([
#     [100,0,0],
#     [0,100,0],
#     [0, 0, 100]
# ])
# atoms.cell = cell
print(atoms.get_cell())
atoms.pbc = [True, True, True]
write('fullerene.lammps-data', atoms)
