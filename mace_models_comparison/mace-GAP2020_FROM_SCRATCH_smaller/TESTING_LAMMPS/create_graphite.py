from ase.lattice.hexagonal import *
import ase.io as io
from ase import Atoms, Atom
import numpy as np

index1=5
index2=5
mya = 2.464
myc = 6.711 #experimental value equal to 2 times the interlayer spacing

stacks = 1 

gra = Graphite(symbol = 'C',latticeconstant={'a':mya,'c':myc},
               size=(index1,index2,stacks))
cell = np.array(gra.cell)
print("original cell:")
print(cell)
void = 100
cell[2] = np.array([0, 0, void])
print("cell with void:")
print(cell)
gra.cell = cell

io.write('2_layer_graphite.lammps-data', gra, format='lammps-data')