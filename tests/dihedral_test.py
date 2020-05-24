import numpy as np
from src.molecule import Molecule
from tests.common import *

if __name__ == '__main__':
    m = Molecule(r'../pdb-files/ala5.psf', r'../params/par_all36m_prot.prm')

    for quad in m.psf.dihedrals:
        a_1, a_2, a_3, a_4 = quad.atom1, quad.atom2, quad.atom3, quad.atom4
        v1 = np.array(m.positions[a_2]) - np.array(m.positions[a_1])
        v2 = np.array(m.positions[a_3]) - np.array(m.positions[a_1])
        n1 = np.cross(v1, v2)

        v3 = np.array(m.positions[a_4]) - np.array(m.positions[a_3])
        v4 = -v2
        n2 = np.cross(v3, v4)

        prod = np.dot(n1, n2)

        if prod == 0:
            angle = np.degrees(0)
        else:
            prod /= (np.linalg.norm(n1) * np.linalg.norm(n2))
            prod = np.clip(prod, -1, 1)
            angle = np.degrees(np.arccos(prod))

        true_angle = GetTijkl(np.array(m.positions[a_1]),
                              np.array(m.positions[a_2]),
                              np.array(m.positions[a_3]),
                              np.array(m.positions[a_4]))

        print("Computed: " + str(angle))
        print("True: " + str(true_angle))

