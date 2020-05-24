import numpy as np
from src.molecule import Molecule
from tests.common import GetAijk

if __name__ == '__main__':
    m = Molecule(r'../pdb-files/ala5.psf', r'../params/par_all36m_prot.prm')

    for triplet in m.psf.angles:
        c_atom = triplet.atom2
        a_1 = triplet.atom1
        a_2 = triplet.atom3

        v1 = np.array(m.positions[a_1]) - np.array(m.positions[c_atom])
        v2 = np.array(m.positions[a_2]) - np.array(m.positions[c_atom])

        prod = np.dot(v1, v2)

        if prod == 0:
            this_angle = np.degrees(np.pi / 2)
        else:
            prod /= (np.linalg.norm(v1) * np.linalg.norm(v2))
            prod = np.clip(prod, -1, 1)
            this_angle = np.degrees(np.arccos(prod))

        # Now compute true angles
        real_angle = GetAijk(np.array(m.positions[a_1]), np.array(m.positions[c_atom]), np.array(m.positions[a_2]))

        print("Computed: " + str(this_angle))
        print("Real: " + str(real_angle))

        print()
