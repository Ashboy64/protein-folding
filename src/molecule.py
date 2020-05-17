import numpy as np
import parmed as pmd
from parmed.charmm import CharmmParameterSet


class Molecule(object):
    """Represents a molecule"""

    def __init__(self, psf_filepath, param_filepath):
        super(Molecule, self).__init__()
        self.psf = pmd.load_file(psf_filepath)
        self.params = CharmmParameterSet(param_filepath)
        self.initialize_positions()

    def initialize_positions(self):
        not_in_params = []
        self.positions = {}

        counter = 0

        for atom in self.psf.atoms:
            if atom.type not in self.params.atom_types.keys():
                not_in_params.append(atom.name)
            else:
                self.positions[atom] = [2*np.random.uniform(), 2*np.random.uniform(), 2*np.random.uniform()]
                counter += 1

        if len(not_in_params) != 0:
            print("not_in_params: " + str(not_in_params))

    @property
    def dihedrals(self):
        quads = self.get_quadruples()
        print(len(quads))
        d_angles = []
        for quad in self.psf.dihedrals:
            a_1, a_2, a_3, a_4 = quad.atom1, quad.atom2, quad.atom3, quad.atom4
            v1 = np.array(self.positions[a_2]) - np.array(self.positions[a_1])
            v2 = np.array(self.positions[a_3]) - np.array(self.positions[a_1])
            n1 = np.cross(v1, v2)

            v3 = np.array(self.positions[a_4]) - np.array(self.positions[a_3])
            v4 = -v2
            n2 = np.cross(v3, v4)

            prod = np.dot(n1, n2)

            if prod == 0:
                angle = 0
            else:
                prod /= (np.linalg.norm(n1) * np.linalg.norm(n2))
                prod = np.clip(prod, -1, 1)
                angle = np.arccos(prod)

            d_angles.append([(a_1.type, a_2.type, a_3.type), angle])

        return d_angles

    @property
    def angles(self):
        angles = []
        for c_atom in self.psf.atoms:
            for i in range(len(c_atom.bonds)):
                for j in range(i + 1, len(c_atom.bonds)):
                    a_1 = self.get_other_atom(c_atom.bonds[i], c_atom)
                    a_2 = self.get_other_atom(c_atom.bonds[j], c_atom)

                    v1 = np.array(self.positions[a_1]) - np.array(self.positions[c_atom])
                    v2 = np.array(self.positions[a_2]) - np.array(self.positions[c_atom])

                    prod = np.dot(v1, v2)
                    prod /= (np.linalg.norm(v1) * np.linalg.norm(v2))

                    prod = np.clip(prod, -1, 1)

                    angle = np.arccos(prod)
                    angles.append([(a_1.type, c_atom.type, a_2.type), angle])
        return angles

    @property
    def bonds(self):
        all = []
        for bond in self.psf.bonds:
            a1 = bond.atom1
            a2 = bond.atom2
            all.append([(a1.type, a2.type), self.dist(a1, a2)])
        return all

    def get_other_atom(self, bond, atom):
        if bond.atom1 == atom:
            return bond.atom2
        return bond.atom1

    def dist(self, a1, a2):
        a1_x, a1_y, a1_z = self.positions[a1]
        a2_x, a2_y, a2_z = self.positions[a2]
        return np.sqrt((a1_x - a2_x)**2 + (a1_y - a2_y)**2 + (a1_z - a2_z)**2)


if __name__ == '__main__':
    m = Molecule(r'../pdb-files/step1_pdbreader.psf', r'../params/par_all36m_prot.prm')
    print(m.bonds)
    print(m.angles)
    print(m.dihedrals)
