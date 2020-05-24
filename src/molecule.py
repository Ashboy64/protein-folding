import autograd.numpy as np
import parmed as pmd
from parmed.charmm import CharmmParameterSet


class Molecule(object):
    """Represents a molecule"""

    def __init__(self, psf_filepath, param_filepath, crds=None, positions=None):
        super(Molecule, self).__init__()
        self.psf = pmd.load_file(psf_filepath)
        self.params = CharmmParameterSet(param_filepath)
        self.crds = crds
        self.initialize(positions)

    def initialize(self, positions):
        not_in_params = []
        self.positions = {}

        counter = 0

        if self.crds is not None:
            for atom in self.crds.atoms:
                self.positions[atom] = [atom.xx, atom.xy, atom.xz]
        else:
            for atom in self.psf.atoms:
                if atom.type not in self.params.atom_types.keys():
                    not_in_params.append(atom.name)
                else:
                    if positions is not None:
                        self.positions[atom] = positions[counter]
                    else:
                        self.positions[atom] = [counter * np.random.uniform(0, 1), counter * np.random.uniform(0, 1),
                                                counter * np.random.uniform(0, 1)]
                    counter += 1

        if len(not_in_params) != 0:
            print("not_in_params: " + str(not_in_params))

        # Keep track of which pairs are nonbonded
        self.nonbonded_pairs = []

        for i in range(len(self.psf.atoms)):
            for j in range(len(self.psf.atoms)):
                if i != j:
                    self.nonbonded_pairs.append((self.psf.atoms[i], self.psf.atoms[j]))

        for angle in self.psf.angles:
            if (angle.atom1, angle.atom2) in self.nonbonded_pairs:
                self.nonbonded_pairs.remove((angle.atom1, angle.atom2))
            if (angle.atom2, angle.atom1) in self.nonbonded_pairs:
                self.nonbonded_pairs.remove((angle.atom2, angle.atom1))
            if (angle.atom2, angle.atom3) in self.nonbonded_pairs:
                self.nonbonded_pairs.remove((angle.atom2, angle.atom3))
            if (angle.atom3, angle.atom2) in self.nonbonded_pairs:
                self.nonbonded_pairs.remove((angle.atom3, angle.atom2))
            if (angle.atom1, angle.atom3) in self.nonbonded_pairs:
                self.nonbonded_pairs.remove((angle.atom1, angle.atom3))
            if (angle.atom3, angle.atom1) in self.nonbonded_pairs:
                self.nonbonded_pairs.remove((angle.atom3, angle.atom1))

    def random_unit_normal(self):
        arr = np.random.uniform(-1, 0, 3)
        return list(arr / np.linalg.norm(arr))

    def update_positions(self, new_pos):
        self.positions = {}

        counter = 0

        for atom in self.psf.atoms:
            self.positions[atom] = new_pos[counter]
            counter += 1

    @property
    def nonbonded(self):
        out = []
        for pair in self.nonbonded_pairs:
            out.append([(pair[0].type, pair[1].type), self.dist(pair[0], pair[1])])

        return out

    @property
    def ub(self):
        ub_vals = []

        for triplet in self.psf.angles:
            a_1, a_2, a_3 = triplet.atom1, triplet.atom2, triplet.atom3
            ub_vals.append([(a_1.type, a_2.type, a_3.type), self.dist(a_1, a_3)])

        return ub_vals

    @property
    def impropers(self):
        i_angles = []

        for quad in self.psf.impropers:  # a_1 bonded to a_2, a_1 is central atom
            a_1, a_2, a_3, a_4 = quad.atom1, quad.atom2, quad.atom3, quad.atom4
            v1 = np.array(self.positions[a_3]) - np.array(self.positions[a_1])
            v2 = np.array(self.positions[a_4]) - np.array(self.positions[a_1])
            n = np.cross(v1, v2)

            v3 = np.array(self.positions[a_2]) - np.array(self.positions[a_1])
            prod = np.dot(n, v3)

            if prod == 0:
                angle = np.pi / 2
            else:
                prod /= (np.linalg.norm(n) * np.linalg.norm(v3))
                prod = np.clip(prod, -1, 1)
                angle = np.pi / 2 - np.arccos(prod)

            i_angles.append([(a_1.type, a_2.type, a_3.type, a_4.type), np.degrees(angle)])

        return i_angles

    @property
    def dihedrals(self):
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

            d_angles.append([(a_1.type, a_2.type, a_3.type, a_4.type), np.degrees(angle)])

        return d_angles

    @property
    def angles(self):
        angles = []

        for triplet in self.psf.angles:
            c_atom = triplet.atom2
            a_1 = triplet.atom1
            a_2 = triplet.atom3

            v1 = np.array(self.positions[a_1]) - np.array(self.positions[c_atom])
            v2 = np.array(self.positions[a_2]) - np.array(self.positions[c_atom])

            prod = np.dot(v1, v2)

            if prod == 0:
                angle = np.pi/2
            else:
                prod /= (np.linalg.norm(v1) * np.linalg.norm(v2))
                prod = np.clip(prod, -1, 1)
                angle = np.arccos(prod)

            angles.append([(a_1.type, c_atom.type, a_2.type), np.degrees(angle)])

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
        return np.sqrt((a1_x - a2_x) ** 2 + (a1_y - a2_y) ** 2 + (a1_z - a2_z) ** 2)


if __name__ == '__main__':
    m = Molecule(r'../pdb-files/1l2y.psf', r'../params/par_all36m_prot.prm')
    print(m.bonds)
    print(m.angles)
    print(m.dihedrals)
    print(m.impropers)
    print(m.ub)
