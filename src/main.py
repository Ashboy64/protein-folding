import numpy as np
from parmed.charmm import CharmmParameterSet

from src.molecule import Molecule


class StructurePredictor(object):
    """Predict protein structure"""

    def __init__(self, param_filepath):
        super(StructurePredictor, self).__init__()
        self.params = CharmmParameterSet(param_filepath)

    """
    Molecule m has properties:
    - m.bonds: list bonds and their lengths - [[('CE1', 'CE1'), 5], [...], ...]
    - m.angles: list of triples of atoms and angle wrt to central atom
    - m.dihedrals: list of quadruples of atoms and dihedral angle 
    - m.impropers: list of quadruples of atoms and angle for out of plane bending
    - m.ub: Urey-Bradley component for interaction between 1st and 3rd atoms in triplet. List of triples of atoms and 
            distance between them
    - m.nonbonded: list of pairs of atoms separated by at least 3 bonds
    """

    def eval_energy(self, m):
        total = 0

        for bond, length in m.bonds:
            b_type = self.params.bond_types[(bond[0], bond[1])]
            total += b_type.k * ((length - b_type.req) ** 2)

        for triplet, angle in m.angles:
            a_type = self.params.angle_types[(triplet[0], triplet[1], triplet[2])]
            total += a_type.k * ((angle - a_type.theteq) ** 2)

        for quad, d_angle in m.dihedrals:
            if quad in self.params.dihedral_types.keys():
                d_type = self.params.dihedral_types[(quad[0], quad[1], quad[2], quad[3])][0]
            else:
                d_type = self.params.dihedral_types[('X', quad[1], quad[2], 'X')][0]
            total += d_type.phi_k * (1 + np.cos(d_type.per * d_angle - d_type.phase))

        for quad, angle in m.impropers:
            if (quad[0], quad[1], quad[2], quad[3]) in self.params.improper_types.keys():
                i_type = self.params.improper_types[(quad[0], quad[1], quad[2], quad[3])]
            elif (quad[0], 'X', 'X', quad[3]) in self.params.improper_types.keys():
                i_type = self.params.improper_types[(quad[0], 'X', 'X', quad[3])]
            else:
                i_type = self.params.improper_types[(quad[3], 'X', 'X', quad[0])]
            total += i_type.psi_k * ((angle - i_type.psi_eq) ** 2)

        for triplet, dist in m.ub:
            ub_type = self.params.urey_bradley_types[(triplet[0], triplet[1], triplet[2])]
            total += ub_type.k * ((dist - ub_type.req) ** 2)

        for pair, dist in m.nonbonded:
            nb_type_0, nb_type_1 = self.params.atom_types_str[pair[0]], self.params.atom_types_str[pair[1]]

            eps_0, eps_1 = nb_type_0.epsilon, nb_type_1.epsilon
            eps = np.sqrt(eps_0 * eps_1)

            rmin_0, rmin_1 = nb_type_0.rmin, nb_type_1.rmin
            rmin = rmin_0 + rmin_1

            lennard_jones = eps * (((rmin / dist) ** 12) - 2 * ((rmin / dist) ** 6))  # CHECK IF THE 2 IS NEEDED
            total += lennard_jones

            q1, q2 = nb_type_0.charge, nb_type_1.charge
            total += (q1 * q2) / (eps * dist)

        return total


if __name__ == '__main__':
    sp = StructurePredictor(r'../params/par_all36m_prot.prm')
    m = Molecule(r'../pdb-files/5awl.psf', r'../params/par_all36m_prot.prm')

    print("Initialized molecule")
    print()

    print(sp.eval_energy(m))
