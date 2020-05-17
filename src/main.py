import numpy as np
import parmed as pmd
from parmed.charmm import CharmmParameterSet


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

        for triplet, angle in m.bonds:
            a_type = self.params.angle_types[(triplet[0], triplet[1], triplet[2])]
            total += a_type.k * ((angle - a_type.thetaeq) ** 2)

        for quad, d_angle in m.dihedrals:
            d_type = self.params.dihedral_types[(quad[0], quad[1], quad[2], quad[3])]
            total += d_type.phi_k * (1 + np.cos(d_type.per * d_angle - d_type.phase))

        for quad, angle in m.impropers:
            i_type = self.params.improper_types[(quad[0], quad[1], quad[2], quad[3])]
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
    sp = StructurePredictor(r'../params/par_all22_prot.prm')
    pdb = pmd.load_file(r'../pdb-files/1l2y.pdb')
