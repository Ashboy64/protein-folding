import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from parmed.charmm import CharmmParameterSet
from autograd import grad
from molecule import Molecule


class StructurePredictor(object):
    """Predict protein structure"""

    def __init__(self, molecule, param_filepath):
        super(StructurePredictor, self).__init__()
        self.params = CharmmParameterSet(param_filepath)
        self.m = molecule

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

    def eval_energy(self, positions):
        total = 0
        self.m.update_positions(positions)
        m = self.m

        bond_e = 0
        for bond, length in m.bonds:
            b_type = self.params.bond_types[(bond[0], bond[1])]
            bond_e += b_type.k * ((length - b_type.req) ** 2)
        # print("Bond Energy: " + str(bond_e))
        total += bond_e

        angle_e = 0
        for triplet, angle in m.angles:
            a_type = self.params.angle_types[(triplet[0], triplet[1], triplet[2])]
            angle_e += a_type.k * ((angle - a_type.theteq) ** 2)
        # print("Angle Energy: " + str(angle_e))
        total += angle_e

        dihedral_e = 0
        for quad, d_angle in m.dihedrals:
            if quad in self.params.dihedral_types.keys():
                d_type = self.params.dihedral_types[(quad[0], quad[1], quad[2], quad[3])][0]
            else:
                d_type = self.params.dihedral_types[('X', quad[1], quad[2], 'X')][0]
            dihedral_e += d_type.phi_k * (1 + np.cos(d_type.per * d_angle - d_type.phase))
        # print("Dihedral Energy: " + str(dihedral_e))
        total += dihedral_e

        improper_e = 0
        for quad, angle in m.impropers:
            if (quad[0], quad[1], quad[2], quad[3]) in self.params.improper_types.keys():
                i_type = self.params.improper_types[(quad[0], quad[1], quad[2], quad[3])]
            elif (quad[0], 'X', 'X', quad[3]) in self.params.improper_types.keys():
                i_type = self.params.improper_types[(quad[0], 'X', 'X', quad[3])]
            else:
                i_type = self.params.improper_types[(quad[3], 'X', 'X', quad[0])]
            improper_e += i_type.psi_k * ((angle - i_type.psi_eq) ** 2)
        # print("Improper Energy: " + str(improper_e))
        total += improper_e

        ub_e = 0
        for triplet, dist in m.ub:
            ub_type = self.params.urey_bradley_types[(triplet[0], triplet[1], triplet[2])]
            ub_e += ub_type.k * ((dist - ub_type.req) ** 2)
        # print("UB Energy: " + str(ub_e))
        total += ub_e

        nonbonded_e = 0
        for pair, dist in m.nonbonded:
            nb_type_0, nb_type_1 = self.params.atom_types_str[pair[0]], self.params.atom_types_str[pair[1]]

            eps_0, eps_1 = nb_type_0.epsilon, nb_type_1.epsilon
            eps = np.sqrt(eps_0 * eps_1)

            rmin_0, rmin_1 = nb_type_0.rmin, nb_type_1.rmin
            rmin = rmin_0 + rmin_1

            lennard_jones = eps * (((rmin / dist) ** 12) - 2 * ((rmin / dist) ** 6))  # CHECK IF THE 2 IS NEEDED
            nonbonded_e += lennard_jones

            q1, q2 = nb_type_0.charge, nb_type_1.charge
            nonbonded_e += (q1 * q2) / (eps * dist)
        # print("Nonbonded Energy: " + str(nonbonded_e))
        total += nonbonded_e

        return total


def optimize(m, sp):
    positions = [pos for pos in m.positions.values()]

    print("Initialized molecule")
    grad_func = grad(sp.eval_energy)

    for i in range(200):
        print(sp.eval_energy(positions))

        gradient = grad_func(positions)
        positions = [[positions[i][0] - 5e-6 * float(gradient[i][0]),
                      positions[i][1] - 5e-6 * float(gradient[i][1]),
                      positions[i][2] - 5e-6 * float(gradient[i][2])] for i in range(len(positions))]


if __name__ == '__main__':

    m = Molecule(r'../pdb-files/ala5.psf', r'../params/par_all36m_prot.prm')
    sp = StructurePredictor(m, r'../params/par_all36m_prot.prm')


    positions = [pos for pos in m.positions.values()]
    print("Total energy: " + str(sp.eval_energy(positions)))

    optimize(m, sp)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for atom in m.positions.keys():
        pos = list(m.positions[atom])
        pos = (lambda x: [el._value for el in x])(pos)
        ax.scatter(pos[0], pos[1], pos[2])

        for bond in atom.bonds:
            pos_2 = list(m.positions[m.get_other_atom(bond, atom)])
            pos_2 = (lambda x: [el._value for el in x])(pos_2)
            ax.plot([pos[0], pos_2[0]], [pos[1], pos_2[1]], [pos[2], pos_2[2]])

    plt.show()
