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
                self.positions[atom] = [counter, counter, counter]
                counter += 1

        if len(not_in_params) != 0:
            print("not_in_params: " + str(not_in_params))

    @property
    def bonds(self):
        all = []
        for bond in self.psf.bonds:
            a1 = bond.atom1
            a2 = bond.atom2
            all.append([(a1.type, a2.type), self.dist(a1, a2)])
        return all

    def dist(self, a1, a2):
        a1_x, a1_y, a1_z = self.positions[a1]
        a2_x, a2_y, a2_z = self.positions[a2]
        return np.sqrt((a1_x - a2_x)**2 + (a1_y - a2_y)**2 + (a1_z - a2_z)**2)


if __name__ == '__main__':
    m = Molecule(r'../pdb-files/step1_pdbreader.psf', r'../params/par_all36m_prot.prm')
    print(m.bonds)
