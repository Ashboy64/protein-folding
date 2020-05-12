import numpy as np
import parmed as pmd
from parmed.charmm import CharmmParameterSet


class Molecule(object):
    """Represents a molecule"""

    def __init__(self, pdb_filepath, param_filepath):
        super(Molecule, self).__init__()
        self.pdb = pmd.load_file(pdb_filepath)
        self.params = CharmmParameterSet(param_filepath)
        self.check_atoms()

    def check_atoms(self):

        not_in_params = []

        for atom in self.pdb.atoms:
            print(atom.name in self.params.atom_types.keys())

    @property
    def bonds(self):
        all = []
        for bond in self.pdb.bonds:
            a1 = bond.atom1
            a2 = bond.atom2
            all.append([(a1.name, a2.name), self.dist(a1, a2)])
        return all

    def dist(self, a1, a2):
        return np.sqrt((a1.xx - a2.xx)**2 + (a1.xy - a2.xy)**2 + (a1.xz - a2.xz)**2)


if __name__ == '__main__':
    m = Molecule(r'../pdb-files/1l2y.pdb', r'../params/par_all36m_prot.prm')
    print(m.bonds)
