from parmed import load_file, Bond
from parmed.charmm import CharmmPsfFile, CharmmParameterSet

params = CharmmParameterSet(r'../params/par_all36m_prot.prm')
psf = load_file(r'../pdb-files/ala5.psf')
crds = load_file(r'../pdb-files/ala5.pdb')

print(crds.atoms[0].__dict__)