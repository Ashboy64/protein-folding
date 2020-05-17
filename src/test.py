from parmed import load_file
from parmed.charmm import CharmmPsfFile, CharmmParameterSet

params = CharmmParameterSet(r'../params/par_all36m_prot.prm')
psf = load_file(r'../pdb-files/step1_pdbreader.psf')

print(params.cmap_types[('C', 'NH1', 'CT1', 'C', 'NH1', 'CT1', 'C', 'NH1')].grid.__dict__)

