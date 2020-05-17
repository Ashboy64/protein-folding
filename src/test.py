from parmed import load_file, Bond
from parmed.charmm import CharmmPsfFile, CharmmParameterSet

params = CharmmParameterSet(r'../params/par_all36m_prot.prm')
psf = load_file(r'../pdb-files/step1_pdbreader.psf')

print(psf.__dict__)

