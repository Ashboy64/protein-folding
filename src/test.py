from parmed import load_file
from parmed.charmm import CharmmPsfFile, CharmmParameterSet

params = CharmmParameterSet()
params.read_topology_file(r'../params/top_all36_prot.rtf')

psf = load_file('../pdb-files/step1_pdbreader.pdb')
print(psf.__dict__)