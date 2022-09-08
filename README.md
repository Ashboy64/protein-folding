# Protein Folding

Toy programs that attempt to predict protein structure. Right now minimizes an energy function based on the the CHARMM 36m force field.

Parameters from: https://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-unix-html/node25.html
CHARMM Force Field: https://www.nature.com/articles/nmeth.4067

## Usage 

- Download the pdb file of the protein you want to model
- Prepare a psf file based on the pdb file you downloaded using CHARMM GUI (http://www.charmm-gui.org/?doc=input/pdbreader)
- Replace the path to the psf file in main.py with the path to the psf file you want to model
- Run main.py
