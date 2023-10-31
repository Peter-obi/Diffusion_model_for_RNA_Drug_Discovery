import os
import argparse
import pickle
from tqdm.auto import tqdm
from Bio.PDB import PDBParser, Select, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem

class LigandSelect(Select):
    def accept_residue(self, residue):
        return residue.id[0].startswith("H_")

class RNAWithinRadius(Select):
    def __init__(self, ligand_atoms, radius):
        self.ligand_atoms = ligand_atoms
        self.radius = radius

    def accept_residue(self, residue):
        # Exclude ligand residues
        if residue.id[0].startswith("H_"):
            return False
        
        for atom in residue:
            for ligand_atom in self.ligand_atoms:
                if atom - ligand_atom < self.radius:
                    return True
        return False

def extract_ligand_from_pdb(pdb_path, dest):
    structure = PDBParser(QUIET=True).get_structure('struct', pdb_path)
    io = PDBIO()
    io.set_structure(structure)
    ligand_path = os.path.join(dest, os.path.basename(pdb_path).replace('.pdb', '_ligand.pdb'))
    io.save(ligand_path, LigandSelect())
    return ligand_path

def convert_pdb_to_sdf(pdb_file, sdf_file):
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False)
    if mol:
        mol = Chem.AddHs(mol, addCoords=True)
        Chem.MolToMolFile(mol, sdf_file)

def get_rna_within_radius(pdb_file, ligand_file, radius=10.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('nucleic', pdb_file)
    ligand_struct = parser.get_structure('ligand', ligand_file)
    
    ligand_atoms = [atom for atom in ligand_struct.get_atoms()]

    io = PDBIO()
    pocket_file = os.path.join(os.path.dirname(ligand_file), os.path.basename(ligand_file).replace('.pdb', f'_pocket{radius}.pdb'))
    io.set_structure(structure)
    io.save(pocket_file, RNAWithinRadius(ligand_atoms, radius))
    
    return pocket_file

# ... [rest of the import statements and classes]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=os.getcwd(), help="Directory containing the PDB files.")
    parser.add_argument('--dest', type=str, required=True, help="Destination directory for extracted ligands and pockets.")
    parser.add_argument('--radius', type=float, default=10.0, help="Radius around the ligand for pocket extraction.")
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    pdb_files = [f for f in os.listdir(args.source) if f.endswith('.pdb')]
    index = []

    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_path = os.path.join(args.source, pdb_file)
        ligand_path = extract_ligand_from_pdb(pdb_path, args.dest)
        
        # Convert ligand pdb to sdf
        sdf_path = ligand_path.replace(".pdb", ".sdf")
        convert_pdb_to_sdf(ligand_path, sdf_path)
        
        pocket_path = get_rna_within_radius(pdb_path, ligand_path, args.radius)

        # Delete the ligand PDB file after conversion
        os.remove(ligand_path)

        index.append((os.path.basename(pocket_path), os.path.basename(sdf_path), 0))

    with open(os.path.join(args.dest, 'index.pkl'), 'wb') as f:
        pickle.dump(index, f)

    print(f"Saved index to {os.path.join(args.dest, 'index.pkl')}")
