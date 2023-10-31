import os
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
import argparse
import shutil

ATOM_DICT = {
    "C": 0, "N": 1, "O": 2, "P": 3, "H": 4, "S": 5,
    "MG": 6, "K": 7, "CA": 8, "NA": 9, "CL": 10, "ZN": 11, "MN": 12,
    "CU": 13, "FE": 14, "BR": 15, "F": 16, "I": 17
}

RNA_DICT = {
    "A": 0, "C": 1, "G": 2, "U": 3
}

DIST_CUTOFF = 4.5  # Cutoff for interactions in Angstroms

def process_rna_and_ligand(pdbfile):
    structure = PDBParser(QUIET=True).get_structure('RNA_LIG', pdbfile)
    lig_coords = []
    lig_one_hot = []
    rna_coords = []
    rna_one_hot = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                for atom in residue:
                    atom_name = atom.get_name().strip()
                    if atom_name[0] == 'H':  # skip hydrogens
                        continue
                    if "H_" in resname:  # Ligand processing
                        lig_coords.append(atom.get_coord())
                        lig_one_hot.append(ATOM_DICT.get(atom_name, ATOM_DICT["C"]))  # default to carbon
                    else:  # RNA processing
                        rna_coords.append(atom.get_coord())
                        rna_one_hot.append(RNA_DICT.get(resname, RNA_DICT["A"]))  # default to Adenine

    return {
        'lig_coords': np.array(lig_coords),
        'lig_one_hot': np.array(lig_one_hot),
        'rna_coords': np.array(rna_coords),
        'rna_one_hot': np.array(rna_one_hot)
    }

def main(args):
    base_dir = args.basedir

    all_files = [f for f in os.listdir(base_dir) if f.endswith('.pdb')]
    
    lig_coords_list = []
    lig_one_hot_list = []
    rna_coords_list = []
    rna_one_hot_list = []

    for file in all_files:
        filepath = os.path.join(base_dir, file)
        data = process_rna_and_ligand(filepath)
        
        lig_coords_list.append(data['lig_coords'])
        lig_one_hot_list.append(data['lig_one_hot'])
        rna_coords_list.append(data['rna_coords'])
        rna_one_hot_list.append(data['rna_one_hot'])

    lig_coords = np.concatenate(lig_coords_list, axis=0)
    lig_one_hot = np.concatenate(lig_one_hot_list, axis=0)
    rna_coords = np.concatenate(rna_coords_list, axis=0)
    rna_one_hot = np.concatenate(rna_one_hot_list, axis=0)

    # Save the processed data
    np.savez('processed_data.npz', lig_coords=lig_coords, lig_one_hot=lig_one_hot,
             rna_coords=rna_coords, rna_one_hot=rna_one_hot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path, help="Directory containing the PDB files")
    args = parser.parse_args()
    main(args)

