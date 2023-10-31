from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
ATOM_DICT = {"C": 0, "N": 1, "O": 2, "P": 3, "H": 4, "S": 5}  # Sample atom dictionary, can be expanded
RNA_RESIDUE_DICT = {"A": 0, "U": 1, "G": 2, "C": 3}  # RNA residues

def process_rna_and_ligand(pdbfile):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    # Process ligand (assuming HETATM denotes ligand atoms)
    lig_atoms = [atom for atom in pdb_struct.get_atoms() if atom.parent.id == "HETATM"]
    lig_coords = np.array([atom.coord for atom in lig_atoms])
    lig_one_hot = np.array([ATOM_DICT[atom.element] for atom in lig_atoms])

    # Process RNA pocket
    rna_residues = [res for res in pdb_struct[0].get_residues() if res.id[0] == " "]
    rna_coords = np.array([atom.coord for res in rna_residues for atom in res])
    rna_one_hot = np.array([ATOM_DICT[atom.element] for res in rna_residues for atom in res])

    return {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
    }, {
        'rna_coords': rna_coords,
        'rna_one_hot': rna_one_hot,
    }
from rdkit.Chem import MolFromPDBBlock, MolToSmiles

def compute_smiles(pdbfile):
    """
    Compute the SMILES representation for the ligand in the given PDB file.
    """
    with open(pdbfile, 'r') as f:
        pdb_content = f.read()

    mol = MolFromPDBBlock(pdb_content)
    if mol:
        return MolToSmiles(mol)
    return None

def get_type_histograms(lig_one_hot, rna_one_hot):
    """
    Compute histograms for atom types in the ligand and residue types in the RNA.
    """
    atom_counts = {k: np.sum(lig_one_hot == v) for k, v in ATOM_DICT.items()}
    rna_residue_counts = {k: np.sum(rna_one_hot == v) for k, v in RNA_RESIDUE_DICT.items()}
    
    return atom_counts, rna_residue_counts

import os
from pathlib import Path
import argparse

def save_all(filename, pdb_ids, lig_coords, lig_one_hot, rna_coords, rna_one_hot):
    """
    Save the processed data to a numpy file.
    """
    np.savez(filename,
             names=pdb_ids,
             lig_coords=lig_coords,
             lig_one_hot=lig_one_hot,
             rna_coords=rna_coords,
             rna_one_hot=rna_one_hot
             )
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path, help="Base directory containing the PDB files.")
    parser.add_argument('--outdir', type=Path, default=None, help="Output directory for processed data.")
    args = parser.parse_args()

    # Define the directories
    data_dir = args.basedir

    # Create output directory if not specified
    if args.outdir is None:
        processed_dir = Path(data_dir, 'processed')
    else:
        processed_dir = args.outdir

    # Ensure the output directory exists
    processed_dir.mkdir(exist_ok=True, parents=True)

    lig_coords = []
    lig_one_hot = []
    rna_coords = []
    rna_one_hot = []
    pdb_ids = []

    # Iterate through all PDB files in the base directory
    for pdb_file in os.listdir(data_dir):
        if pdb_file.endswith(".pdb"):
            try:
                lig_data, rna_data = process_ligand_and_rna(os.path.join(data_dir, pdb_file))
                lig_coords.append(lig_data['lig_coords'])
                lig_one_hot.append(lig_data['lig_one_hot'])
                rna_coords.append(rna_data['rna_coords'])
                rna_one_hot.append(rna_data['rna_one_hot'])
                pdb_ids.append(pdb_file)
            except Exception as e:
                print(f"Failed processing {pdb_file} due to {str(e)}")
    
    # Convert lists to numpy arrays
    lig_coords = np.array(lig_coords)
    lig_one_hot = np.array(lig_one_hot)
    rna_coords = np.array(rna_coords)
    rna_one_hot = np.array(rna_one_hot)

    # Save the processed data
    save_all(processed_dir / 'processed_data.npz', pdb_ids, lig_coords, lig_one_hot, rna_coords, rna_one_hot)

    print("Processing completed!")
import warnings
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
import seaborn as sns
import matplotlib.pyplot as plt

def compute_smiles(positions, one_hot, ATOM_DICT):
    """
    Compute SMILES strings for ligands.
    """
    print("Computing SMILES ...")
    
    atom_types = [ATOM_DICT[idx] for idx in np.argmax(one_hot, axis=-1)]
    mol = Chem.MolFromXYZBlock(positions, atomTypes=atom_types)

    # Optimize molecule geometry
    try:
        UFFOptimizeMolecule(mol)
    except:
        print("UFF optimization failed. Skipping...")
        return None

    return Chem.MolToSmiles(mol)

def get_type_histograms(lig_one_hot, rna_one_hot, ATOM_DICT, nucleotide_dict):
    """
    Get histograms of ligand and RNA types.
    """
    atom_counts = {k: 0 for k in ATOM_DICT.keys()}
    for a in [ATOM_DICT[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    nuc_counts = {k: 0 for k in nucleotide_dict.keys()}
    for r in [nucleotide_dict[x] for x in rna_one_hot.argmax(1)]:
        nuc_counts[r] += 1

    return atom_counts, nuc_counts

def plot_histogram(data, title, save_path):
    """
    Plot and save a histogram.
    """
    sns.distplot(data)
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()

# Compute statistics & additional information
if __name__ == "__main__":
    with np.load(processed_dir / 'processed_data.npz', allow_pickle=True) as data:
        lig_one_hot = data['lig_one_hot']
        rna_one_hot = data['rna_one_hot']

    # Compute SMILES for ligands
    lig_smiles = compute_smiles(lig_coords, lig_one_hot, ATOM_DICT)
    with open(processed_dir / 'ligand_smiles.txt', 'w') as f:
        for s in lig_smiles:
            f.write(s + "\n")

    # Histograms of ligand and RNA types
    atom_hist, nuc_hist = get_type_histograms(lig_one_hot, rna_one_hot, ATOM_DICT, nucleotide_dict)

    # Plot histograms
    plot_histogram(list(atom_hist.values()), "Ligand Atom Distribution", processed_dir / 'ligand_atom_distribution.png')
    plot_histogram(list(nuc_hist.values()), "RNA Nucleotide Distribution", processed_dir / 'rna_nucleotide_distribution.png')

    print("Summary statistics generated!")
import json

def save_processed_data(filename, ligand_data, rna_data):
    """
    Save the processed ligand and RNA data.
    """
    with open(filename, 'w') as f:
        json.dump({
            "ligand_data": ligand_data,
            "rna_data": rna_data
        }, f)

    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Assuming you've loaded and processed data into `ligand_data` and `rna_data` dictionaries
    save_processed_data(processed_dir / 'final_processed_data.json', ligand_data, rna_data)

    # Create summary string
    summary_string = '# SUMMARY\n\n'
    summary_string += 'Processed RNA-Ligand Data\n\n'
    summary_string += '# Ligand Data\n'
    for key, value in ligand_data.items():
        summary_string += f"{key}: {len(value)} entries\n"
    
    summary_string += '\n# RNA Data\n'
    for key, value in rna_data.items():
        summary_string += f"{key}: {len(value)} entries\n"

    # Save summary
    with open(processed_dir / 'summary.txt', 'w') as f:
        f.write(summary_string)

    # Print summary
    print(summary_string)
