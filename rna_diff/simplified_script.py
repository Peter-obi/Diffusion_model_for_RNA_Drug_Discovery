
import numpy as np
import json

# Manual parsing of the PDB file
def parse_pdb(pdb_path):
    with open(pdb_path, "r") as f:
        pdb_lines = f.readlines()

    atoms_data = []
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            record_type = line[:6].strip()
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain = line[21].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            element = line[76:78].strip()

            atom_data = {
                "atom_name": atom_name,
                "residue_name": residue_name,
                "chain": chain,
                "coordinates": [x, y, z],
                "element": element,
                "record_type": record_type
            }
            atoms_data.append(atom_data)
    return atoms_data

# Calculate Euclidean distance between two 3D coordinates
def calculate_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

# Infer interactions between atoms based on simplified criteria
def infer_interactions_simplified(rna_atoms, ligand_atoms):
    interactions = []
    
    for rna_atom in rna_atoms:
        for ligand_atom in ligand_atoms:
            distance = calculate_distance(rna_atom["coordinates"], ligand_atom["coordinates"])
            
            # Infer type of interaction based on distance and atom types
            interaction_type = None
            if 2.5 <= distance <= 3.5:
                if ("O" in rna_atom["element"] and "H" in ligand_atom["element"]) or                    ("H" in rna_atom["element"] and "O" in ligand_atom["element"]):
                    interaction_type = "hydrogen_bond"
            elif distance <= 4.0:
                interaction_type = "hydrophobic"
            
            if interaction_type:
                interactions.append({
                    "atom1": rna_atoms.index(rna_atom),
                    "atom2": len(rna_atoms) + ligand_atoms.index(ligand_atom),
                    "distance": distance,
                    "interaction_type": interaction_type
                })
    return interactions

# Load the PDB file, infer interactions, and save to JSON
pdb_path = "5kx9.pdb"
atoms_data = parse_pdb(pdb_path)
rna_atoms = [atom for atom in atoms_data if atom["residue_name"] in ["A", "U", "G", "C"] and atom["record_type"] == "ATOM"]
ligand_atoms = [atom for atom in atoms_data if atom["record_type"] == "HETATM"]
interactions = infer_interactions_simplified(rna_atoms, ligand_atoms)

with open("simplified_interactions.json", "w") as file:
    json.dump({"atoms_data": atoms_data, "interactions": interactions}, file)
