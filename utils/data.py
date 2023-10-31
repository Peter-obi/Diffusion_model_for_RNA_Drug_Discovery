import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}

class PDBRNA(object):
    # Nucleotide names and their single letter representation
    NUCL_NAME_SYM = {
        'A': 'A',  # Adenine
        'U': 'U',  # Uracil
        'G': 'G',  # Guanine
        'C': 'C',  # Cytosine
    }

    NUCL_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(NUCL_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_nucleotide_type = []
        # Nucleotide properties
        self.nucleotides = []
        self.nucleotide_type = []
        self.center_of_mass = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break

    def _parse(self):
        # Process atoms
        nucleotides_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_nucleotide_type.append(self.NUCL_NAME_NUMBER.get(atom['res_name'], -1))  # set default to -1 for unrecognized residues

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in nucleotides_tmp:
                nucleotides_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                nucleotides_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process nucleotides
        self.nucleotides = [r for _, r in nucleotides_tmp.items()]
        for nucleotide in self.nucleotides:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in nucleotide['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
            nucleotide['center_of_mass'] = sum_pos / sum_mass

        for nucleotide in self.nucleotides:
            self.nucleotide_type.append(self.NUCL_NAME_NUMBER.get(nucleotide['name'], -1))
            self.center_of_mass.append(nucleotide['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_),
            'atom_name': self.atom_name,
            'atom_to_nucleotide_type': np.array(self.atom_to_nucleotide_type, dtype=np.int64)
        }

    def to_dict_nucleotide(self):
        return {
            'nucleotide_type': np.array(self.nucleotide_type, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32)
        }



    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id


def parse_sdf_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int64)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
    }
    return data
