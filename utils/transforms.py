import torch
import torch.nn.functional as F
import numpy as np

from datasets.pl_data import RNALigandData  # Ensure you have the correct import for RNA-ligand data representation

# Retaining the full atom type mappings
MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,   # Hydrogen (H)
    6: 1,   # Carbon (C)
    7: 2,   # Nitrogen (N)
    8: 3,   # Oxygen (O)
    9: 4,   # Fluorine (F)
    15: 5,  # Phosphorus (P)
    16: 6,  # Sulfur (S)
    17: 7,  # Chlorine (Cl)
    34: 8   # Selenium (Se)
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}

def get_atomic_number_from_index(index, mode):
    return [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]

def get_index(atom_num, mode):
    return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]

class FeaturizeRNAAtom(object):
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(list(MAP_ATOM_TYPE_ONLY_TO_INDEX.keys()))

    @property
    def feature_dim(self):
        return len(self.atomic_numbers)

    def __call__(self, data: RNALigandData):
        element = (data.rna_element.view(-1, 1) == self.atomic_numbers.view(1, -1)).float()
        data.rna_atom_feature = element
        return data

class FeaturizeLigandAtom(object):
    def __init__(self, mode='basic'):
        super().__init__()
        self.mode = mode

    @property
    def feature_dim(self):
        return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)

    def __call__(self, data: RNALigandData):
        element_list = data.ligand_element
        x = [get_index(e, self.mode) for e in element_list]
        x = torch.tensor(x)
        data.ligand_atom_feature = x
        return data

class RandomRotation(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data: RNALigandData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.rna_pos = data.rna_pos @ Q
        return data

class FeaturizeLigandBond(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data: RNALigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=4)    # Assuming 4 bond types
        return data



class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True, 
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        return data


class EdgeConnection(object):
    def __init__(self, kind, k):
        super(EdgeConnection, self).__init__()
        self.kind = kind
        self.k = k

    def __call__(self, data):
        pos = torch.cat([data.protein_pos, data.ligand_pos], dim=0)
        if self.kind == 'knn':
            data.edge_index = knn_graph(pos, k=self.k, flow='target_to_source')
        return data


def convert_to_single_emb(x, offset=128):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x