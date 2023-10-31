import torch
import torch.nn.functional as F
import numpy as np

from dataset.py import RNALigandData  # This line might need adjustments depending on RNA data representation

# Simplified mapping for atom types (assuming RNA structures)
MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,   # Hydrogen (H)
    6: 1,   # Carbon (C)
    7: 2,   # Nitrogen (N)
    8: 3,   # Oxygen (O)
    15: 4   # Phosphorus (P)
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}

def get_atomic_number_from_index(index, mode):
    return [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]

def get_index(atom_num, mode):
    return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]

class FeaturizeRNAAtom(object):
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 15])  # H, C, N, O, P

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
        data.protein_pos = data.protein_pos @ Q
        return data
