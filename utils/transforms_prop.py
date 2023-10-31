import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

# Assume you've already defined this class in your project
from datasets.rna_ligand_data import RNALigandData

class FeaturizeRNAAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 15])    # H, C, N, O, P

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0)

    def __call__(self, data: RNALigandData):
        element = data.rna_element.view(-1, 1) == self.atomic_numbers.view(1, -1)
        data.rna_atom_feature = element
        return data


class FeaturizeLigandAtom(object):
    
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0)

    def __call__(self, data: RNALigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)
        atom_feature = data.ligand_atom_feature
        data.ligand_atom_feature_full = torch.cat([element, atom_feature], dim=-1)
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: RNALigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=4)
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
