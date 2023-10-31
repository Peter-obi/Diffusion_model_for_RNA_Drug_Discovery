
import torch
import json
from torch_geometric.data import Data

# Load the saved atoms data and interactions from the JSON file
with open("simplified_interactions.json", "r") as file:
    data = json.load(file)
    atoms_data = data["atoms_data"]
    interactions = data["interactions"]

# Extract unique elements and interaction types for one-hot encoding
unique_elements = list(set(atom["element"] for atom in atoms_data))
unique_interactions = list(set(interaction["interaction_type"] for interaction in interactions))

# Create node features (one-hot encoded element types)
x = torch.zeros(len(atoms_data), len(unique_elements))
for i, atom in enumerate(atoms_data):
    x[i, unique_elements.index(atom["element"])] = 1

# Create edge indices and edge attributes (one-hot encoded interaction types)
edge_index = torch.tensor([[interaction["atom1"], interaction["atom2"]] for interaction in interactions], dtype=torch.long).t().contiguous()
edge_attr = torch.zeros(len(interactions), len(unique_interactions))
for i, interaction in enumerate(interactions):
    edge_attr[i, unique_interactions.index(interaction["interaction_type"])] = 1

# Create the graph
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# You can now use this graph_data as input to GNN models in PyTorch Geometric
