import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# MLP Model
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(73728, 3072)  # Adjusted to match input size
        self.fc2 = nn.Linear(3072, 96)
        self.fc3 = nn.Linear(96, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# GCN Model
class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNConv(128, 128)
        self.gcn2 = GCNConv(128, 4)
        self.fc = nn.Linear(4, 10)  # Adjusted to match gcn2 output size

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = self.fc(x)
        return x

# Example usage
if __name__ == "__main__":
    # Example data for MLP
    mlp_input = torch.randn(1, 73728)  # Adjusted input size
    mlp_model = MLPModel()
    mlp_output = mlp_model(mlp_input)
    print("MLP Output:", mlp_output)

    # Example data for GCN
    gcn_data = Data(x=torch.randn(24, 128), edge_index=torch.tensor([[0, 1], [1, 2]]))
    gcn_model = GCNModel()
    gcn_output = gcn_model(gcn_data)
    print("GCN Output:", gcn_output.shape)
