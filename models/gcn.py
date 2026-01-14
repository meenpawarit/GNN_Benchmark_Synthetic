import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            
        # Output layer
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        """
        Forward pass for GCN.
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer
        x = self.layers[-1](x, edge_index)
        return x
