import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=1, dropout=0.5):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            
        self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        """
        Forward pass for GAT.
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.elu(x) # ELU is often used in GAT papers
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer
        x = self.layers[-1](x, edge_index)
        return x
