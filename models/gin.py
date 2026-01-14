import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # GIN requires an MLP for the aggregation step
        # Layer 1
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.layers.append(GINConv(mlp1))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            mlp_hidden = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.layers.append(GINConv(mlp_hidden))
            
        # Output Layer
        # Standard GIN usage: We usually get node embeddings and then use a linear classifier
        # But GINConv expects an MLP. So we can keep it as GINConv or just add a final Linear layer.
        # Here we follow the pattern: GINConv -> GINConv -> ... -> Linear Classifier
        
        mlp_last = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.layers.append(GINConv(mlp_last))
        
        # Final classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass for GIN.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Classifier
        x = self.classifier(x)
        return x
