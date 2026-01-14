import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index=None):
        """
        Forward pass for MLP.
        
        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (Tensor, optional): Graph connectivity (Ignored by MLP)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer (no activation usually for logits, handled by CrossEntropyLoss)
        x = self.layers[-1](x)
        return x
