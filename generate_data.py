import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import os

def generate_node_features(start_idx, end_idx, feature_dim, mean_shift=0.0):
    """
    Generates Gaussian node features.
    If mean_shift > 0, different blocks can have features shifted by this amount,
    correlating features with the class.
    """
    num_nodes = end_idx - start_idx
    # Base standard normal distribution
    features = torch.randn(num_nodes, feature_dim)
    # Shift mean if specified (simple way to signal class info via features)
    features += mean_shift
    return features

def generate_sbm_graph(num_nodes, num_classes, p_intra, p_inter, feature_dim=16):
    """
    Generates a Stochastic Block Model graph.
    """
    # Define block sizes (approx equal)
    block_size = num_nodes // num_classes
    sizes = [block_size] * num_classes
    # Handle remainder
    sizes[-1] += num_nodes % num_classes

    # Define probability matrix
    probs = [[p_inter] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        probs[i][i] = p_intra

    # Generate graph using NetworkX
    G_nx = nx.stochastic_block_model(sizes, probs, seed=42)
    
    # Create Data object
    data = from_networkx(G_nx)
    
    # Assign Features and Labels
    # In SBM from networkx, nodes are ordered by block 0, 1, 2...
    # We need to assign labels and features accordingly
    x_list = []
    y_list = []
    
    current_idx = 0
    for class_id, size in enumerate(sizes):
        # Create features for this block
        # We shift the mean of features based on class_id to make them discriminative
        # e.g. class 0 mean=0, class 1 mean=1, etc.
        # This is a simple assumption; we can adjust 'difficulty' later.
        feat = generate_node_features(0, size, feature_dim, mean_shift=class_id * 1.0)
        x_list.append(feat)
        
        # Labels
        y_list.append(torch.full((size,), class_id, dtype=torch.long))
        
        current_idx += size

    data.x = torch.cat(x_list, dim=0)
    data.y = torch.cat(y_list, dim=0)
    
    return data

def generate_er_graph(num_nodes, p, num_classes=2, feature_dim=16):
    """
    Generates an Erdős-Rényi (Random) graph.
    Labels are assigned randomly since there's no inherent structure.
    """
    G_nx = nx.erdos_renyi_graph(n=num_nodes, p=p, seed=42)
    data = from_networkx(G_nx)
    
    # Randomly assign labels
    labels = torch.randint(0, num_classes, (num_nodes,))
    data.y = labels
    
    # Features correlated with labels (otherwise it's impossible to learn!)
    x_list = []
    for i in range(num_nodes):
        label = labels[i].item()
        feat = torch.randn(1, feature_dim) + (label * 1.0) # Shift based on label
        x_list.append(feat)
    
    data.x = torch.cat(x_list, dim=0)
    
    return data

def generate_ba_graph(num_nodes, m, num_classes=2, feature_dim=16):
    """
    Generates a Barabási-Albert (Power-Law) graph.
    m: number of edges to attach from a new node to existing nodes.
    Labels assigned randomly (or could be degree-based, but we'll stick to random class for now)
    """
    G_nx = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=42)
    data = from_networkx(G_nx)
    
    # Randomly assign labels
    labels = torch.randint(0, num_classes, (num_nodes,))
    data.y = labels
    
    # Features correlated with labels
    x_list = []
    for i in range(num_nodes):
        label = labels[i].item()
        feat = torch.randn(1, feature_dim) + (label * 1.0)
        x_list.append(feat)
        
    data.x = torch.cat(x_list, dim=0)
    
    return data

if __name__ == "__main__":
    # 1. Size Effect (Scalability)
    # Goal: See how training time/memory scales with N. Keep homophily/density roughly constant.
    print("--- Generating Size Effect Dataset ---")
    size_dir = "data/synthetic/size_effect"
    os.makedirs(size_dir, exist_ok=True)
    
    sizes = [100, 500, 1000, 2000, 5000] # Can add 10000 if machine permits
    for n in sizes:
        # Standard SBM with reasonable homophily
        data = generate_sbm_graph(num_nodes=n, num_classes=5, p_intra=0.05, p_inter=0.005)
        torch.save(data, os.path.join(size_dir, f"sbm_{n}.pt"))
        print(f"Saved sbm_{n}.pt")


    # 2. Homophily Effect (Edge Complexity)
    # Goal: See how GNNs perform when edges don't match labels (heterophily). Fixed N.
    print("\n--- Generating Homophily Effect Dataset ---")
    hom_dir = "data/synthetic/homophily_effect"
    os.makedirs(hom_dir, exist_ok=True)
    
    fixed_n = 1000
    # H = p_intra / (p_intra + (num_classes - 1) * p_inter) (approx)
    # We want to vary "homophily strength". 
    # Let's verify p_intra vs p_inter ratios explicitly.
    
    configs = [
        ("high", 0.1, 0.001),   # Strong homophily
        ("medium", 0.05, 0.01), # Moderate
        ("low", 0.02, 0.02),    # Random mixing (approx)
        ("structural", 0.005, 0.05) # Heterophily (more likely to connect to others)
    ]
    
    for name, p_in, p_out in configs:
        data = generate_sbm_graph(num_nodes=fixed_n, num_classes=5, p_intra=p_in, p_inter=p_out)
        torch.save(data, os.path.join(hom_dir, f"sbm_hom_{name}.pt"))
        print(f"Saved sbm_hom_{name}.pt")


    # 3. Structure Effect (Hubs vs Uniform)
    # Goal: Compare ER (Uniform) vs BA (Hubs). Fixed N.
    # CRITICAL: We should try to match the Average Degree so the comparison is fair (density controlled).
    print("\n--- Generating Structure Effect Dataset ---")
    struct_dir = "data/synthetic/structure_effect"
    os.makedirs(struct_dir, exist_ok=True)
    
    fixed_n = 1000
    m_val = 5 # BA parameter (each new node adds 5 edges)
    # Avg degree in BA approx 2*m = 10.
    
    # Generate BA
    ba_data = generate_ba_graph(num_nodes=fixed_n, m=m_val, num_classes=5)
    torch.save(ba_data, os.path.join(struct_dir, "graph_ba_hubs.pt"))
    print("Saved graph_ba_hubs.pt")
    
    # Generate ER with matching avg degree ~10
    # p = avg_degree / (n-1) = 10 / 999 ~= 0.01
    p_matched = (2 * m_val) / (fixed_n - 1)
    er_data = generate_er_graph(num_nodes=fixed_n, p=p_matched, num_classes=5)
    torch.save(er_data, os.path.join(struct_dir, "graph_er_uniform.pt"))
    print("Saved graph_er_uniform.pt")
