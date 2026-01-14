import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, homophily
import os
import glob

def plot_degree_distribution(data, title, save_path):
    """Plots and saves the degree distribution of a graph."""
    G = to_networkx(data, to_undirected=True)
    degrees = [d for n, d in G.degree()]
    
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=50, alpha=0.75, color='b', edgecolor='black')
    plt.title(f"Degree Distribution: {title}")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved degree plot to {save_path}")

def plot_adjacency_heatmap(data, title, save_path):
    """Plots the adjacency matrix to visualize clustering/homophily."""
    # Sort nodes by label to see the block structure
    sorted_indices = torch.argsort(data.y)
    edge_index = data.edge_index
    
    # Create a sparse matrix or just plot points for large graphs
    # For visualization, we can just scatter plot the edges
    # Mapping old indices to new sorted indices
    
    # This is a bit complex to map efficiently for large N, 
    # but for visualization we can reorder the adjacency matrix.
    
    # Easier way: Convert to dense, reorder, plot
    if data.num_nodes > 2000:
        print(f"Skipping heatmap for large graph {title} (N={data.num_nodes})")
        return

    adj = torch.zeros((data.num_nodes, data.num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    
    # Reorder based on labels
    adj_sorted = adj[sorted_indices][:, sorted_indices]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(adj_sorted.numpy(), cmap='Greys', interpolation='nearest')
    plt.title(f"Adjacency Matrix (Sorted by Class): {title}")
    plt.xlabel("Node Index (Class Sorted)")
    plt.ylabel("Node Index (Class Sorted)")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")

def analyze_homophily(data, name):
    """Calculates homophily ratio."""
    # Node homophily: fraction of edges connecting nodes of same class
    h = homophily(data.edge_index, data.y, method='edge')
    print(f"[{name}] Homophily: {h:.4f}")
    return h

def main():
    output_dir = "results/data_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze Structure Effect (BA vs ER)
    print("\n--- Analyzing Structure (Degree Distribution) ---")
    ba_path = "data/synthetic/structure_effect/graph_ba_hubs.pt"
    er_path = "data/synthetic/structure_effect/graph_er_uniform.pt"
    
    if os.path.exists(ba_path):
        data = torch.load(ba_path, weights_only=False)
        plot_degree_distribution(data, "BA (Hubs)", f"{output_dir}/degree_ba.png")
    
    if os.path.exists(er_path):
        data = torch.load(er_path, weights_only=False)
        plot_degree_distribution(data, "ER (Uniform)", f"{output_dir}/degree_er.png")

    # 2. Analyze Homophily Effect (SBM)
    print("\n--- Analyzing Homophily (Adjacency Structure) ---")
    hom_files = glob.glob("data/synthetic/homophily_effect/*.pt")
    for f in hom_files:
        name = os.path.basename(f).replace(".pt", "")
        data = torch.load(f, weights_only=False)
        
        analyze_homophily(data, name)
        plot_adjacency_heatmap(data, name, f"{output_dir}/heatmap_{name}.png")

if __name__ == "__main__":
    main()
