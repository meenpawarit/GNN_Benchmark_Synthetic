# Step 1: Synthetic Graph Generation

## Overview
This step focuses on generating the synthetic datasets required for our controlled study. The goal is to create graphs where we can explicitly tune specific properties ($N$, degree distribution, homophily) to observe their effect on GNN performance.

## Theoretical Concepts & Algorithms

### 1. Stochastic Block Model (SBM)
The Stochastic Block Model is a generative model for random graphs. It tends to produce graphs with community structure, which is ideal for simulating node classification tasks where "communities" correspond to "classes".

*   **Mechanism**: The graph is divided into $k$ communities (blocks).
*   **Parameters**:
    *   $P$: A $k \times k$ matrix of edge probabilities. $P_{ij}$ is the probability of an edge between a node in community $i$ and a node in community $j$.
    *   **Homophily Control**: By making diagonal elements ($P_{ii}$) much larger than off-diagonal elements ($P_{ij}$), we create strong homophily (nodes connect to their own class). By reversing this, we can simulate heterophily.
*   **Why use it?**: It allows precise control over the "difficulty" of the classification task by adjusting the ratio of intra-class to inter-class edges.

### 2. Random Graphs (Erdős-Rényi)
*   **Mechanism**: $G(N, p)$ connects every pair of $N$ nodes with probability $p$.
*   **Properties**: Produces a Poisson degree distribution (fairly uniform).
*   **Use Case**: Serves as a baseline "unstructured" graph to compare against structured ones.

### 3. Power-Law / Barabási-Albert (BA) Models
*   **Mechanism**: Uses preferential attachment. New nodes are more likely to attach to existing nodes with high degrees.
*   **Properties**: distinct "heavy-tailed" degree distribution (hubs exist).
*   **Use Case**: To test GNN behavior in the presence of "hub" nodes (dominance of high-degree nodes).

## Implementation Plan

### Tools
We will use **PyTorch Geometric (PyG)** `torch_geometric.datasets` (if applicable) or **NetworkX** `networkx.generators` converted to PyG Data objects.
*   `networkx.stochastic_block_model`: For SBM.
*   `networkx.barabasi_albert_graph`: For power-law graphs.

### Key Variables to Control
1.  **Graph Size ($N$)**: range from small (e.g., 100) to large (e.g., 5,000) to test scalability (Oversmoothing).
2.  **Average Degree ($d$)**: Controlled by the probability $p$ in SBM/ER.
3.  **Homophily ($h$)**: Defined as the fraction of edges connecting nodes of the same class.

### Output
The script `generate_data.py` outputs `Data` objects saved as `.pt` (PyTorch) files in `data/synthetic/` subdirectories:
*   `size_effect/`: Graphs with increasing $N$.
*   `homophily_effect/`: Graphs with varying homophily.
*   `structure_effect/`: BA vs ER graphs.
