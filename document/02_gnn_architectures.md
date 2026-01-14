# Step 2: GNN Architectures & Baselines

## Overview
In this step, we implement the neural network architectures used for the study. We compare Message Passing Neural Networks (MPNNs) against a simple Multi-Layer Perceptron (MLP) baseline.

## 1. Multi-Layer Perceptron (MLP) - The Baseline
*   **Concept**: A standard feedforward neural network that operates on each node's features *independently*.
*   **Message Passing**: None. It ignores the graph structure ($A$) completely.
*   **Equation**: $H^{(l+1)} = \sigma(H^{(l)} W^{(l)})$
*   **Why use it?**: To check if graph structure is actually useful. If MLP performs as well as GNNs, it suggests the graph structure adds no signal (or the GNN is failing to exploit it).

## 2. Graph Convolutional Network (GCN)
*   **Concept**: Aggregates information from immediate neighbors using a fixed, symmetric normalization.
*   **Mechanism**: Averages neighbor features (smoothing).
*   **Equation**: $H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} H^{(l)} W^{(l)})$
*   **Behavior**: Good at smoothing local information. Susceptible to "oversmoothing" in deep networks or dense graphs.

## 3. Graph Attention Network (GAT)
*   **Concept**: Computes learnable attention weights for each neighbor.
*   **Mechanism**: A node can learn to value some neighbors more than others.
*   **Equation**: $H^{(l+1)} = \sigma(\sum_{j \in N(i)} \alpha_{ij} W H_j^{(l)})$
*   **Behavior**: More expensive (O(V+E)) but capable of filtering out noisy edges. Should perform better in heterophilous settings if tuned correctly.

## 4. Graph Isomorphism Network (GIN)
*   **Concept**: Designed to be as expressive as the Weisfeiler-Lehman (WL) graph isomorphism test.
*   **Mechanism**: Uses a sum aggregator (instead of mean) and an MLP after aggregation.
*   **Equation**: $H^{(l+1)} = \text{MLP}((1+\epsilon)H^{(l)} + \sum_{j \in N(i)} H_j^{(l)})$
*   **Behavior**: Very powerful at distinguishing graph structures. Can map different structures to different embeddings better than GCN.

## Implementation Details
All models will inherit from `torch.nn.Module` and share a similar interface:
*   `__init__(in_channels, hidden_channels, out_channels)`
*   `forward(x, edge_index)`: Note that MLP will ignore `edge_index`.
