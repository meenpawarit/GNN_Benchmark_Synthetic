# Step 4: Results & Analysis Methodology

## Overview
This document outlines how we interpret the raw experimental results to draw conclusions about the GNN behaviors.

## 1. Analyzing Scalability (Size Effect)
*   **Plot**: Training Time (y-axis) vs Graph Size $N$ (x-axis) [Line Plot]
*   **Plot**: Peak Memory (y-axis) vs Graph Size $N$ (x-axis) [Line Plot]
*   **Plot**: Epochs to Convergence (y-axis) vs Graph Size $N$ (x-axis) [Line Plot]
*   **Analysis**:
    *   Linear vs Quadratic growth trends.
    *   Point of failure (OOM) for simple vs attention models.
    *   **Convergence**: Does size make optimization harder (more epochs) or is it invariant?

## 2. Analyzing Edge Complexity (Homophily Effect)
*   **Plot**: Test Accuracy (y-axis) vs Homophily Ratio (x-axis) [Bar/Line Plot]
*   **Plot**: Epochs to Convergence (y-axis) vs Homophily Ratio (x-axis) [Bar Plot]
*   **Expected Trend**: Positive correlation for GCN. Flatter slope for GAT/GIN implies robustness.
*   **Analysis**: 
    *   Identifying the "breaking point" where GNNs perform worse than the MLP baseline (meaning the graph structure is misleading).
    *   **Complexity Gap**: Calculate $\Delta = Acc_{GNN} - Acc_{MLP}$. If $\Delta < 0$, message passing is harmful.
    *   **Convergence**: Do heterophilous graphs requires more training epochs to find a solution?

## 3. Analyzing Structure Effect (Hubs vs Uniform)
*   **Plot**: Test Accuracy comparison between BA and ER graphs for each model [Grouped Bar Chart].
*   **Analysis**: Does the presence of Hubs aid (short paths) or hinder (noise propagation) message passing for GCN?

## 4. Analyzing Oversmoothing (Depth Effect)
*   **Plot**: Test Accuracy (y-axis) vs Network Depth (Layers 2, 4, 8, 16, 32) [Line Plot]
*   **Analysis**: Sharp decline indicates oversmoothing. We look for which architecture maintains performance longest.


## 5. Visualizing Embeddings
*   **Technique**: PCA or t-SNE on the latent representation of the final layer (before classifier).
*   **Goal**: Visually confirm if classes are separable.
    *   *Good result*: Distinct clusters corresponding to classes.
    *   *Oversmoothed*: All nodes collapse to a single point or indiscernible "blob".

## 6. Evaluating Expected Outcomes (Checklist)
Use this guide to validate your original proposal points:

### Outcome 1: Scalability & Computational Cost
> "Observe that graph size impacts GNN architectures differently..."

*   **Check**: `plot_scalability_runtime.png` & `plot_scalability_accuracy.png`.
*   **Look For**: Does GAT's runtime curve slope upwards much faster (quadratic-like) than GCN/GIN as $N$ increases?
*   **Look For**: Does `plot_convergence_size.png` show that larger graphs require significantly more epochs to train?

### Outcome 2: Edge Relationship Complexity
> "Find that edge relationship complexity strongly affects message passing..."

*   **Check**: `plot_homophily.png`.
*   **Look For**: A sharp drop in accuracy for GCN as you move from "High" to "Structural" (heterophily).
*   **Look For**: GAT or split-MLP models maintaining higher accuracy in the "Structural" column compared to GCN.

### Outcome 3: Oversmoothing Sensitivity
> "See that GCN is more sensitive to oversmoothing in larger or highly connected graphs."

*   **Check**: `plot_depth_oversmoothing.png`.
*   **Look For**: GCN accuracy crashing to random guess (approx 1/num_classes) at Depth=8 or 16, while GAT or GIN might degrade slower.
*   **Check**: `plot_pca_embeddings_GCN_depth32.png`. If it looks like a single dense blob, that is visual proof of oversmoothing.

### Outcome 4: GIN's Power Limits
> "Observe that GIN’s expressive power does not always translate to better performance when edge relationships become complex."

*   **Check**: `plot_homophily.png`.
*   **Look For**: GIN performing worse than GAT (or even MLP) in the "Structural" / Low Homophily setting. This suggests its theoretical power assumes homophily.

### Outcome 5: Harmful Message Passing
> "Identify scenarios where message passing becomes harmful, and MLP baselines outperform GNNs."

*   **Check**: All plots where MLP is included.
*   **Critical**: Look for any bar where **MLP > GCN/GAT/GIN**. This usually happens in `plot_homophily.png` under the "Structural" config. It proves that the graph edges are "noise" rather than "signal".

