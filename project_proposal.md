# Third Project Proposal

## Team Members
*   Mateja Zatezalo
*   Pawarit Jamjod
*   Joan Acero

## Preliminary Title
A controlled study of how graph size and edge complexity affect message-passing graph neural networks

## Motivation
Graph Neural Networks (GNNs) learn by repeatedly passing and aggregating messages between connected nodes, yet it is often difficult to develop an intuitive understanding of how this message-passing mechanism behaves under different graph conditions. In particular, factors such as graph size and the complexity of edge relationships can significantly influence how information propagates through the network, potentially leading to issues such as oversmoothing, dominance of highly connected nodes, or increased computational cost. Different GNN architectures, including GCN, GIN, and GAT, implement message passing in distinct ways, but these differences are not always clear when models are only evaluated on fixed real-world datasets. The motivation of this project is therefore to use controlled graph settings to better understand how message passing works in practice, and how changes in graph size and edge structure affect the performance and behavior of different GNN architectures in node classification tasks. By focusing on these factors, the project aims to build intuition about when and why certain GNN designs work better than others.

## Goal
Our main goal is to:
*   Observe how different GNN architectures behave during message passing, and how their aggregation mechanisms change node representations as information propagates through the graph.
*   Study how graph size affects message passing and node classification performance by gradually increasing the number of nodes while controlling other graph properties.
*   Analyze the impact of edge relationship complexity on GNN behavior, focusing on how different connectivity patterns (e.g., uniform degree vs hub-dominated graphs, homophilous vs heterophilous edges) influence information flow.
*   Compare node classification performance across different GNN architectures, including GCN, GIN, and GAT, under controlled variations of graph size and edge structure.
*   Develop an intuitive understanding of when and why certain GNN architectures perform better or fail, with emphasis on message passing phenomena such as oversmoothing, dominance of high-degree nodes, and sensitivity to edge structure.

## Techniques to apply
**Models:**
*   Graph Convolutional Network (GCN)
*   Graph Isomorphism Network (GIN)
*   Graph Attention Network (GAT)
*   (Baseline) Multi-Layer Perceptron (MLP, no message passing)

**Graph Generation (Synthetic Data):**
*   Random graphs with controlled average degree
*   Configuration model graphs with increasing degree variance
*   Power-law graphs with hub-dominated structures
*   Stochastic Block Models (SBM) to control edge mixing between classes

**Controlled Factors:**
*   Number of nodes $N$
*   Edge relationship complexity
*   Degree distribution (uniform to skewed)
*   Edge mixing patterns (homophily to heterophily)

**Data Preprocessing:**
*   Fixed node feature distribution across experiments
*   Consistent train/validation/test splits
*   Same initialization seeds for fair comparison

## Evaluation
**For Node Classification:**
*   Accuracy, Precision, Recall, F1-score

**Computational Aspects:**
*   Training time vs graph size
*   Memory usage (especially for attention-based models)

## Visualization
*   Accuracy vs graph size plots for each GNN architecture
*   Accuracy vs edge complexity (degree variance, homophily ratio)
*   Performance vs depth plots to observe oversmoothing effects
*   2D visualizations of node embeddings (e.g., via PCA) colored by class
*   Runtime scaling plots comparing GCN, GIN, and GAT

## Data Sources
*   Fully synthetic graph datasets generated with controlled structural properties
*   Graph statistics computed for each generated graph (average degree, degree variance, assortativity, clustering coefficient)

## Expected Outcomes
*   Observe that graph size impacts GNN architectures differently, with attention-based models becoming more computationally expensive as graphs scale.
*   Find that edge relationship complexity strongly affects message passing, particularly in graphs with high degree skew or weak label-edge alignment.
*   See that GCN is more sensitive to oversmoothing in larger or highly connected graphs.
*   Observe that GIN’s expressive power does not always translate to better performance when edge relationships become complex.
*   Identify scenarios where message passing becomes harmful, and MLP baselines outperform GNNs.

## Fundamental References
*   William L. Hamilton, *Graph Representation Learning*
*   Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*
*   Xu et al., *How Powerful Are Graph Neural Networks?*
*   Velickovic et al., *Graph Attention Networks*

## Computational Resources
*   Experiments will be conducted on personal laptops
*   Graph sizes and model configurations are chosen to remain computationally feasible while still demonstrating scaling and complexity effects.

---

### Comment (Feedback)
> great goal, perhaps too ambitious by the time you have! - try not to reduce it to a purely experimental work; if it has to be like this, then make it statistically correct, in order to be able to extract valid conclusions; look for how to perform a statistically correct comparison of machine learning algortihms
