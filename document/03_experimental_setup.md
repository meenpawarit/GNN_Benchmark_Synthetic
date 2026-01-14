# Step 3: Experimental Setup

## Overview
This document details the experimental framework used to evaluate the GNN architectures. We ensure fair comparison by fixing random seeds and using consistent data splits.

## 1. Experimental Hypotheses
We aim to validate the following hypotheses:

*   **H1 (Scalability)**: Attention-based models (GAT) scale worse in runtime and memory than GCN/GIN as graph size ($N$) increases.
*   **H2 (Homophily)**: GCN performance degrades significantly in low-homophily (heterophilous) graphs, while GAT (via attention) and GIN (via higher expressivity) retain better performance.
*   **H3 (Structure)**: GCN is sensitive to "hub" nodes (Power-law graphs), potentially leading to oversmoothing or dominance of high-degree nodes compared to uniform graphs.
*   **H4 (Oversmoothing)**: Increasing the depth (number of layers) of GCN causes a faster drop in accuracy compared to GAT or ResNet-style GCNs, due to the oversmoothing phenomenon.

## 2. Evaluation Metrics
For all node classification tasks, we report:
*   **Accuracy**: Overall fraction of correct predictions.
*   **Precision (Macro)**: Unweighted mean of precision per class.
*   **Recall (Macro)**: Unweighted mean of recall per class.
*   **F1-Score (Macro)**: Harmonic mean of Precision and Recall.
*   **Training Time**: Time taken to complete the full training cycle.
*   **Peak Memory**: Maximum GPU/CPU memory usage during training.

## 3. Training Protocol
*   **Data Splits**: 60% Train, 20% Validation, 20% Test.
    *   *Note*: Splits are generated once using a fixed seed and saved/loaded to ensure all models see the exact same samples.
*   **Optimizer**: Adam
*   **Learning Rate**: 0.01 (tuned via initial sweeps)
*   **Weight Decay**: 5e-4
*   **Epochs**: 200
*   **Early Stopping**: Patience of 20 epochs based on Validation Loss.
*   **Runs**: Each experiment is repeated 5 times with different initialization seeds, and we report the mean ± std.

## 4. Hardware
Experiments are run on a personal machine (CPU/MPS). We log the device specifications for reproducibility.
