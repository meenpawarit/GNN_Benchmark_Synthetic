# GNN Synthetic Benchmark

This project benchmarks Graph Neural Network architectures (GCN, GAT, GIN) and MLP on synthetic datasets to evaluate various properties:
1.  **Scalability** (Size Effect)
2.  **Homophily** vs Heterophily
3.  **Graph Structure** (Power-law vs Uniform)
4.  **Oversmoothing** (Depth Effect)

## Project Structure

- `models/`: Implementations of GCN, GAT, GIN, MLP.
- `data/`: Directory for synthetic datasets (generated on demand).
- `results/`: Directory for experiment logs and plots.
- `run_experiments.py`: Main script to execute training and evaluation.
- `plot_results.py`: Script to generate plots and LaTeX tables from results.
- `generate_data.py`: Script to generate the synthetic datasets.
- `train_utils.py`: Utilities for training loops and hyperparameter tuning.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```
Key dependencies: `torch`, `torch_geometric`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `networkx`, `scikit-learn`.

## Reproducing Results

### 1. Generate Data
Create the synthetic graphs required for experiments:
```bash
python3 generate_data.py
```

### 2. Run Experiments
Execute the full benchmark suite. This will perform hyperparameter tuning (Random Search, 5 trials) and run each configuration 5 times with different seeds.
```bash
python3 run_experiments.py
```
*Note: This process may take significant time depending on hardware.*

### 3. Visualize Results
Generate plots and LaTeX tables:
```bash
python3 plot_results.py
```
Output location: `results/plots/`
- Plots: `*.png`
- Tables: `table_*.tex` (Mean ± Std Dev)


