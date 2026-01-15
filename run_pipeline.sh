# This script runs all experiments and generates visualizations.

# 1. Setup
echo "Starting GNN Project Pipeline"
mkdir -p results/experiments
mkdir -p results/plots

# 2. Run Experiments
# Note: This may take a significant amount of time depending on hardware.
echo "[1/2] Running Experiments (Training & Evaluation)..."
echo "      This will run training for GCN, GAT, GIN, and MLP across all datasets."
python3 run_experiments.py

if [ $? -ne 0 ]; then
    echo "Error: Experiment execution failed."
    exit 1
fi

# 3. Generate Visualizations
echo "--------------------------------------------------------"
echo "[2/2] Generating Visualizations..."
python3 plot_results.py

if [ $? -ne 0 ]; then
    echo "Error: Visualization generation failed."
    exit 1
fi

echo "Pipeline Completed Successfully!"
echo "Results saved in: results/experiments/"
echo "Plots saved in:   results/plots/"
