import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style
sns.set_theme(style="whitegrid")
RESULTS_DIR = "results/experiments"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_latex_table(df, experiment_name, columns):
    """
    Generates and saves a LaTeX table from the dataframe.
    """
    table_path = os.path.join(PLOTS_DIR, f"table_{experiment_name}.tex")
    
    
    display_df = df.copy()
    
    # Auto-detect mean/std pairs 
    metric_cols = [c for c in columns if 'std' not in c and c in df.columns]
    
    final_cols = []
    
    for col in metric_cols:# e.g. "Test Accuracy" -> "Test Accuracy_std" NO, keys are usually distinct
        
        
        pass
        
    with open(table_path, 'w') as f:
        f.write("% Auto-generated table\n")
        f.write(display_df.to_latex(index=False, float_format="%.4f"))
    print(f"Saved LaTeX table to {table_path}")


def load_results(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_scalability(data):
    """
    Plots Training Time vs Num Nodes (Size Effect).
    """
    print("Plotting Scalability...")
    records = []
    for entry in data:
        records.append({
            'Model': entry['model'],
            'Num Nodes': entry['num_nodes'],
            'Training Time (s)': entry['train_time'],
            'Test Accuracy': entry['test_acc'],
            'Epochs': np.mean([r['epochs_run'] for r in entry['all_runs']])
        })
    df = pd.DataFrame(records)
    
    # Sort by N
    df = df.sort_values(by='Num Nodes')
    
    if df.empty:
        print("Error: No data found for scalability plot.")
        return

    # 1. Runtime Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Num Nodes', y='Training Time (s)', hue='Model', marker='o')
    plt.title('Training Runtime vs Graph Size')
    plt.ylabel('Time (s)')
    plt.xlabel('Number of Nodes')
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_scalability_runtime.png'))
    plt.close()
    
    # 2. Accuracy Plot 
    plt.figure(figsize=(10, 6))
    # Plot mean lines
    sns.lineplot(data=df, x='Num Nodes', y='Test Accuracy', hue='Model', marker='o')
    
    # Add error bands if std is available
    if 'test_acc_std' in df.columns:
        models = df['Model'].unique()
        colors = sns.color_palette()
        for i, model in enumerate(models):
            model_df = df[df['Model'] == model].sort_values('Num Nodes')
            plt.fill_between(model_df['Num Nodes'], 
                             model_df['Test Accuracy'] - model_df['test_acc_std'],
                             model_df['Test Accuracy'] + model_df['test_acc_std'],
                             alpha=0.2, color=colors[i])
                             
    plt.title('Accuracy vs Graph Size')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_scalability_accuracy.png'))
    plt.close()

    # 3. Convergence Plot (Epochs)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Num Nodes', y='Epochs', hue='Model', marker='o')
    plt.title('Training Convergence vs Graph Size')
    plt.ylabel('Epochs to Converge')
    plt.xlabel('Number of Nodes')
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_convergence_size.png'))
    plt.close()
    
    # Generate Table
    generate_latex_table(df, "size_effect", ['Num Nodes', 'Model', 'Test Accuracy', 'test_acc_std', 'Training Time (s)'])

def plot_homophily(data):
    """
    Plots Accuracy vs Homophily Configuration.
    """
    print("Plotting Homophily Effect...")
    records = []
    
    order_map = {'high': 0, 'medium': 1, 'low': 2, 'structural': 3}
    
    for entry in data:
        records.append({
            'Model': entry['model'],
            'Config': entry['config'],
            'Order': order_map.get(entry['config'], 99),
            'Test Accuracy': entry['test_acc'],
            'test_acc_std': entry.get('test_acc_std', 0),
            'F1 Score': entry['test_f1'],
            'test_f1_std': entry.get('test_f1_std', 0),
            'Epochs': np.mean([r['epochs_run'] for r in entry['all_runs']])
        })
    df = pd.DataFrame(records)
    if df.empty: return
    df = df.sort_values(by='Order')
    
    # 1. Accuracy Plot with Error Bars
    plt.figure(figsize=(10, 6))
    
    models = df['Model'].unique()
    configs = df['Config'].unique()
    
    # Use proper bar plot with error bars
    sns.barplot(data=df, x='Config', y='Test Accuracy', hue='Model')
    
    plt.title('Model Performance vs Homophily Level')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_homophily.png'))
    plt.close()
    
    # Generate Table
    generate_latex_table(df, "homophily", [])

    # 2. Convergence Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Config', y='Epochs', hue='Model')
    plt.title('Convergence Speed vs Homophily Level')
    plt.ylabel('Epochs to Converge')
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_homophily_convergence.png'))
    plt.close()

def plot_structure(data):
    """
    Plots Accuracy on BA vs ER graphs.
    """
    print("Plotting Structure Effect...")
    records = []
    for entry in data:
        # Simplification: "graph_ba_hubs.pt" -> "BA (Hubs)"
        graph_type = "BA (Hubs)" if "ba" in entry['dataset'] else "ER (Uniform)"
        
        records.append({
            'Model': entry['model'],
            'Graph Type': graph_type,
            'Test Accuracy': entry['test_acc'],
            'test_acc_std': entry.get('test_acc_std', 0)
        })
    df = pd.DataFrame(records)
    if df.empty: return
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Graph Type', y='Test Accuracy', hue='Model')
    plt.title('Impact of Graph Structure (Hubs vs Uniform)')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_structure_perf.png'))
    plt.close()
    
    generate_latex_table(df, "structure", [])

def plot_oversmoothing(data):
    """
    Plots Accuracy vs Depth (Number of Layers).
    """
    print("Plotting Oversmoothing...")
    records = []
    for entry in data:
        records.append({
            'Model': entry['model'],
            'Depth': entry['depth'],
            'Test Accuracy': entry['test_acc'],
            'test_acc_std': entry.get('test_acc_std', 0)
        })
    df = pd.DataFrame(records)
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    # Plot mean
    sns.lineplot(data=df, x='Depth', y='Test Accuracy', hue='Model', marker='o')
    
    # Add error bands
    if 'test_acc_std' in df.columns:
        models = df['Model'].unique()
        colors = sns.color_palette()
        for i, model in enumerate(models):
            model_df = df[df['Model'] == model].sort_values('Depth')
            plt.fill_between(model_df['Depth'], 
                             model_df['Test Accuracy'] - model_df['test_acc_std'],
                             model_df['Test Accuracy'] + model_df['test_acc_std'],
                             alpha=0.2, color=colors[i])

    plt.title('Oversmoothing: Accuracy vs Network Depth')
    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy')
    plt.xticks(sorted(df['Depth'].unique()))
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_depth_oversmoothing.png'))
    plt.close()
    
    generate_latex_table(df, "depth", [])

def plot_embeddings():
    """
    Visualizes saved embeddings using PCA/t-SNE.
    """
    print("Plotting Embeddings...")
    # Find all embedding files
    files = glob.glob(os.path.join(RESULTS_DIR, "embeddings_*.pt"))
    if not files:
        print("No embedding files found.")
        return
        
    for f in files:
        name = os.path.basename(f).replace(".pt", "")
        # e.g. embeddings_GCN_depth32
        
        saved = torch.load(f, weights_only=False)
        emb = saved['embeddings'].numpy()
        y = saved['labels'].numpy()
        
        # PCA
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb)
        
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title(f"Node Embeddings (PCA): {name}")
        plt.savefig(os.path.join(PLOTS_DIR, f"plot_pca_{name}.png"))
        plt.close()

if __name__ == "__main__":
    import glob
    
    # 1. Scalability
    res_size = load_results("size_effect_results.json")
    if res_size: plot_scalability(res_size['data'])
    
    # 2. Homophily
    res_hom = load_results("homophily_effect_results.json")
    if res_hom: plot_homophily(res_hom['data'])
    
    # 3. Structure
    res_struct = load_results("structure_effect_results.json")
    if res_struct: plot_structure(res_struct['data'])
    
    # 4. Oversmoothing
    res_depth = load_results("depth_effect_results.json")
    if res_depth: plot_oversmoothing(res_depth['data'])
    
    # 5. Embeddings
    plot_embeddings()
    
    print("\nAll plots generated in results/plots/")
