import torch
import os
import glob
import json
import numpy as np
import argparse
from tqdm import tqdm

from models.gcn import GCN
from models.gat import GAT
from models.gin import GIN
from models.mlp import MLP
from models.mlp import MLP
from train_utils import run_single_experiment, tune_hyperparameters

# Hyperparameters 
DEFAULT_PARAMS = {
    'hidden_channels': 64,
    'num_layers': 3,
    'dropout': 0.5,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 200,
    'patience': 20
}

MODELS = {
    'GCN': GCN,
    'GAT': GAT,
    'GIN': GIN,
    'MLP': MLP
}

RESULTS_DIR = "results/experiments"

def save_results(results, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Convert tensors/numpy to list for JSON
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, (np.ndarray, np.generic)):
            serializable_results[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            serializable_results[k] = v.tolist()
        else:
            serializable_results[k] = v
            
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Saved results to {filepath}")

def run_size_effect_experiment(seeds=[42, 43, 44, 45, 46]):
    print("\n Running Experiment 1: Size Effect ")
    data_dir = "data/synthetic/size_effect"
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    
    summary = []
    
    for f in tqdm(files, desc="Datasets"):
        dataset_name = os.path.basename(f)
        try:
            n_nodes = int(dataset_name.split('_')[1].split('.')[0])
        except:
            n_nodes = 0
            
        data = torch.load(f, weights_only=False)
        
        for model_name, model_cls in MODELS.items():
            
            print(f" > Tuning {model_name} for {dataset_name}...")
            best_params = tune_hyperparameters(model_cls, data, DEFAULT_PARAMS, n_trials=5)
            
            run_metrics = []
            for seed in seeds:
                res = run_single_experiment(model_cls, data, best_params, seed)
                if 'embeddings' in res: del res['embeddings']
                if 'labels' in res: del res['labels']
                run_metrics.append(res)
            
            # Average over seeds
            avg_acc = np.mean([r['test_acc'] for r in run_metrics])
            std_acc = np.std([r['test_acc'] for r in run_metrics])
            avg_time = np.mean([r['train_time'] for r in run_metrics])
            
            entry = {
                'dataset': dataset_name,
                'num_nodes': n_nodes,
                'model': model_name,
                'test_acc': avg_acc,
                'test_acc_std': std_acc,
                'train_time': avg_time,
                'tuned_params': best_params,
                'all_runs': run_metrics
            }
            summary.append(entry)
    
    save_results({"experiment": "size_effect", "data": summary}, "size_effect_results.json")

def run_homophily_effect_experiment(seeds=[42, 43, 44, 45, 46]):
    print("\n Running Experiment 2: Homophily Effect ")
    data_dir = "data/synthetic/homophily_effect"
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    
    summary = []
    
    for f in tqdm(files, desc="Datasets"):
        dataset_name = os.path.basename(f)
        # Extract config name
        config_name = dataset_name.replace("sbm_hom_", "").replace(".pt", "")
        
        data = torch.load(f, weights_only=False)
        
        for model_name, model_cls in MODELS.items():
            print(f" > Tuning {model_name} for {dataset_name}...")
            best_params = tune_hyperparameters(model_cls, data, DEFAULT_PARAMS, n_trials=5)
            
            run_metrics = []
            for seed in seeds:
                res = run_single_experiment(model_cls, data, best_params, seed)
                if 'embeddings' in res: del res['embeddings']
                if 'labels' in res: del res['labels']
                run_metrics.append(res)
            
            avg_acc = np.mean([r['test_acc'] for r in run_metrics])
            std_acc = np.std([r['test_acc'] for r in run_metrics])
            avg_f1 = np.mean([r['test_f1'] for r in run_metrics])
            std_f1 = np.std([r['test_f1'] for r in run_metrics])
            
            entry = {
                'dataset': dataset_name,
                'config': config_name,
                'model': model_name,
                'test_acc': avg_acc,
                'test_acc_std': std_acc,
                'test_f1': avg_f1,
                'test_f1_std': std_f1,
                'tuned_params': best_params,
                'all_runs': run_metrics
            }
            summary.append(entry)
            
    save_results({"experiment": "homophily_effect", "data": summary}, "homophily_effect_results.json")

def run_structure_effect_experiment(seeds=[42, 43, 44, 45, 46]):
    print("\n Running Experiment 3: Structure Effect ")
    data_dir = "data/synthetic/structure_effect"
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    
    summary = []
    
    for f in tqdm(files, desc="Datasets"):
        dataset_name = os.path.basename(f)
        data = torch.load(f, weights_only=False)
        
        for model_name, model_cls in MODELS.items():
            print(f" > Tuning {model_name} for {dataset_name}...")
            best_params = tune_hyperparameters(model_cls, data, DEFAULT_PARAMS, n_trials=5)
            
            run_metrics = []
            for seed in seeds:
                res = run_single_experiment(model_cls, data, best_params, seed)
                if 'embeddings' in res: del res['embeddings']
                if 'labels' in res: del res['labels']
                run_metrics.append(res)
            
            avg_acc = np.mean([r['test_acc'] for r in run_metrics])
            std_acc = np.std([r['test_acc'] for r in run_metrics])
            
            entry = {
                'dataset': dataset_name,
                'model': model_name,
                'test_acc': avg_acc,
                'test_acc_std': std_acc,
                'tuned_params': best_params,
                'all_runs': run_metrics
            }
            summary.append(entry)
            
    save_results({"experiment": "structure_effect", "data": summary}, "structure_effect_results.json")

def run_depth_effect_experiment(seeds=[42, 43, 44, 45, 46]):
    print("\n Running Experiment 4: Depth (Oversmoothing) Effect ")
    # Use a medium sized SBM graph
    data_path = "data/synthetic/size_effect/sbm_1000.pt"
    if not os.path.exists(data_path):
        print("Dataset for depth effect not found. Skipping.")
        return
        
    data = torch.load(data_path, weights_only=False)
    depths = [2, 4, 8, 16, 32]
    
    summary = []
    
    oversmoothing_models = ['GCN', 'GAT', 'GIN']
    
    for model_name in oversmoothing_models:
        model_cls = MODELS[model_name]
        
        for d in depths:
            print(f" > Tuning {model_name} for depth {d}...")
            # We fix num_layers in the tuning base
            tune_base = DEFAULT_PARAMS.copy()
            tune_base['num_layers'] = d
            
            best_params = tune_hyperparameters(model_cls, data, tune_base, n_trials=5)
            
            run_metrics = []
            for seed in seeds:
                res = run_single_experiment(model_cls, data, best_params, seed)
                
                if d == 32 and seed == seeds[-1]:
                    # Save embeddings separately to avoid massive JSON
                    save_embeddings(res['embeddings'], res['labels'], f"embeddings_{model_name}_depth32.pt")
                
                if 'embeddings' in res: del res['embeddings']
                if 'labels' in res: del res['labels']
                    
                run_metrics.append(res)
            
            avg_acc = np.mean([r['test_acc'] for r in run_metrics])
            std_acc = np.std([r['test_acc'] for r in run_metrics])
            
            entry = {
                'depth': d,
                'model': model_name,
                'test_acc': avg_acc,
                'test_acc_std': std_acc,
                'tuned_params': best_params,
                'all_runs': run_metrics
            }
            summary.append(entry)
            
    save_results({"experiment": "depth_effect", "data": summary}, "depth_effect_results.json")

def save_embeddings(embeddings, labels, filename):
    torch.save({'embeddings': embeddings, 'labels': labels}, os.path.join(RESULTS_DIR, filename))
    print(f"Saved embeddings to {filename}")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    run_size_effect_experiment()
    run_homophily_effect_experiment()
    run_structure_effect_experiment()
    run_depth_effect_experiment()
