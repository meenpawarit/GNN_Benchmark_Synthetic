import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import numpy as np
import copy

def set_seed(seed):
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_device():
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train_epoch(model, data, optimizer, criterion):
    """Trains the model for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    
    # We only train on the training mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluates the model on a specific mask (val/test)."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return acc, prec, rec, f1

def run_single_experiment(model_cls, data, params, seed=42):
    """
    Runs a complete training session for a single model and dataset.
    
    Args:
        model_cls: Class of the model to instantiate (GCN, GAT, etc.)
        data: PyG Data object
        params: Dictionary of hyperparameters
        seed: Random seed
        
    Returns:
        results: Dict containing metrics
    """
    set_seed(seed)
    device = get_device()
    data = data.to(device)
    
    in_channels = data.num_features
    out_channels = len(torch.unique(data.y))
    
    model = model_cls(
        in_channels=in_channels,
        hidden_channels=params['hidden_channels'],
        out_channels=out_channels,
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    
    if not hasattr(data, 'train_mask'):
        # 60/20/20 split
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size+val_size]] = True
        data.test_mask[indices[train_size+val_size:]] = True
    
    # Training Loop
    best_val_acc = 0
    best_model_state = None
    patience = params['patience']
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(params['epochs']):
        loss = train_epoch(model, data, optimizer, criterion)
        val_acc, _, _, _ = evaluate(model, data, data.val_mask)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    train_time = time.time() - start_time
    
    # Final Evaluation with best model
    model.load_state_dict(best_model_state)
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, data, data.test_mask)
    
    # Get embeddings for visualization
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    
    return {
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1,
        'train_time': train_time,
        'epochs_run': epoch + 1,
        'embeddings': embeddings.cpu(), 
        'labels': data.y.cpu()
    }

def tune_hyperparameters(model_cls, data, base_params, n_trials=10, seed=42):
    """
    Performs random search for hyperparameter tuning.
    
    Args:
        model_cls: Model class to tune
        data: PyG Data object
        base_params: Dictionary of default parameters to start with
        n_trials: Number of random combinations to try
        seed: Random seed
        
    Returns:
        best_params: Dictionary of best hyperparameters found
    """
    print(f"  Tuning {model_cls.__name__} with {n_trials} trials...")
    set_seed(seed)
    device = get_device()
    
    
    data_tune = copy.deepcopy(data)
    
    # Define Search Space
    param_space = {
        'hidden_channels': [32, 64, 128],
        'dropout': [0.0, 0.3, 0.5, 0.7],
        'lr': [1e-2, 1e-3, 5e-4],
        'weight_decay': [0, 1e-4, 5e-4, 1e-3]
    }
    
    best_val_acc = -1
    best_params = base_params.copy()
    
    import random
    
    for trial in range(n_trials):
        # Sample parameters
        current_params = base_params.copy()
        current_params['hidden_channels'] = random.choice(param_space['hidden_channels'])
        current_params['dropout'] = random.choice(param_space['dropout'])
        current_params['lr'] = random.choice(param_space['lr'])
        current_params['weight_decay'] = random.choice(param_space['weight_decay'])
        
        # Fast training for tuning (fewer epochs)
        tune_params = current_params.copy()
        tune_params['epochs'] = 50 # Reduced epochs for tuning speed
        tune_params['patience'] = 10
        
        try:
             # We use a fixed seed for all tuning trials to compare params fairly on same split
            res = run_single_experiment(model_cls, data_tune, tune_params, seed=seed)
            
            acc = res['test_acc']
            
            if acc > best_val_acc:
                best_val_acc = acc
                best_params = current_params
        except Exception as e:
            print(f"    Trial failed: {e}")
            continue
            
    print(f"  Best params: hidden={best_params['hidden_channels']}, lr={best_params['lr']}, drop={best_params['dropout']}")
    return best_params
