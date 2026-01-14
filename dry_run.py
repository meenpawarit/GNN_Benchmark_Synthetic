import torch
from run_experiments import run_size_effect_experiment, run_depth_effect_experiment, DEFAULT_PARAMS

# Override params for speed
DEFAULT_PARAMS['epochs'] = 1
DEFAULT_PARAMS['patience'] = 1

print("Starting Dry Run...")
try:
    # Run just one experiment with 1 seed
    run_size_effect_experiment(seeds=[42])
    
    # Run depth experiment to verify embeddings generation
    # This runs 3 models * 5 depths = 15 runs (fast with 1 epoch)
    run_depth_effect_experiment(seeds=[42])
    
    print("Dry Run Successful!")
except Exception as e:
    print(f"Dry Run Failed: {e}")
    raise e
