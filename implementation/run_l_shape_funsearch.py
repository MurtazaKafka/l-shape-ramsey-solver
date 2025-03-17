"""Run FunSearch for the L-shape Ramsey Grid problem."""
import os
import sys
import torch
from funsearch import main as funsearch_main
import config

def get_device():
    """Get the appropriate device for PyTorch operations"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def run_funsearch():
    # Ensure we're in the implementation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Print device information
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        print(f"Current GPU Memory: {torch.mps.current_allocated_memory() / (1024**3):.1f}GB")
    elif device.type == "cuda":
        print(f"Current GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.1f}GB")
    
    # Read the specification file
    with open("specification_l_shape_ramsey.txt", "r") as f:
        specification = f.read()
    
    # Create configuration
    cfg = config.Config(
        programs_database=config.ProgramsDatabaseConfig(
            functions_per_prompt=2,
            num_islands=5,  # Reduced for local testing
            reset_period=3600,  # 1 hour
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30000
        ),
        num_evaluators=1,  # Reduced for local testing
        num_samplers=1,    # Reduced for local testing
        samples_per_prompt=1
    )
    
    # Define input sizes to try
    inputs = [(4, 0), (5, 0), (6, 0)]  # (size, initial_seed)
    
    # Run FunSearch
    funsearch_main(specification, inputs, cfg)

if __name__ == "__main__":
    run_funsearch() 