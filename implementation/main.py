#!/usr/bin/env python3
"""
Main script to run FunSearch for the L-shape Ramsey problem.

This script orchestrates the entire FunSearch workflow:
1. Run FunSearch on grid sizes from 3x3 to 10x10
2. Analyze the results and generate visualizations
3. Create comparative analysis of the solutions

Usage:
    python main.py [--quick]

Options:
    --quick: Run a quicker version with fewer iterations for testing
"""
import os
import sys
import time
from datetime import datetime
import argparse

# Make sure we're in the implementation directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def print_header(message):
    """Print a header message with decoration."""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80)

def main():
    """Run the entire FunSearch workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run FunSearch for L-shape Ramsey problem")
    parser.add_argument("--quick", action="store_true", help="Run a quick version with fewer iterations")
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a logfile
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/funsearch_run_{timestamp}.log"
    
    # Set up logging to both console and file
    def log_message(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    
    log_message(f"Starting FunSearch for L-shape Ramsey problem at {datetime.now()}")
    log_message(f"Quick mode: {args.quick}")
    
    # 1. Run FunSearch
    print_header("STEP 1: Running FunSearch")
    
    if args.quick:
        # Modify the configuration file for a quick run
        log_message("Setting up quick configuration...")
        from run_l_shape_funsearch import FunSearchConfig
        
        # Override configuration
        FunSearchConfig.GRID_SIZES = [3, 4, 5, 6]  # Smaller grid sizes
        FunSearchConfig.ITERATIONS_PER_SIZE = 50   # Fewer iterations
        FunSearchConfig.POPULATION_SIZE = 20       # Smaller population
        FunSearchConfig.TIME_LIMIT = 10 * 60       # 10 minutes per grid size
        
        log_message("Running quick FunSearch...")
    else:
        log_message("Running full FunSearch (this may take several hours)...")
    
    try:
        # Import and run the FunSearch script
        from run_l_shape_funsearch import run_full_funsearch
        run_full_funsearch()
        log_message("FunSearch completed successfully.")
    except Exception as e:
        log_message(f"Error during FunSearch: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        return
    
    # 2. Analyze Results
    print_header("STEP 2: Analyzing Results")
    log_message("Running analysis of FunSearch results...")
    
    try:
        # Import and run the analysis script
        from analyze_funsearch_results import analyze_results, generate_multiple_grid_comparison
        
        # Run analysis
        analyze_results()
        
        # Generate combined visualization
        generate_multiple_grid_comparison()
        
        log_message("Analysis completed successfully.")
    except Exception as e:
        log_message(f"Error during analysis: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
    
    # 3. Summarize findings
    print_header("STEP 3: Summary of Findings")
    
    # Try to load analysis summary
    analysis_files = [f for f in os.listdir("analysis") if f.startswith("analysis_summary_")]
    if analysis_files:
        latest_analysis = sorted(analysis_files)[-1]  # Get the most recent
        try:
            with open(f"analysis/{latest_analysis}", "r") as f:
                summary = f.read()
            
            log_message("Summary of findings:")
            log_message(summary)
        except Exception as e:
            log_message(f"Error reading summary: {str(e)}")
    else:
        log_message("No analysis summary found.")
    
    # Finish timing
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    log_message(f"\nTotal runtime: {hours}h {minutes}m {seconds}s")
    log_message(f"Run completed at: {datetime.now()}")
    log_message(f"Log file: {log_file}")
    
    print_header("COMPLETED")
    print(f"  Total runtime: {hours}h {minutes}m {seconds}s")
    print(f"  Results and visualizations are in the 'analysis' directory")
    print(f"  Log file: {log_file}")
    print("="*80)

if __name__ == "__main__":
    main() 