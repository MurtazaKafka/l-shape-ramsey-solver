#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import time
import signal

def get_gpu_processes():
    """Get information about processes using the GPU."""
    try:
        # Run nvidia-smi to get process information
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            universal_newlines=True
        )
        
        # Parse output
        processes = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split(", ")
            if len(parts) >= 3:
                pid = int(parts[0])
                name = parts[1]
                memory = parts[2]
                processes.append({
                    'pid': pid,
                    'name': name,
                    'memory': memory
                })
        
        return processes
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []

def get_gpu_memory_info():
    """Get information about GPU memory usage."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
            universal_newlines=True
        )
        
        gpus = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'total_memory': parts[2],
                    'used_memory': parts[3],
                    'free_memory': parts[4]
                })
        
        return gpus
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return []

def kill_process(pid, force=False):
    """Kill a process by PID."""
    try:
        if force:
            os.kill(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False

def clear_gpu_cache():
    """Try to clear GPU memory cache if using PyTorch."""
    try:
        import torch
        torch.cuda.empty_cache()
        print("Cleared PyTorch CUDA cache")
    except ImportError:
        print("PyTorch not available, skipping cache clearing")
    except Exception as e:
        print(f"Error clearing CUDA cache: {e}")

def main():
    parser = argparse.ArgumentParser(description="GPU Process Management Utility")
    parser.add_argument("--list", action="store_true", help="List processes using the GPU")
    parser.add_argument("--info", action="store_true", help="Show GPU memory information")
    parser.add_argument("--kill", type=int, metavar="PID", help="Kill a specific process by PID")
    parser.add_argument("--kill-all", action="store_true", help="Kill all processes using the GPU")
    parser.add_argument("--exclude", type=int, nargs="+", metavar="PID", help="PIDs to exclude when using --kill-all")
    parser.add_argument("--force", action="store_true", help="Use SIGKILL instead of SIGTERM")
    parser.add_argument("--clear-cache", action="store_true", help="Clear GPU memory cache")
    parser.add_argument("--min-free", type=int, metavar="MB", help="Ensure at least this much free memory (in MB)")
    
    args = parser.parse_args()
    
    # Default action if none specified
    if not (args.list or args.info or args.kill or args.kill_all or args.clear_cache or args.min_free):
        args.info = True
        args.list = True
    
    # Clear cache if requested
    if args.clear_cache:
        clear_gpu_cache()
    
    # Get GPU memory info
    if args.info:
        gpus = get_gpu_memory_info()
        if gpus:
            print("\nGPU Memory Information:")
            for gpu in gpus:
                print(f"GPU {gpu['index']} ({gpu['name']}):")
                print(f"  Total memory: {gpu['total_memory']}")
                print(f"  Used memory:  {gpu['used_memory']}")
                print(f"  Free memory:  {gpu['free_memory']}")
        else:
            print("No GPU information available")
    
    # Get processes
    processes = get_gpu_processes()
    
    # List processes
    if args.list:
        if processes:
            print("\nProcesses using the GPU:")
            print(f"{'PID':<8} {'Memory':<12} {'Name'}")
            print("-" * 50)
            for proc in processes:
                print(f"{proc['pid']:<8} {proc['memory']:<12} {proc['name']}")
        else:
            print("No processes are currently using the GPU")
    
    # Kill a specific process
    if args.kill is not None:
        pid = args.kill
        if kill_process(pid, args.force):
            print(f"Successfully killed process {pid}")
        else:
            print(f"Failed to kill process {pid}")
    
    # Kill all processes
    if args.kill_all:
        exclude = args.exclude or []
        killed = 0
        for proc in processes:
            pid = proc['pid']
            if pid not in exclude:
                if kill_process(pid, args.force):
                    print(f"Killed process {pid} ({proc['name']})")
                    killed += 1
                else:
                    print(f"Failed to kill process {pid} ({proc['name']})")
        
        if killed:
            print(f"Killed {killed} processes")
        else:
            print("No processes were killed")
    
    # Ensure minimum free memory
    if args.min_free is not None:
        gpus = get_gpu_memory_info()
        if not gpus:
            print("Cannot check GPU memory, information not available")
            return
        
        for gpu in gpus:
            # Extract numeric value from free memory string (remove " MiB" or similar)
            free_mem = int("".join(c for c in gpu['free_memory'] if c.isdigit()))
            if free_mem < args.min_free:
                print(f"GPU {gpu['index']} has only {free_mem} MB free, less than requested {args.min_free} MB")
                if not processes:
                    processes = get_gpu_processes()
                
                # Try to free up memory by killing processes
                exclude = args.exclude or []
                for proc in processes:
                    pid = proc['pid']
                    if pid not in exclude:
                        print(f"Attempting to kill process {pid} ({proc['name']}) to free memory...")
                        if kill_process(pid, args.force):
                            print(f"Killed process {pid}")
                        time.sleep(1)  # Wait a bit for memory to be released
                
                # Clear cache again
                clear_gpu_cache()
                
                # Check if we have enough memory now
                new_gpus = get_gpu_memory_info()
                for new_gpu in new_gpus:
                    if new_gpu['index'] == gpu['index']:
                        new_free_mem = int("".join(c for c in new_gpu['free_memory'] if c.isdigit()))
                        print(f"GPU {new_gpu['index']} now has {new_free_mem} MB free memory")
                        break

if __name__ == "__main__":
    main()
