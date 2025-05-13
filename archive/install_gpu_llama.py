#!/usr/bin/env python3
"""
Script to install llama-cpp-python with proper CUDA support for NVIDIA A6000 GPU.
This version uses modern commands and properly configures CMake.
"""
import os
import sys
import subprocess
import platform

def run_command(cmd, env=None, check=True):
    """Run a command and return its output."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            env=env or os.environ.copy(),
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def install_cuda_llama():
    """Install llama-cpp-python with CUDA support properly configured."""
    print("Installing llama-cpp-python with CUDA support for NVIDIA A6000...")
    
    # Load CUDA module if available (common on HPC/university clusters)
    try:
        run_command(["module", "load", "cuda"], check=False)
    except:
        print("Module command not available or CUDA module not found (this is normal on some systems)")
    
    # Try to find nvcc to verify CUDA installation
    try:
        nvcc_result = run_command(["which", "nvcc"], check=False)
        if nvcc_result.returncode == 0:
            nvcc_path = nvcc_result.stdout.strip()
            print(f"Found NVCC at: {nvcc_path}")
            # Get CUDA version
            nvcc_version = run_command([nvcc_path, "--version"], check=False)
            print(f"NVCC version: {nvcc_version.stdout}")
        else:
            print("Warning: NVCC not found in PATH, CUDA may not be properly installed")
    except:
        print("Could not check for NVCC")
    
    # Set environment variables for CUDA build
    build_env = os.environ.copy()
    
    # A6000 is Ampere architecture (GA102) with compute capability 8.6
    build_env["CMAKE_ARGS"] = "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86"
    build_env["FORCE_CMAKE"] = "1"
    build_env["VERBOSE"] = "1"  # Make compilation verbose for debugging
    
    # Make sure pip is up to date
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install build dependencies
    run_command([
        sys.executable, "-m", "pip", "install",
        "setuptools>=49.6.0",
        "wheel>=0.38.1",
        "scikit-build>=0.13.1",
        "cmake>=3.21",
        "ninja"
    ])
    
    # Remove any existing installations
    run_command([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "llama-cpp-python", "llama_cpp_python"
    ], check=False)
    
    # Install with force reinstall
    result = run_command([
        sys.executable, "-m", "pip", "install", 
        "llama-cpp-python==0.2.65",  # Use a specific version known to work well with CUDA
        "--force-reinstall",
        "--no-cache-dir",
        "--verbose"
    ], env=build_env, check=False)
    
    # Test if CUDA support is enabled
    print("\nTesting CUDA support in llama-cpp-python:")
    test_script = """
import llama_cpp
print(f"llama-cpp-python version: {llama_cpp.__version__}")

# Check if CUDA is available
cuda_available = False
for attr in dir(llama_cpp.llama_cpp):
    if "CUDA" in attr or "cuda" in attr:
        cuda_available = True
        print(f"Found CUDA indicator: {attr}")

if cuda_available:
    print("✅ CUDA support is available!")
else:
    print("❌ CUDA support is NOT available")
    
# Try to create an instance with GPU layers
try:
    model = llama_cpp.Llama(
        model_path="./models/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=-1,
        verbose=True
    )
    print("✅ Successfully initialized model with GPU support")
except Exception as e:
    print(f"❌ Error initializing model with GPU: {e}")
"""
    
    with open("test_cuda.py", "w") as f:
        f.write(test_script)
    
    run_command([sys.executable, "test_cuda.py"], check=False)
    
    print("\nInstallation completed. If CUDA support is not available, check error messages above.")

if __name__ == "__main__":
    install_cuda_llama()