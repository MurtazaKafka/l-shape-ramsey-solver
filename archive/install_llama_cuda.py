#!/usr/bin/env python3
"""
Script to install llama-cpp-python with CUDA support for the NVIDIA A6000 GPU.
"""
import os
import sys
import subprocess
import platform

def install_llama_cpp_with_cuda():
    """Install llama-cpp-python with CUDA support for A6000 GPU."""
    print("Installing llama-cpp-python with CUDA support for NVIDIA A6000...")
    
    # Set environment variables for CUDA optimization
    # Using GGML_CUDA instead of deprecated LLAMA_CUBLAS flag
    env_vars = {
        "CMAKE_ARGS": "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86",  # A6000 is Ampere architecture (compute capability 8.6)
        "FORCE_CMAKE": "1"
    }
    
    # Update environment with CUDA settings
    install_env = os.environ.copy()
    for key, value in env_vars.items():
        install_env[key] = value
        print(f"Setting {key}={value}")
    
    try:
        print("Removing existing llama-cpp-python installation...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"],
            check=True
        )
        
        print("Installing llama-cpp-python with CUDA support...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir", "--force-reinstall"],
            env=install_env,
            check=True
        )
        
        print("Installation complete. Testing CUDA support...")
        # Test if CUDA support was properly enabled
        test_code = """
import llama_cpp
print("llama-cpp-python version:", llama_cpp.__version__)
print("CUDA available in llama-cpp-python:", hasattr(llama_cpp.llama_cpp, "GGML_CUDA") or hasattr(llama_cpp.llama_cpp, "GGML_USE_CUDA"))
"""
        subprocess.run(
            [sys.executable, "-c", test_code],
            check=True
        )
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if install_llama_cpp_with_cuda():
        print("Successfully installed llama-cpp-python with CUDA support!")
    else:
        print("Failed to install llama-cpp-python with CUDA support.")
        sys.exit(1)