
try:
    import llama_cpp
    print(f"llama-cpp-python version: {llama_cpp.__version__}")
    
    # Check if CUDA is available in llama-cpp-python
    has_cuda = False
    for attr in dir(llama_cpp.llama_cpp):
        if "CUDA" in attr or "cuda" in attr or "GPU" in attr:
            has_cuda = True
            print(f"Found CUDA indicator: {attr}")
    
    if has_cuda:
        print("✅ llama-cpp-python has CUDA support")
    else:
        print("⚠️ No CUDA support found in llama-cpp-python")
except ImportError:
    print("❌ llama-cpp-python not installed")
