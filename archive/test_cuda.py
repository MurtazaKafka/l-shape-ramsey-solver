
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
