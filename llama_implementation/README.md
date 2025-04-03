# L-shape Ramsey Problem Solver using Llama 3.2 8B

This implementation uses the new Llama 3.2 8B language model to solve the L-shape Ramsey problem through function generation and verification.

## About Llama 3.2

Llama 3.2 was released by Meta on September 25, 2024, featuring:
- Lightweight models (1B and 3B) for edge and mobile devices
- Vision-capable models (11B and 90B)
- Improved performance over previous Llama models
- Enhanced context length of 128K tokens for small models

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download the Llama 3.2 model:
```bash
python download_llama.py
```

Note: You need to have access to the Llama 3.2 model on Hugging Face. You may need to:
1. Create a Hugging Face account
2. Request access to Llama 3.2 at https://huggingface.co/meta-llama/Llama-3.2-8B-Instruct
3. Set up your Hugging Face token:
```bash
huggingface-cli login
```

## Usage

Run the solver:
```bash
python l_shape_llama.py
```

The solver will attempt to find valid solutions for grid sizes 3×3, 4×4, 5×5, and 6×6, with visualizations saved to the `visualizations` directory.

## Implementation Details

The solver works by:
1. Using Llama 3.2 to generate Python functions that create grid colorings
2. Verifying the generated solutions for valid L-shape avoidance 
3. Using multiple attempts to find valid solutions
4. Supporting GPU acceleration when available
5. Visualizing successful solutions with matplotlib

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 20GB+ disk space for model storage

## License

This implementation is part of the larger L-shape Ramsey problem project. See the main LICENSE file for details. 