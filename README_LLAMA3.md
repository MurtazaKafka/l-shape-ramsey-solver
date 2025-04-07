# Using Llama 3.3 70B with FunSearch for L-shape Ramsey Problem

This README provides step-by-step instructions for converting the Meta Llama 3.3 70B model to a format compatible with the Hugging Face `transformers` library, and using it with the `llama_funsearch.py` script for solving the L-shape Ramsey problem.

## Problem: Meta Format vs. Hugging Face Format

The model files in `/home/DAVIDSON/munikzad/.llama/checkpoints/Llama3.3-70B-Instruct` are in Meta's original format (with `consolidated.*.pth` files), not in the Hugging Face format (which requires a `config.json` file). 

The `transformers` library expects the Hugging Face format, which is why you're encountering errors like:
```
Unrecognized model in .. Should have a `model_type` key in its config.json
```

## Solution: Convert the Model

We've provided a script to convert the model from Meta format to Hugging Face format:

```bash
# 1. Install required dependencies
python convert_meta_to_hf.py --skip_deps  # Remove --skip_deps flag to auto-install dependencies

# 2. Run the conversion (will take some time due to the model size)
python convert_meta_to_hf.py --input_dir "/home/DAVIDSON/munikzad/.llama/checkpoints/Llama3.3-70B-Instruct" --output_dir "./llama3_hf"
```

The conversion process will:
1. Install necessary dependencies (unless `--skip_deps` is used)
2. Download the conversion script from Hugging Face if needed
3. Convert the model files to Hugging Face format
4. Save the converted model in the specified output directory

## After Conversion: Update the Scripts

After converting the model, you need to update the `llama_funsearch.py` script to use the converted model:

```bash
# Update the llama_funsearch.py script to use the converted model
python update_model_path.py --file llama_funsearch.py --model_path "./llama3_hf"
```

## Running FunSearch

Now you can run the FunSearch algorithm with the converted model:

```bash
# Run FunSearch for a 3×3 grid with 3 colors
python llama_funsearch.py --grid_size 3 --colors 3
```

## Troubleshooting

### Insufficient Disk Space

Converting the 70B model requires significant disk space (approximately 140GB). If you encounter disk space issues, consider:
- Cleaning up unnecessary files
- Using a different directory with more space
- Converting a smaller model if available

### Memory Errors During Conversion

If you encounter memory errors during conversion, try:
- Using a machine with more RAM
- Converting on a GPU server if possible
- Reducing the model size (if possible)

### Missing Dependencies

If you encounter errors about missing dependencies, install them manually:

```bash
pip install torch transformers sentencepiece blobfile protobuf accelerate
```

### Model Loading Errors

If you still encounter model loading errors after conversion:
1. Check that the conversion completed successfully (verify `config.json` exists in the output directory)
2. Verify that the model path in `llama_funsearch.py` is correctly updated
3. Try running with `--load_in_8bit` or `--load_in_4bit` flags for memory efficiency

```bash
python llama_funsearch.py --model_path "./llama3_hf" --load_in_8bit
```

## Additional Resources

- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/index
- Meta Llama 3 documentation: https://ai.meta.com/llama/
- L-shape Ramsey problem research: [Include relevant links] 