# LLM Adversarial Testing Framework

This repository contains two approaches for running adversarial testing on Language Models using the GCG (Greedy Coordinate Gradient) algorithm:

1. Jupyter Notebook (`notebook.ipynb`)
2. Runbook with RUNME integration (`adversarial-testing.md`)

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- Hugging Face account with API token
- RUNME CLI (for runbook execution)

## Environment Setup

```bash
# Install dependencies
pip install --upgrade pip
pip install transformers==4.34.0 huggingface-hub==0.19.4 accelerate==0.24.1
pip install fastchat==0.2.31 bitsandbytes==0.41.1 livelossplot==0.5.5
pip install matplotlib numpy ipython optimum

# Clone required repository
git clone https://github.com/rabbidave/ZeroDay.Tools.git
cd ZeroDay.Tools && pip install -e .
```

## Usage Options

### 1. Jupyter Notebook

1. Open `notebook.ipynb` in Jupyter
2. Replace `"your_token_here"` with your Hugging Face API token
3. Execute cells sequentially
4. View results in generated plots and logs

### 2. RUNME Runbook

1. Install RUNME:
   ```bash
   brew install runme  # or: npx runme
   ```
2. Execute runbook:
   ```bash
   runme run adversarial-testing.md
   ```
3. Monitor progress in terminal

## Configuration

Key parameters in both implementations:

- `MODEL_ID`: Target model identifier
- `NUM_STEPS`: Number of optimization iterations
- `BATCH_SIZE`: Batch size for candidate evaluation
- `TOPK`: Number of top candidates per position

## Output

The framework generates:
- Loss progression plots
- Attack success logs
- Optimized adversarial strings

## Security Note

This framework is intended for security research and testing model robustness. Use responsibly and in accordance with applicable laws and ethical guidelines.
