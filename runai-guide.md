# RunAI Interactive Mode Guide

**Another summary opus wrote for me."

## Launching Interactive Session

```bash
# Start an interactive job with GPU
runai submit-interactive cs336-dev \
  --image ghcr.io/YOUR_USERNAME/cs336-assignment1-basics:latest \
  --gpu 1 \
  --interactive \
  --attach

# Or with more resources
runai submit-interactive cs336-dev \
  --image ghcr.io/YOUR_USERNAME/cs336-assignment1-basics:latest \
  --gpu 1 \
  --cpu 8 \
  --memory 32Gi \
  --interactive \
  --attach
```

## Once Inside the Container

Everything works exactly as it does locally:

```bash
# You'll be in /app with all dependencies installed

# Run training
uv run python -m cs336_basics.train

# Run sweep
uv run python -m cs336_basics.sweep \
  --cfg_cls configs:TSCfg \
  --lr 5e-5 \
  --bs 32 \
  --steps 100 \
  --print_every 1 \
  --device cuda

# Check GPU availability
uv run python -c "import torch; print(torch.cuda.is_available())"

# Run any Python script
uv run python -m cs336_basics.accounting

# Or use Python directly (since uv has set everything up)
python -c "import cs336_basics; print(cs336_basics.__version__)"
```

## Running Jupyter Notebooks

To run Jupyter in RunAI interactive mode:

```bash
# 1. Install Jupyter (if not in your dependencies)
uv pip install jupyter notebook

# 2. Start Jupyter with port forwarding
# Option A: If RunAI supports port forwarding
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Option B: Using RunAI's port forwarding (check your cluster docs)
runai submit-interactive cs336-jupyter \
  --image ghcr.io/YOUR_USERNAME/cs336-assignment1-basics:latest \
  --gpu 1 \
  --port 8888:8888 \
  --interactive \
  --attach

# Then inside:
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Alternative: VS Code Remote
Many RunAI clusters support VS Code remote development:
1. Install VS Code with Remote-SSH extension
2. Connect to your RunAI pod via SSH
3. Use VS Code's built-in Jupyter support

### Running the accounting notebook specifically:
```bash
# Convert notebook to script and run
uv run jupyter nbconvert --to script cs336_basics/accounting.ipynb
uv run python cs336_basics/accounting.py

# Or run cells programmatically
uv run python -c "
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
with open('cs336_basics/accounting.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {'metadata': {'path': 'cs336_basics/'}})
"
```

## Mounting Volumes

If you need to access data or save outputs:

```bash
runai submit-interactive cs336-dev \
  --image ghcr.io/YOUR_USERNAME/cs336-assignment1-basics:latest \
  --gpu 1 \
  --volume /path/on/cluster/data:/app/data \
  --volume /path/on/cluster/outputs:/app/outputs \
  --interactive \
  --attach
```

## Tips for RunAI

1. **Persistent Storage**: RunAI containers are ephemeral. Save important outputs to mounted volumes.

2. **Environment Variables**: Set them in the runai command:
   ```bash
   runai submit-interactive cs336-dev \
     --image ghcr.io/YOUR_USERNAME/cs336-assignment1-basics:latest \
     --gpu 1 \
     -e WANDB_API_KEY=your_key \
     -e CUDA_VISIBLE_DEVICES=0 \
     --interactive --attach
   ```

3. **Detaching/Reattaching**: 
   - Detach: `Ctrl+P` then `Ctrl+Q`
   - Reattach: `runai attach cs336-dev`

4. **Non-Interactive Jobs**: For long runs:
   ```bash
   runai submit cs336-train \
     --image ghcr.io/YOUR_USERNAME/cs336-assignment1-basics:latest \
     --gpu 1 \
     --command -- uv run python -m cs336_basics.train
   ```

## Debugging

```bash
# Check Python version
python --version  # Should show 3.11.x

# Check uv installation
uv --version

# List installed packages
uv pip list

# Check CUDA
nvidia-smi
``` 