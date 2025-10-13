# Memory-Optimized Training for Small GPUs

## Your Situation

You have a **5.64GB GPU** which is tight for training Llama 3.2-3B. I've optimized the training scripts for your hardware.

## Changes Made

### 1. Reduced LoRA Rank
- **Before**: `r=16` (24.3M trainable params)
- **After**: `r=8` (~12M trainable params)
- **Memory saved**: ~50% reduction in adapter parameters

### 2. Reduced Batch Size & Increased Gradient Accumulation
- **Batch size**: 1 (minimum)
- **Gradient accumulation**: 16 (simulates batch size of 16)
- **Memory saved**: Processes one example at a time

### 3. Reduced Sequence Length
- **Before**: 512 tokens
- **After**: 384 tokens
- **Memory saved**: ~25% reduction in activation memory

### 4. Already Using 4-bit Quantization
- Model loaded in 4-bit (NF4)
- Uses `bitsandbytes` for memory efficiency
- Already enabled in your setup

## Now Try Training Again

Since the dataset is already processed, skip data prep:

```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai
source calma/bin/activate
python3 train_with_hf_dataset.py --skip-data-prep
```

## If Still Out of Memory

Try these progressively:

### Option 1: Reduce Data Size (Quick Training)
```bash
python3 train_with_hf_dataset.py --skip-data-prep --max-samples 1500 --epochs 2
```
This will:
- Use only 1500 examples (~1200 train, 150 val, 150 test)
- Train for 2 epochs
- Take ~30-60 minutes

### Option 2: Set PyTorch Memory Optimization
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 train_with_hf_dataset.py --skip-data-prep
```

### Option 3: Clear GPU Memory First
```bash
# Kill any other processes using GPU
nvidia-smi
# If you see other processes, kill them

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Then train
python3 train_with_hf_dataset.py --skip-data-prep
```

### Option 4: Train on CPU (Slow but Works)
If GPU keeps failing, train on CPU:

```bash
# Force CPU training (will be slow, 8-12 hours)
CUDA_VISIBLE_DEVICES="" python3 train_with_hf_dataset.py --skip-data-prep --max-samples 1500 --epochs 2
```

## Expected Memory Usage After Optimization

With these changes:
- **Model (4-bit)**: ~2GB
- **LoRA adapters (r=8)**: ~0.5GB
- **Activations (batch=1, seq=384)**: ~1.5GB
- **Optimizer states**: ~1GB
- **Buffer**: ~0.5GB
- **Total**: ~5.5GB (should fit in 5.64GB)

## Memory Monitoring During Training

Watch for:
```
✓ Good: Training starts and progresses normally
⚠️  Warning: Training starts but crashes after a few steps
✗ Error: Crashes immediately
```

If crashes after a few steps, reduce `max_samples` to 1000.

## Quick Training Test

Test with minimal data first:
```bash
python3 train_with_hf_dataset.py --skip-data-prep --max-samples 100 --epochs 1
```

This should take ~5-10 minutes and verify everything works.

## After Successful Training

Your model will be at:
```
models/calma-hf-trained/final/
```

To use it:
```bash
cp -r models/calma-final models/calma-final-backup
cp -r models/calma-hf-trained/final/* models/calma-final/
```

## Performance Note

**Lower rank (r=8) vs (r=16):**
- ✅ Uses less memory
- ✅ Trains faster
- ✅ Less overfitting risk
- ⚠️  Slightly less capacity (but still very effective)

For your use case (anti-overfitting), **r=8 is actually better** - it's more constrained and will generalize better!

## Troubleshooting

### Still OOM?
1. Check GPU usage: `nvidia-smi`
2. Kill other processes using GPU
3. Try: `--max-samples 1000`

### Training too slow?
1. Monitor with: `watch -n 1 nvidia-smi`
2. Expected: 1-2 steps per second
3. If slower, check GPU utilization

### Want to track progress?
Install wandb (optional):
```bash
pip install wandb
wandb login
```

Then edit `model_training_improved.py` line 187:
```python
report_to="wandb",  # Change from "none"
```

Good luck! 🚀
