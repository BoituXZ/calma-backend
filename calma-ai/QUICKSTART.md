# Quick Start - Retrain Calma Model (Anti-Overfitting)

This guide will help you retrain your Calma model using the Hugging Face mental health counseling dataset with anti-overfitting measures.

## 🚀 Quick Start (3 commands)

```bash
# 1. Setup environment and install dependencies
./setup_training.sh

# 2. Login to Hugging Face (first time only)
source calma/bin/activate
huggingface-cli login

# 3. Start training
python3 train_with_hf_dataset.py
```

That's it! The training will run for ~2-4 hours depending on your hardware.

---

## 📋 Detailed Steps

### Step 1: Environment Setup

```bash
cd /path/to/calma-backend/calma-ai
./setup_training.sh
```

This script will:
- Activate or create a virtual environment
- Install all required packages
- Run tests to verify setup

**What you'll need:**
- Python 3.8+
- 16GB RAM minimum (32GB recommended)
- GPU with 8GB+ VRAM (optional but recommended)
- ~20GB free disk space

### Step 2: Hugging Face Authentication

```bash
source calma/bin/activate
huggingface-cli login
```

You'll need:
1. A Hugging Face account (free at https://huggingface.co)
2. An access token (get one at https://huggingface.co/settings/tokens)

Paste your token when prompted.

### Step 3: Test Run (Recommended)

Before full training, test with a small sample:

```bash
python3 train_with_hf_dataset.py --max-samples 100 --epochs 1
```

This takes ~5-10 minutes and verifies everything works.

### Step 4: Full Training

```bash
python3 train_with_hf_dataset.py
```

**What happens:**
1. Downloads dataset (~15-30 min, one-time)
2. Processes and filters data (~10-20 min)
3. Trains model (~2-4 hours)
4. Evaluates on test set
5. Saves model to `models/calma-hf-trained/final`

**Expected output:**
```
Training examples: ~8000
Validation examples: ~1000
Test examples: ~1000

Training... (this will take a while)

✓ Training completed!
Validation Loss: 1.234
Test Loss: 1.245
Loss Difference: 0.011  ← Good! (should be < 0.1)
```

---

## 🎛️ Common Options

### Quick Test
```bash
python3 train_with_hf_dataset.py --max-samples 1000 --epochs 2
```

### More Aggressive Anti-Overfitting
```bash
python3 train_with_hf_dataset.py \
  --weight-decay 0.02 \
  --lora-dropout 0.2 \
  --epochs 2
```

### CPU Training (slower)
```bash
python3 train_with_hf_dataset.py --max-samples 2000 --epochs 2
```
(Automatically uses CPU if no GPU available)

### Custom Output
```bash
python3 train_with_hf_dataset.py --output-dir ./models/my-model
```

---

## 📊 Monitoring Training

Watch the console output for these metrics:

### Good Signs ✓
```
Training Loss: 2.5 → 1.8 → 1.4 → 1.2
Validation Loss: 2.4 → 1.9 → 1.5 → 1.3
```
Both decreasing smoothly, small gap between them.

### Warning Signs ⚠️
```
Training Loss: 2.5 → 1.2 → 0.8 → 0.3
Validation Loss: 2.4 → 1.8 → 2.1 → 2.5
```
Training loss decreasing but validation increasing = overfitting!

**If you see overfitting:**
- Press Ctrl+C to stop
- Rerun with: `--weight-decay 0.02 --epochs 2`

---

## 🎯 After Training

### 1. Check Performance

The script automatically evaluates on test set. Look for:
- **Loss difference < 0.1**: Great generalization ✓
- **Loss difference > 0.2**: Still overfitting ⚠️

### 2. Update Your Model

Your trained model is at: `models/calma-hf-trained/final`

**To use it in your two-model system:**

```bash
# Backup old model
cp -r models/calma-final models/calma-final-backup

# Replace with new model
cp -r models/calma-hf-trained/final/* models/calma-final/
```

### 3. Test the Model

```bash
# Test with evaluation script
python3 src/model_evaluation_cpu.py

# Or start the inference server
python3 src/inference.py
```

### 4. Integrate with NestJS Backend

Your backend is already configured for the two-model system:
- **Casual Model**: Base Llama (early conversation)
- **Therapeutic Model**: Your new fine-tuned model (mental health support)

Just restart your backend:
```bash
cd ../  # Back to calma-backend
npm run start:dev
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
source calma/bin/activate
pip install -r requirements-training.txt
```

### "CUDA out of memory"
```bash
# Use smaller batch size or limit data
python3 train_with_hf_dataset.py --max-samples 5000
```

Or edit `src/model_training_improved.py` line 102:
```python
batch_size = 1  # Change from 2 to 1
```

### "Dataset loading failed"
```bash
# Make sure you're logged in
huggingface-cli login

# Try loading manually
python3 -c "from datasets import load_dataset; ds = load_dataset('Amod/mental_health_counseling_conversations')"
```

### "Model still overfitting"
Try progressive training:
```bash
# Start small
python3 train_with_hf_dataset.py --max-samples 1000

# If good, increase
python3 train_with_hf_dataset.py --max-samples 5000

# If still good, use all data
python3 train_with_hf_dataset.py
```

### Need help?
Run the test script:
```bash
python3 test_setup.py
```

---

## 📁 Files Overview

- **`train_with_hf_dataset.py`** - Main training script (run this!)
- **`setup_training.sh`** - One-command setup
- **`test_setup.py`** - Verify your setup
- **`src/data_processing_hf.py`** - Dataset processor
- **`src/model_training_improved.py`** - Training with anti-overfitting
- **`RETRAINING_GUIDE.md`** - Detailed documentation
- **`requirements-training.txt`** - Python dependencies

---

## ⏱️ Time Estimates

| Task | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Setup | 5-10 min | 5-10 min |
| Dataset download | 15-30 min | 15-30 min |
| Processing | 10-20 min | 10-20 min |
| Training (100 samples) | 5 min | 15 min |
| Training (1000 samples) | 30 min | 2 hours |
| Training (full dataset) | 2-3 hours | 8-12 hours |

---

## 💡 Pro Tips

1. **Start small**: Test with `--max-samples 100` first
2. **Monitor closely**: Watch validation vs training loss
3. **Use early stopping**: Built in! Training stops if no improvement
4. **Save disk space**: Old checkpoints auto-deleted (keeps best 3)
5. **GPU recommended**: 5-10x faster than CPU
6. **Backup first**: Keep your old model just in case

---

## 🎓 Understanding the Fix

**Why was your model overfitting?**
- Small dataset (248KB) with aggressive augmentation
- Model memorized training examples
- Poor generalization to new conversations

**How does this fix it?**
1. **Larger dataset**: 10,000+ diverse counseling conversations
2. **Proper splits**: 80% train / 10% validation / 10% test
3. **Quality filtering**: Removes low-quality examples
4. **Better regularization**: Weight decay, dropout, early stopping
5. **No aggressive augmentation**: Uses real data, not synthetic
6. **Dynamic padding**: Prevents overfitting on padding tokens

**Result**: Model learns general patterns, not specific examples.

---

## 📚 Further Reading

- [RETRAINING_GUIDE.md](RETRAINING_GUIDE.md) - Comprehensive guide
- [TWO_MODEL_SYSTEM.md](../TWO_MODEL_SYSTEM.md) - Architecture overview
- [CLAUDE.md](CLAUDE.md) - Project documentation

---

Need help? Check the detailed guide or open an issue! 🚀
