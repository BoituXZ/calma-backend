# Training Guide - tcabanski Quality Dataset

## Overview

This guide explains how to train your Calma model using the **tcabanski/mental_health_counseling_responses** dataset - a high-quality, professionally-rated counseling dataset with 26k+ conversations.

## Why tcabanski Dataset?

### Previous Problem (Amod dataset)
- ❌ Only 3,512 examples (too small for Llama 3B)
- ❌ No quality ratings
- ❌ Led to incoherent, random responses
- ❌ Model couldn't learn proper conversation patterns

### Solution (tcabanski dataset)
- ✅ 26,140 total examples
- ✅ Quality-rated (empathy, appropriateness, relevance scores 1-5)
- ✅ Professional therapist-style responses
- ✅ Natural Q&A format (questionText → answerText)
- ✅ After filtering (scores ≥ 4): **~20,000 high-quality examples**

## Dataset Features

Each example includes:
- **questionTitle**: Brief topic
- **questionText**: Full user question
- **answerText**: Professional counselor response
- **empathy**: 1-5 rating
- **appropriateness**: 1-5 rating
- **relevance**: 1-5 rating

**Quality Filtering**: We only use examples where all three scores ≥ 4/5.

## Quick Start

### Prerequisites

1. **Virtual environment activated**:
```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai
source calma/bin/activate
```

2. **Hugging Face login** (one-time):
```bash
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

3. **GPU available** (recommended):
- Your 5.64GB RTX 4050 is perfect
- CPU training possible but ~5x slower

### Step 1: Process Data (~10-15 minutes)

```bash
python3 src/data_processing_tcabanski.py
```

**What this does**:
- Downloads tcabanski dataset from Hugging Face
- Filters for high-quality responses (empathy/appropriateness/relevance ≥ 4)
- Injects Zimbabwe cultural context
- Formats into Llama chat template
- Creates 80/10/10 train/validation/test splits
- Tokenizes and saves to `data/processed/tcabanski_mental_health/`

**Expected output**:
```
Loaded 26,140 total examples
After quality filtering: ~20,000 examples
Training:   16,000 examples
Validation:  2,000 examples
Test:        2,000 examples
```

### Step 2: Train Model (~2-3 hours)

```bash
python3 train_tcabanski.py
```

**What this does**:
- Loads base Llama 3.2-3B (fresh start, NO previous training)
- Applies LoRA adapters (rank=8 for memory efficiency)
- Trains with anti-overfitting measures
- Saves model to `models/calma-tcabanski-final/`

**Training parameters**:
- Epochs: 3 (with early stopping)
- Learning rate: 5e-5 (conservative)
- LoRA rank: 8 (memory-optimized)
- Batch size: 1 (gradient accumulation: 16)
- Max tokens: 384 (fits GPU comfortably)
- Early stopping: patience=5

**What to watch for**:
```
Epoch 1: train_loss=1.8, val_loss=1.7  ← Good
Epoch 2: train_loss=1.4, val_loss=1.5  ← Good
Epoch 3: train_loss=1.2, val_loss=1.3  ← Excellent

Final:
  Validation Loss: 1.30
  Test Loss: 1.32
  Loss Difference: 0.02  ← Great! (< 0.1 means good generalization)
```

### Step 3: Deploy New Model

**Update config**:
Edit `calma-ai-service/app/config.py` line 19:
```python
model_path: str = "../models/calma-tcabanski-final/final"
```

**Restart service**:
```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend
./start-calma.sh
```

**Test conversation**:
```
User: "I'm having relationship problems. My partner said he doesn't trust me."
Bot: [Should give relevant, empathetic relationship advice - NOT deflect to random topics]
```

## Advanced Options

### Quick Test Run

Train on small subset to verify everything works:
```bash
# Edit src/data_processing_tcabanski.py line 243:
max_samples=1000  # Instead of None

# Then train:
python3 train_tcabanski.py --epochs 2
```

Takes ~30 minutes, good for testing pipeline.

### Adjust Training Parameters

**If still some overfitting** (loss difference > 0.1):
```bash
python3 train_tcabanski.py \
  --weight-decay 0.02 \
  --lora-dropout 0.15 \
  --epochs 2
```

**For faster training** (less conservative):
```bash
python3 train_tcabanski.py \
  --learning-rate 1e-4 \
  --epochs 2
```

**Just evaluate existing model** (no training):
```bash
python3 train_tcabanski.py --test-only
```

## Understanding the Metrics

### Loss Values
- **Training Loss**: How well model fits training data
- **Validation Loss**: How well model generalizes to unseen data
- **Test Loss**: Final check on completely separate data

### Good Training Signs ✓
```
Epoch 1: train=2.0, val=1.9  (decreasing together)
Epoch 2: train=1.5, val=1.5  (staying close)
Epoch 3: train=1.3, val=1.3  (still close)
Final diff: 0.05  ← Excellent!
```

### Overfitting Signs ⚠️
```
Epoch 1: train=2.0, val=1.9  (OK)
Epoch 2: train=1.2, val=1.8  (gap opening)
Epoch 3: train=0.8, val=2.1  (BAD - overfitting!)
Final diff: 1.3  ← Model memorized training data
```

### What Loss Difference Means
- **< 0.05**: Excellent generalization ✓✓✓
- **0.05-0.1**: Good generalization ✓✓
- **0.1-0.2**: Fair generalization ✓
- **> 0.2**: Likely overfitting ⚠️

## Troubleshooting

### "No module named 'datasets'"
```bash
source calma/bin/activate
pip install -r requirements-training.txt
```

### "CUDA out of memory"
Already optimized for your GPU, but if it happens:
```bash
# Option 1: Process less data
# Edit data_processing_tcabanski.py line 243:
max_samples=15000  # Reduce from 20k

# Option 2: Clear GPU memory first
nvidia-smi  # Check what's using GPU
pkill -9 <PID>  # Kill other processes
```

### "Dataset download failed"
```bash
# Make sure you're logged in
huggingface-cli login

# Test connection
python3 -c "from datasets import load_dataset; ds = load_dataset('tcabanski/mental_health_counseling_responses', split='train[:5]')"
```

### "Model still deflects to random topics"
This means training data quality issue. Try:
1. Check if model actually loaded (check logs)
2. Verify config.py points to new model
3. Try higher quality threshold (edit line 238: quality_threshold=5)

## Comparison: Before vs After

### Before (Amod 3.5k)
```
User: "I'm tired"
Bot: "Tiredness affects all of us... sports & leisure... balance between sports..."
```
❌ Random, incoherent, disconnected

### Expected After (tcabanski 20k)
```
User: "I'm tired"
Bot: "I hear you. Feeling tired can really affect everything in your life.
Can you tell me more about what's been going on that's making you feel this way?
Is it physical exhaustion, emotional fatigue, or both?"
```
✓ Coherent, empathetic, contextually relevant

## File Structure

```
calma-ai/
├── src/
│   ├── data_processing_tcabanski.py     ← NEW: Data processor
│   └── model_training_improved.py       ← Reused: Training engine
├── data/
│   └── processed/
│       └── tcabanski_mental_health/     ← NEW: Processed dataset
│           ├── train/
│           ├── validation/
│           ├── test/
│           └── stats.json
├── models/
│   └── calma-tcabanski-final/           ← NEW: Trained model
│       ├── checkpoint-xxx/              (best checkpoints)
│       └── final/                       (final model for deployment)
├── train_tcabanski.py                    ← NEW: Training script
└── TRAINING_TCABANSKI.md                 ← This guide
```

## Performance Expectations

### Data Processing
- Time: 10-15 minutes
- Memory: ~2GB RAM
- Output: ~20k examples, ~500MB

### Training (GPU)
- Time: 2-3 hours
- Memory: ~5GB VRAM
- Model size: ~2.5GB (with LoRA)
- Checkpoints: ~3 (saves only best)

### Training (CPU)
- Time: 10-15 hours
- Memory: ~8GB RAM
- Same model size
- Not recommended unless necessary

## Success Criteria

Training is successful if:
- ✅ Validation loss < 1.5
- ✅ Test loss similar to validation (diff < 0.1)
- ✅ Responses are coherent and contextually relevant
- ✅ No random deflections or topic switching
- ✅ Appropriate empathy and professional counseling tone
- ✅ Culturally aware responses

## Next Steps After Training

1. **Backup old model** (optional but recommended):
```bash
cp -r models/calma-final models/calma-final-backup
```

2. **Update service config**:
Edit `calma-ai-service/app/config.py`:
```python
model_path: str = "../models/calma-tcabanski-final/final"
```

3. **Restart service**:
```bash
./start-calma.sh
```

4. **Test thoroughly**:
- Greetings ("Hi", "Hello")
- Light topics ("I'm bored")
- Moderate issues ("Feeling stressed")
- Serious topics ("Relationship problems")
- Crisis scenarios ("Feeling hopeless")

5. **Monitor production**:
- Watch for coherence
- Check cultural relevance
- Verify appropriate responses
- Look for any remaining deflection patterns

## Additional Resources

- [HuggingFace Dataset](https://huggingface.co/datasets/tcabanski/mental_health_counseling_responses)
- [Training Parameters Guide](RETRAINING_GUIDE.md)
- [Project Overview](CLAUDE.md)
- [Two-Model System](TWO_MODEL_SYSTEM.md)

## Questions?

Run the test setup to verify everything:
```bash
python3 test_setup.py
```

Check logs if training fails:
```bash
# During training, logs are printed to console
# After training, model info is in:
models/calma-tcabanski-final/final/training_args.bin
```

---

**Ready to train?** Start with Step 1! 🚀
