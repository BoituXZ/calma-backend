# Training Pipeline Summary

## 🎯 What Was Done

Your Calma AI model was overfitting. I've created a complete retraining pipeline with anti-overfitting measures using a large, diverse Hugging Face dataset.

## 📦 New Files Created

### Core Training Scripts
1. **`train_with_hf_dataset.py`** - Complete training pipeline (main entry point)
2. **`src/data_processing_hf.py`** - HuggingFace dataset processor with quality filtering
3. **`src/model_training_improved.py`** - Training with anti-overfitting measures

### Setup & Testing
4. **`setup_training.sh`** - One-command environment setup
5. **`test_setup.py`** - Verify installation and configuration
6. **`requirements-training.txt`** - Python dependencies

### Documentation
7. **`QUICKSTART.md`** - Quick start guide (recommended read!)
8. **`RETRAINING_GUIDE.md`** - Comprehensive training documentation
9. **`TRAINING_SUMMARY.md`** - This file

## 🔧 Anti-Overfitting Measures Implemented

### Dataset Level
- ✅ Using large, diverse HF dataset (10,000+ conversations vs 248KB)
- ✅ Proper train/validation/test splits (80/10/10)
- ✅ Quality filtering (removes repetitive, low-quality examples)
- ✅ Dynamic padding (no padding during tokenization)
- ✅ Removed aggressive data augmentation

### Model Level
- ✅ Early stopping (stops when validation loss stops improving)
- ✅ Increased LoRA dropout (0.15 vs 0.1)
- ✅ Weight decay / L2 regularization (0.01)
- ✅ Lower learning rate (1e-4 vs 2e-4)
- ✅ Stronger gradient clipping (1.0 vs 0.3)
- ✅ Rank-stabilized LoRA (`use_rslora=True`)
- ✅ Frequent validation checks for early stopping

### Training Process
- ✅ Separate validation and test sets
- ✅ Saves only best 3 checkpoints
- ✅ Monitors validation vs test loss difference
- ✅ Automatic warnings if overfitting detected

## 🚀 How to Use

### Quick Start (3 commands)
```bash
./setup_training.sh              # Setup environment
huggingface-cli login            # Login (first time only)
python3 train_with_hf_dataset.py # Start training
```

### Test First (Recommended)
```bash
./setup_training.sh
huggingface-cli login
python3 train_with_hf_dataset.py --max-samples 100 --epochs 1
```

### Full Documentation
See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## 📊 Expected Results

### Before (Overfitting)
```
Training Loss: 0.3
Validation Loss: 2.5
Test Loss: 2.8
→ Large gap indicates overfitting
```

### After (Good Generalization)
```
Training Loss: 1.2
Validation Loss: 1.3
Test Loss: 1.3
→ Small gap (< 0.1) indicates good generalization
```

## 🎯 Key Improvements

| Aspect | Old | New |
|--------|-----|-----|
| Dataset | 248KB (~300 examples) | 10,000+ conversations |
| Splits | Train/Test (80/20) | Train/Val/Test (80/10/10) |
| Augmentation | Aggressive (1.3x) | Minimal, quality-focused |
| Dropout | 0.1 | 0.15 |
| Early Stopping | ❌ | ✅ |
| Weight Decay | 0.001 | 0.01 |
| Learning Rate | 2e-4 | 1e-4 |
| Quality Filtering | ❌ | ✅ |

## 📁 Directory Structure After Training

```
calma-ai/
├── train_with_hf_dataset.py          ← Run this!
├── setup_training.sh                 ← Setup script
├── test_setup.py                     ← Test setup
├── QUICKSTART.md                     ← Quick start guide
├── RETRAINING_GUIDE.md              ← Detailed guide
├── requirements-training.txt         ← Dependencies
├── src/
│   ├── data_processing_hf.py        ← HF dataset processor
│   ├── model_training_improved.py   ← Training script
│   └── ...
├── data/
│   └── processed/
│       └── hf_mental_health_dataset/ ← Processed data
├── models/
│   ├── calma-final/                 ← Old model (backup this!)
│   └── calma-hf-trained/            ← New trained model
│       └── final/                   ← Use this!
└── calma/                           ← Virtual environment
```

## ⚙️ Configuration Options

All options for `train_with_hf_dataset.py`:

```bash
--max-samples N               # Limit dataset size (for testing)
--epochs N                    # Training epochs (default: 3)
--learning-rate FLOAT         # Learning rate (default: 1e-4)
--weight-decay FLOAT          # L2 regularization (default: 0.01)
--lora-dropout FLOAT          # LoRA dropout (default: 0.15)
--early-stopping-patience N   # Stop after N evals without improvement
--skip-data-prep             # Use existing processed dataset
--output-dir PATH            # Model output directory
```

## 🔍 How to Know If It Worked

### During Training
Watch for:
- Validation loss decreasing smoothly
- Small gap between training and validation loss
- Early stopping trigger (means it converged)

### After Training
Check the metrics:
```
Loss Difference: 0.05 → ✅ Excellent (< 0.1)
Loss Difference: 0.15 → ⚠️  Fair (0.1-0.2)
Loss Difference: 0.35 → ❌ Still overfitting (> 0.2)
```

### In Production
Test with real conversations:
- More natural responses
- Better handling of diverse topics
- Doesn't repeat memorized phrases

## 🔄 Integration with Two-Model System

Your system uses two models (from [TWO_MODEL_SYSTEM.md](../TWO_MODEL_SYSTEM.md)):

1. **Casual Model** - Base Llama (unchanged)
2. **Therapeutic Model** - Your fine-tuned model (← update this)

After training:
```bash
# Backup old model
cp -r models/calma-final models/calma-final-backup

# Use new model
cp -r models/calma-hf-trained/final/* models/calma-final/

# Restart backend
cd ..
npm run start:dev
```

## 🐛 Troubleshooting

### Setup Issues
Run: `python3 test_setup.py`

### Authentication Issues
```bash
huggingface-cli login
```

### Memory Issues
- Reduce samples: `--max-samples 5000`
- Lower batch size: Edit `model_training_improved.py` line 102

### Still Overfitting
Progressive approach:
```bash
python3 train_with_hf_dataset.py --max-samples 1000  # Start small
python3 train_with_hf_dataset.py --max-samples 5000  # Increase
python3 train_with_hf_dataset.py                     # Full dataset
```

## 📚 Dataset Information

**Source**: [`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)

**Content**: Professional mental health counseling conversations
**Size**: ~10,000+ conversations
**Quality**: High (professional counselor responses)
**License**: Check dataset page on Hugging Face

## 🎓 Technical Details

### Why This Fixes Overfitting

1. **More diverse data** → Model learns patterns, not examples
2. **Quality filtering** → Removes noise that causes overfitting
3. **Early stopping** → Stops before memorization begins
4. **Regularization** → Penalizes complex models
5. **Proper validation** → Detects overfitting during training
6. **Dynamic padding** → Doesn't learn padding patterns

### LoRA Configuration
```python
r=16                    # Rank
lora_alpha=32           # Scaling
lora_dropout=0.15       # Dropout (increased)
use_rslora=True         # Rank-stabilized (new)
target_modules=[        # All attention + MLP
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Training Configuration
```python
learning_rate=1e-4              # Lower for stability
weight_decay=0.01               # L2 regularization
max_grad_norm=1.0               # Gradient clipping
lr_scheduler_type="cosine"      # Smooth decay
early_stopping_patience=3       # Stop after 3 no-improve
```

## ✅ Checklist

Before training:
- [ ] Virtual environment activated
- [ ] Dependencies installed (`setup_training.sh`)
- [ ] Hugging Face login (`huggingface-cli login`)
- [ ] Test passed (`python3 test_setup.py`)
- [ ] Backed up old model

After training:
- [ ] Validation-test loss difference < 0.1
- [ ] Tested with evaluation script
- [ ] Integrated with backend
- [ ] Tested in production
- [ ] Monitored real-world performance

## 🎉 Next Steps

1. **Setup**: Run `./setup_training.sh`
2. **Login**: `huggingface-cli login`
3. **Test**: `python3 train_with_hf_dataset.py --max-samples 100 --epochs 1`
4. **Train**: `python3 train_with_hf_dataset.py`
5. **Evaluate**: Check loss difference
6. **Deploy**: Update model in production
7. **Monitor**: Watch real-world performance

## 📖 Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [RETRAINING_GUIDE.md](RETRAINING_GUIDE.md) - Detailed documentation
- [TWO_MODEL_SYSTEM.md](../TWO_MODEL_SYSTEM.md) - Model architecture
- [CLAUDE.md](CLAUDE.md) - Project overview

---

**Questions?** Check the documentation or run `python3 test_setup.py` to verify setup.

**Ready to train?** Start with [QUICKSTART.md](QUICKSTART.md)! 🚀
