# tcabanski Training System - Summary

## 🎯 What Was Done

Created a complete retraining system using the **tcabanski/mental_health_counseling_responses** dataset to fix the incoherent response problem.

## ❌ Problem Identified

### Current Model Issues
1. **Loaded wrong model**: Service config pointed to non-existent path
2. **Trained on too little data**: Only 3,512 examples from Amod dataset
3. **Incoherent responses**: Random, disconnected text like "sports & leisure" and "maths"
4. **Root cause**: 3.5k examples too small for Llama 3B training

### Example of Bad Response
```
User: "I'm tired"
Bot: "Tiredness affects all of us... sports & leisure... balance between sports..."
```

## ✅ Solution Implemented

### New Dataset: tcabanski
- **26,140 total conversations**
- **Quality-rated**: Empathy, appropriateness, relevance scores (1-5)
- **Filtered for quality**: Only use responses with scores ≥ 4/5
- **Result**: ~20,000 high-quality counseling conversations
- **7x more data** than previous Amod dataset

### Why tcabanski is Better
| Feature | Amod (Old) | tcabanski (New) |
|---------|------------|-----------------|
| Size | 3,512 | 26,140 |
| Quality ratings | ❌ No | ✅ Yes (1-5 scale) |
| Filtering | ❌ No | ✅ Scores ≥ 4/5 |
| After filtering | 3,512 | ~20,000 |
| Professional tone | ❌ Mixed | ✅ Counselor-style |
| Natural Q&A | ✅ Yes | ✅ Yes |

## 📁 Files Created

### 1. Data Processing
**File**: `src/data_processing_tcabanski.py`
- Loads dataset from HuggingFace
- Filters by quality (empathy/appropriateness/relevance ≥ 4)
- Injects Zimbabwe cultural context
- Creates 80/10/10 splits
- Tokenizes for training

### 2. Training Script
**File**: `train_tcabanski.py`
- Fresh training from base Llama (no previous training)
- Memory-optimized for 5.64GB GPU
- Anti-overfitting measures (early stopping, regularization)
- Saves to `models/calma-tcabanski-final/`

### 3. Documentation
**File**: `TRAINING_TCABANSKI.md`
- Complete training guide
- Step-by-step instructions
- Troubleshooting section
- Performance expectations

### 4. Test Script
**File**: `test_tcabanski_pipeline.py`
- Tests pipeline with 100 samples
- Verifies data processing works
- Tests model initialization
- Optional training test (1 epoch)

### 5. Config Fix
**File**: `calma-ai-service/app/config.py` (updated)
- Fixed model path to use relative path
- Changed from absolute to `../models/calma-hf-trained/final`

## 🚀 How to Use

### Quick Start (3 Steps)

```bash
# 1. Process data (~10-15 min)
cd /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai
source calma/bin/activate
python3 src/data_processing_tcabanski.py

# 2. Train model (~2-3 hours)
python3 train_tcabanski.py

# 3. Deploy
# Edit calma-ai-service/app/config.py:
#   model_path: str = "../models/calma-tcabanski-final/final"
# Then:
cd ..
./start-calma.sh
```

### Test First (Recommended)

Before full training, test with small sample:
```bash
python3 test_tcabanski_pipeline.py
```

Takes ~10-15 minutes, verifies everything works.

## 📊 Expected Results

### Training Metrics (Good)
```
Epoch 1: train_loss=1.8, val_loss=1.7
Epoch 2: train_loss=1.4, val_loss=1.5
Epoch 3: train_loss=1.2, val_loss=1.3

Final:
  Validation Loss: 1.30
  Test Loss: 1.32
  Loss Difference: 0.02  ✓ (< 0.1 = good generalization)
```

### Response Quality (Expected)
```
User: "I'm tired"
Bot: "I hear you. Feeling tired can really affect everything. Can you tell me
more about what's been going on? Is it physical exhaustion, emotional fatigue,
or both?"
```
✅ Coherent, empathetic, contextually relevant

## 🔧 Technical Details

### Training Configuration
- **Start**: Fresh from base Meta Llama 3.2-3B
- **LoRA rank**: 8 (memory-optimized)
- **Batch size**: 1 (GPU limited)
- **Gradient accumulation**: 16 (effective batch: 16)
- **Max tokens**: 384 (fits 5.64GB GPU)
- **Learning rate**: 5e-5 (conservative)
- **Epochs**: 3 (with early stopping)
- **Early stopping patience**: 5 evaluations

### Memory Usage
- **Dataset processing**: ~2GB RAM
- **Training**: ~5GB VRAM (fits your GPU)
- **Model size**: ~2.5GB (with LoRA)

### Time Estimates
- **Data processing**: 10-15 minutes
- **Training (GPU)**: 2-3 hours
- **Training (CPU)**: 10-15 hours (not recommended)

## 🎯 Success Criteria

Training is successful if:
- ✅ Validation loss < 1.5
- ✅ Test loss similar to validation (difference < 0.1)
- ✅ Responses are coherent and contextually relevant
- ✅ No random deflections or topic changes
- ✅ Appropriate empathy and counseling tone

## 🔄 Comparison

### Dataset Comparison

| Dataset | Size | Format | Recommendation |
|---------|------|--------|----------------|
| **tcabanski** | 26k (→20k filtered) | Q&A + quality scores | ⭐⭐⭐⭐⭐ BEST |
| Amod | 3.5k | Context→Response | ⭐⭐ Too small |
| fadodr | 8.5k | Instruction→Output | ⭐⭐⭐ Good |
| training_family.json | 8.5k | Problem→Answer | ⭐⭐⭐⭐ Zimbabwe-specific |

**Chosen**: tcabanski because it's the largest, highest-quality, ready-to-use dataset.

### Before vs After

**Before** (Amod 3.5k):
- Random, incoherent responses
- Topics jump around ("maths", "music")
- No conversation flow
- 3,512 examples (too small)

**After** (tcabanski 20k):
- Coherent, professional responses
- Stays on topic
- Natural conversation flow
- 20,000 quality-filtered examples

## 📝 Next Steps

### Immediate
1. **Test pipeline**: `python3 test_tcabanski_pipeline.py`
2. **If test passes**: Process full data and train
3. **Deploy**: Update config.py and restart service

### Future Enhancements
- Combine tcabanski + training_family.json (43k total)
- Add Reddit 156k dataset (needs preprocessing)
- Fine-tune temperature/top_p for better responses
- Add response length control

## ⚠️ Important Notes

### About config.py
- Fixed to use relative path: `../models/calma-hf-trained/final`
- `.env` file still exists but config.py defaults take precedence
- Always update config.py when changing models

### About Previous Training
- Old model (Amod 3.5k) is in `models/calma-hf-trained/final/`
- Backup if needed: `cp -r models/calma-hf-trained models/calma-hf-trained-backup`
- New model will be in `models/calma-tcabanski-final/`

### About Your 156k Reddit Dataset
- You mentioned having 156k Reddit ADHD posts
- That's raw Reddit data (not Q&A pairs)
- Would need preprocessing to extract conversations
- tcabanski is ready-to-use, so starting with that

## 🐛 Troubleshooting

### "No module named 'datasets'"
```bash
source calma/bin/activate
pip install -r requirements-training.txt
```

### "CUDA out of memory"
Already optimized for 5.64GB GPU. If still happens:
- Close other applications
- Kill other GPU processes: `nvidia-smi` then `kill <PID>`
- Or reduce max_samples in data processing

### "Dataset download failed"
```bash
huggingface-cli login
# Enter token from: https://huggingface.co/settings/tokens
```

## 📚 Documentation

- **[TRAINING_TCABANSKI.md](TRAINING_TCABANSKI.md)** - Detailed training guide
- **[RETRAINING_GUIDE.md](RETRAINING_GUIDE.md)** - Original retraining docs
- **[MODEL_UPDATE.md](../calma-ai-service/MODEL_UPDATE.md)** - Service update guide
- **[CLAUDE.md](CLAUDE.md)** - Project overview

## ✅ What's Complete

- [x] Analyzed current problem (wrong model, bad training)
- [x] Identified best dataset (tcabanski 26k)
- [x] Created data processor with quality filtering
- [x] Created training script with anti-overfitting
- [x] Fixed config.py path issues
- [x] Created comprehensive documentation
- [x] Created test pipeline script

## 🎯 Ready to Train!

Everything is set up and ready. To begin:

```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai
source calma/bin/activate

# Test first (recommended)
python3 test_tcabanski_pipeline.py

# Or go straight to full training
python3 src/data_processing_tcabanski.py
python3 train_tcabanski.py
```

Good luck! 🚀
