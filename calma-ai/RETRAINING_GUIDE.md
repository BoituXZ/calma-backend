# Calma AI Retraining Guide - Anti-Overfitting

This guide explains how to retrain your Calma model using the Hugging Face mental health counseling dataset with anti-overfitting measures.

## Problem: Model Overfitting

Your model was overfitting, which means:
- It memorized the training data too well
- Poor generalization to new conversations
- High training accuracy but poor validation/test accuracy

## Solution: New Training Pipeline

We've created an improved training pipeline with:

### 1. Larger, More Diverse Dataset
- **Source**: Hugging Face `Amod/mental_health_counseling_conversations`
- **Size**: Thousands of mental health counseling conversations
- **Quality**: Professional counselor responses
- **Diversity**: Various mental health topics and scenarios

### 2. Anti-Overfitting Measures

#### Data Processing (`data_processing_hf.py`)
- **Proper splits**: 80% train / 10% validation / 10% test
- **Quality filtering**: Removes low-quality, repetitive examples
- **Dynamic padding**: No padding during tokenization (added during training)
- **Text cleaning**: Removes noise and normalizes text
- **No aggressive augmentation**: Previous augmentation may have caused overfitting

#### Model Training (`model_training_improved.py`)
- **Early stopping**: Stops when validation loss stops improving
- **Weight decay**: L2 regularization (0.01)
- **Increased LoRA dropout**: 0.15 (up from 0.1)
- **Lower learning rate**: 1e-4 (more stable)
- **Gradient clipping**: Max norm of 1.0 (more conservative)
- **Better scheduling**: Cosine learning rate decay with warmup
- **Rank-stabilized LoRA**: Uses `use_rslora=True` for better stability

## Quick Start

### Prerequisites

1. **Install Hugging Face CLI** (if not already installed):
```bash
pip install huggingface_hub
```

2. **Login to Hugging Face**:
```bash
huggingface-cli login
```
Enter your Hugging Face token (get one at https://huggingface.co/settings/tokens)

### Basic Usage

**Full pipeline (recommended for first run):**
```bash
python train_with_hf_dataset.py
```

This will:
1. Download and process the dataset (~15-30 minutes)
2. Train the model with anti-overfitting measures
3. Evaluate on validation and test sets
4. Save the model to `models/calma-hf-trained/final`

### Advanced Options

**Quick test with limited samples:**
```bash
python train_with_hf_dataset.py --max-samples 1000 --epochs 2
```

**Skip data preparation (if already processed):**
```bash
python train_with_hf_dataset.py --skip-data-prep
```

**Adjust hyperparameters:**
```bash
python train_with_hf_dataset.py \
  --epochs 3 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --lora-dropout 0.15 \
  --early-stopping-patience 3
```

**Custom output directory:**
```bash
python train_with_hf_dataset.py --output-dir ./models/my-custom-model
```

### All Available Options

```
--max-samples N              Limit dataset size (useful for testing)
--epochs N                   Number of training epochs (default: 3)
--learning-rate FLOAT        Learning rate (default: 1e-4)
--weight-decay FLOAT         L2 regularization (default: 0.01)
--lora-dropout FLOAT         LoRA dropout rate (default: 0.15)
--early-stopping-patience N  Stop after N evaluations without improvement (default: 3)
--skip-data-prep            Use existing processed dataset
--output-dir PATH           Where to save the trained model
```

## Monitoring Training

During training, watch for:

### Good Signs ✓
- Validation loss decreasing smoothly
- Small gap between training and validation loss
- Test loss similar to validation loss
- Early stopping triggers (model converged)

### Warning Signs ⚠️
- Validation loss increasing while training loss decreases (overfitting)
- Large gap between validation and test loss (> 0.1)
- Validation loss plateaus early but training continues

### If Still Overfitting

If you see warning signs, try:

1. **Reduce epochs**:
```bash
python train_with_hf_dataset.py --epochs 2
```

2. **Increase regularization**:
```bash
python train_with_hf_dataset.py --weight-decay 0.02 --lora-dropout 0.2
```

3. **Lower learning rate**:
```bash
python train_with_hf_dataset.py --learning-rate 5e-5
```

4. **Use less data** (counterintuitive but sometimes helps):
```bash
python train_with_hf_dataset.py --max-samples 5000
```

## After Training

### 1. Test the Model

Use the evaluation script to check performance:
```bash
python src/model_evaluation_cpu.py
```

### 2. Update Two-Model System

According to your [TWO_MODEL_SYSTEM.md](TWO_MODEL_SYSTEM.md), you use:
- **Casual Model**: Base Llama (for early conversation)
- **Therapeutic Model**: Fine-tuned with LoRA (for mental health support)

**To update the therapeutic model:**

1. Copy the new LoRA adapters:
```bash
cp -r models/calma-hf-trained/final/* models/calma-final/
```

2. Or update your model loading path in `inference_service.py` or `model_service.py` to point to the new model.

### 3. Compare Performance

Test both old and new models:
```bash
# Old model
python src/inference.py  # Uses models/calma-final

# New model (update path in code first)
python src/inference.py  # Uses models/calma-hf-trained/final
```

## Troubleshooting

### "Failed to load dataset from Hugging Face"
- Run `huggingface-cli login` and enter your token
- Check internet connection
- Verify dataset exists: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations

### "CUDA out of memory"
- Reduce batch size (edit `model_training_improved.py`, change `batch_size = 2` to `batch_size = 1`)
- Use `--max-samples` to limit dataset size
- Train on CPU (slower but works): Just run the script, it will auto-detect

### "Dataset field names don't match"
The HF dataset preprocessing script tries multiple field name combinations. If it fails:
1. Check the dataset structure manually
2. Update field names in `data_processing_hf.py` line 169-180
3. Look for fields like: `Context`/`Response`, `questionText`/`answerText`, `input`/`output`, or `prompt`/`response`

### "Model still overfitting"
Try the progressive approach:
1. Start with 1000 samples: `--max-samples 1000`
2. If good, increase to 5000: `--max-samples 5000`
3. If still good, use all data (remove `--max-samples`)

## Understanding the Metrics

After training, you'll see:

```
Validation Loss: 1.2345
Test Loss: 1.2567
Loss Difference: 0.0222
```

**Interpretation:**
- **Low difference (< 0.1)**: Good generalization ✓
- **Medium difference (0.1-0.2)**: Acceptable, monitor in production
- **High difference (> 0.2)**: Likely overfitting, retrain with more regularization ⚠️

**Perplexity** is `exp(loss)`:
- Lower is better
- Typical range for mental health chatbots: 2-5
- If > 10, model is uncertain (needs more training or data)

## Files Created

This retraining system includes:

1. **`src/data_processing_hf.py`** - HuggingFace dataset processor with quality filtering
2. **`src/model_training_improved.py`** - Training script with anti-overfitting measures
3. **`train_with_hf_dataset.py`** - Complete pipeline script (recommended entry point)
4. **`RETRAINING_GUIDE.md`** - This guide

## Next Steps

1. Run the training pipeline
2. Monitor validation vs test loss
3. If satisfied, update your production model
4. Test in your NestJS backend
5. Monitor real-world performance

For questions or issues, refer to:
- [CLAUDE.md](CLAUDE.md) - Project overview
- [TWO_MODEL_SYSTEM.md](TWO_MODEL_SYSTEM.md) - Model architecture
- Original training: [src/model_training.py](src/model_training.py)

Good luck with your retraining! 🚀
