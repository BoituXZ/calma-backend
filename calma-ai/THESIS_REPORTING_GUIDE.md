## Thesis Reporting System - Complete Guide

## Overview

Your training pipeline now includes comprehensive automatic reporting for thesis Chapter 4. All metrics, evaluations, and visualizations are automatically generated and saved in thesis-ready formats.

## What's New

### 1. Training Reporter Module
**File**: `src/training_reporter.py`

Automatically tracks and saves:
- Training metrics (loss, time, memory) per epoch
- Model configuration and hyperparameters
- Evaluation results from test scenarios
- Performance metrics
- Loss curves for plotting
- Thesis-ready Markdown summaries

### 2. Enhanced Training Script
**File**: `train_with_reporting.py`

Training script that:
- Integrates comprehensive logging
- Auto-saves metrics after each epoch
- Runs test scenarios after training
- Generates all reports automatically
- Handles interruptions gracefully

### 3. Evaluation Script
**File**: `evaluate_trained_model.py`

Standalone evaluation for your trained model:
- 20+ test scenarios covering all use cases
- Detailed response analysis
- Performance metrics calculation
- Thesis-ready reports

## Quick Start

### Option 1: Evaluate Your Existing Model (RECOMMENDED)

Since you've already trained the tcabanski model:

```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai
source calma/bin/activate
python3 evaluate_trained_model.py
```

**Time**: ~5-10 minutes
**Output**: Complete thesis reports in `results/run_TIMESTAMP/`

### Option 2: Train With Reporting (Future Training)

For your next training run:

```bash
python3 train_with_reporting.py --dataset tcabanski --epochs 3
```

This will:
1. Train the model
2. Log all metrics during training
3. Run test scenarios
4. Generate all reports

## Generated Reports

After running either script, you'll get:

### 1. `training_report.txt`
Human-readable training log:
```
CALMA AI TRAINING REPORT
========================

MODEL INFORMATION
- model_name: meta-llama/Llama-3.2-3B-Instruct
- trainable_parameters: 12,345,678
- total_parameters: 3,237,063,680
- trainable_percentage: 0.38%

HYPERPARAMETERS
- learning_rate: 5e-05
- epochs: 3
- batch_size: 1
- lora_rank: 8

TRAINING TIMELINE
- Start time: 2025-10-13T03:00:00
- End time: 2025-10-13T05:30:00
- Total time: 9000 seconds (2.5 hours)

EPOCH-BY-EPOCH RESULTS
Epoch    Train Loss   Val Loss     LR           Time (s)
1        1.8000       1.7500       0.000050     3000
2        1.4000       1.4500       0.000040     3000
3        1.2000       1.2500       0.000030     3000
```

### 2. `training_metrics.json`
Structured training data:
```json
{
  "start_time": "2025-10-13T03:00:00",
  "end_time": "2025-10-13T05:30:00",
  "total_time_seconds": 9000,
  "model_info": {
    "trainable_parameters": 12345678,
    "total_parameters": 3237063680
  },
  "hyperparameters": {
    "learning_rate": 5e-05,
    "epochs": 3
  },
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 1.8,
      "val_loss": 1.75,
      "learning_rate": 0.00005,
      "epoch_time_seconds": 3000,
      "gpu_memory_mb": 5120
    }
  ]
}
```

### 3. `evaluation_report.txt`
Detailed test scenario results:
```
CALMA AI EVALUATION REPORT
==========================

PERFORMANCE SUMMARY
- total_scenarios: 20
- successful_responses: 20
- success_rate_percent: 100.0
- average_response_length_words: 85.5
- average_response_time_seconds: 2.341

DETAILED TEST SCENARIOS
========================

SCENARIO 1: Relationship - Trust
User Input: My partner said he doesn't trust me
Generated Response: [Full AI response here]
Response Length: 92 words
Response Time: 2.15 seconds
```

### 4. `evaluation_results.json`
Structured evaluation data:
```json
{
  "timestamp": "2025-10-13T06:00:00",
  "test_scenarios": [
    {
      "scenario_name": "Relationship - Trust",
      "category": "relationship",
      "user_input": "My partner said he doesn't trust me",
      "response": "[Full AI response]",
      "response_word_count": 92,
      "response_time_seconds": 2.15,
      "error": null
    }
  ],
  "performance_metrics": {
    "success_rate_percent": 100.0,
    "average_response_length_words": 85.5
  }
}
```

### 5. `performance_metrics.json`
Performance summary:
```json
{
  "total_scenarios": 20,
  "successful_responses": 20,
  "success_rate_percent": 100.0,
  "average_response_length_words": 85.5,
  "average_response_time_seconds": 2.341,
  "errors_encountered": 0,
  "error_rate_percent": 0.0
}
```

### 6. `loss_curves.csv`
CSV data for plotting in Excel/Python:
```csv
epoch,train_loss,val_loss,learning_rate
1,1.8000,1.7500,0.000050
2,1.4000,1.4500,0.000040
3,1.2000,1.2500,0.000030
```

### 7. `CHAPTER4_SUMMARY.md`
**Thesis-ready Markdown report** - Copy directly into your thesis!

```markdown
# Chapter 4: Results and Discussion

## 4.1 Model Configuration

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Llama-3.2-3B-Instruct |
| LoRA Rank | 8 |
| Trainable Parameters | 12,345,678 |
| Total Parameters | 3,237,063,680 |
| Trainable % | 0.38% |

### Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 5e-05 |
| Epochs | 3 |
| Batch Size | 1 |
| Weight Decay | 0.01 |

## 4.2 Training Results

**Training Duration**: 2.50 hours

### Loss Progression

| Epoch | Training Loss | Validation Loss | Learning Rate |
|-------|---------------|-----------------|---------------|
| 1 | 1.8000 | 1.7500 | 0.000050 |
| 2 | 1.4000 | 1.4500 | 0.000040 |
| 3 | 1.2000 | 1.2500 | 0.000030 |

## 4.3 Model Evaluation

### Performance Metrics

| Metric | Value |
|--------|-------|
| Success Rate | 100.0% |
| Avg Response Length | 85.5 words |
| Avg Response Time | 2.341s |

### Sample Model Responses

[Detailed examples with actual AI responses]
```

## Using the Reports in Your Thesis

### For Tables
Copy directly from `CHAPTER4_SUMMARY.md` - all tables are formatted for Markdown/LaTeX.

### For Graphs
Import `loss_curves.csv` into:
- **Excel**: Create line charts
- **Python/Matplotlib**:
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  df = pd.read_csv('loss_curves.csv')
  plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
  plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('loss_curve.png')
  ```

### For Metrics
Use `performance_metrics.json` for:
- Success rates
- Average response times
- Response lengths
- Error analysis

### For Qualitative Analysis
Use `evaluation_report.txt` to:
- Show example conversations
- Analyze response quality
- Discuss model behavior

## Directory Structure

After running evaluation:
```
results/
└── run_20251013_060000/
    ├── training_report.txt              ← Human-readable training log
    ├── training_metrics.json            ← Structured training data
    ├── evaluation_report.txt            ← Test scenario results
    ├── evaluation_results.json          ← Structured evaluation data
    ├── performance_metrics.json         ← Performance summary
    ├── loss_curves.csv                  ← CSV for plotting
    └── CHAPTER4_SUMMARY.md              ← Thesis-ready report ⭐
```

## Test Scenarios Included

Your evaluation includes 20+ comprehensive scenarios:

**1. Greetings**
- Simple greeting
- "How are you" variations

**2. Physical Symptoms**
- Fatigue (simple and detailed)

**3. Relationship Issues**
- Trust problems
- Communication issues
- Constant fighting

**4. Family Issues**
- Family conflict
- Parental pressure
- Cultural disapproval

**5. Mental Health**
- Anxiety
- Stress/overwhelm
- Depression
- Loneliness

**6. Work/Academic**
- Work pressure
- Exam stress

**7. Cultural Context**
- Ubuntu philosophy
- Community expectations

**8. Conversational**
- Follow-up responses
- Clarification requests

## Customizing Test Scenarios

Edit `evaluate_trained_model.py` to add your own scenarios:

```python
scenarios = [
    {
        "name": "Your Scenario Name",
        "input": "User input text",
        "category": "your_category"
    },
    # Add more...
]
```

## Integration with Your Existing Model

The model is already configured in `config.py`:

```python
model_path: str = "../models/calma-tcabanski-final/checkpoint-400"
```

To test the deployed model in production:
```bash
cd /home/boitu/Desktop/Coding/Calma/calma-backend
./start-calma.sh
```

Then test via API or frontend.

## Troubleshooting

### "Model not found"
Update the model path in `evaluate_trained_model.py` line 148:
```python
model_path = "models/calma-tcabanski-final/checkpoint-400"  # Your actual path
```

### "CUDA out of memory"
Edit the script to use CPU:
```python
device = "cpu"  # Force CPU instead of auto-detect
```

### "No scenarios run"
Check that model loaded successfully. Look for:
```
✓ Model loaded successfully
```

### "Empty reports"
Make sure evaluation completed. Check for:
```
✓ ALL SCENARIOS COMPLETED
```

## Next Steps

1. **Run evaluation now**:
   ```bash
   python3 evaluate_trained_model.py
   ```

2. **Review generated reports** in `results/run_TIMESTAMP/`

3. **Copy tables** from `CHAPTER4_SUMMARY.md` into your thesis

4. **Create visualizations** using `loss_curves.csv`

5. **Analyze responses** from `evaluation_report.txt`

6. **Add your discussion** to the Markdown report

## Tips for Thesis Writing

### For Chapter 4.1 (Methodology)
Use:
- `training_metrics.json` → Model architecture table
- `CHAPTER4_SUMMARY.md` → Hyperparameters table

### For Chapter 4.2 (Training Results)
Use:
- `training_report.txt` → Training timeline
- `loss_curves.csv` → Loss progression graph
- `CHAPTER4_SUMMARY.md` → Loss table

### For Chapter 4.3 (Evaluation)
Use:
- `performance_metrics.json` → Performance metrics table
- `evaluation_report.txt` → Example conversations
- `CHAPTER4_SUMMARY.md` → Sample responses

### For Chapter 4.4 (Discussion)
Use:
- All reports for qualitative analysis
- Compare with baseline/other models
- Discuss successes and limitations

## Support

All files are automatically saved with error handling. If training/evaluation is interrupted:
- Training metrics are auto-saved after each epoch
- Partial results are still available
- Re-run the script to complete

---

**Ready to generate your thesis reports?**

```bash
python3 evaluate_trained_model.py
```

All reports will be in `results/run_TIMESTAMP/` ready for your thesis! 🎓
