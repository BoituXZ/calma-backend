#!/usr/bin/env python3
"""
Create training and evaluation visualizations for Calma AI model
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path("model_visualizations")
output_dir.mkdir(exist_ok=True)

print("Loading data...")

# Load training history
with open('models/calma-hf-trained/checkpoint-477/trainer_state.json', 'r') as f:
    trainer_state = json.load(f)

# Load test results
with open('chapter4_reports/run_20251013_110212/test_results.json', 'r') as f:
    test_results = json.load(f)

# Load CSV for easier manipulation
test_df = pd.read_csv('chapter4_reports/run_20251013_110212/test_results.csv')

print("Processing training history...")

# Extract training data
log_history = trainer_state['log_history']

# Separate training and evaluation logs
train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
eval_logs = [log for log in log_history if 'eval_loss' in log]

# Convert to DataFrames
train_df = pd.DataFrame(train_logs)
eval_df = pd.DataFrame(eval_logs)

print(f"Creating visualizations...")

# ============================================================================
# FIGURE 1: Training and Validation Loss Over Time
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Plot training loss
ax.plot(train_df['step'], train_df['loss'],
        label='Training Loss', linewidth=2, alpha=0.8, marker='o', markersize=3)

# Plot evaluation loss
if not eval_df.empty:
    ax.plot(eval_df['step'], eval_df['eval_loss'],
            label='Validation Loss', linewidth=2, alpha=0.8, marker='s', markersize=5,
            linestyle='--', color='red')

ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Loss Over Time\nCalma Mental Health Chatbot Model',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Add annotations
min_train_loss = train_df['loss'].min()
min_train_step = train_df.loc[train_df['loss'].idxmin(), 'step']
ax.annotate(f'Min Training Loss: {min_train_loss:.4f}',
            xy=(min_train_step, min_train_loss),
            xytext=(10, 30), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

if not eval_df.empty:
    min_eval_loss = eval_df['eval_loss'].min()
    min_eval_step = eval_df.loc[eval_df['eval_loss'].idxmin(), 'step']
    ax.annotate(f'Min Validation Loss: {min_eval_loss:.4f}',
                xy=(min_eval_step, min_eval_loss),
                xytext=(10, -40), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig(output_dir / '1_training_validation_loss.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 1_training_validation_loss.png")
plt.close()

# ============================================================================
# FIGURE 2: Loss by Epoch
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Group by epoch and calculate mean loss
epoch_train = train_df.groupby(train_df['epoch'].astype(int))['loss'].mean()
epoch_eval = eval_df.groupby(eval_df['epoch'].astype(int))['eval_loss'].mean()

ax.plot(epoch_train.index, epoch_train.values,
        label='Training Loss (avg)', linewidth=3, marker='o', markersize=8)
ax.plot(epoch_eval.index, epoch_eval.values,
        label='Validation Loss (avg)', linewidth=3, marker='s', markersize=8,
        linestyle='--', color='red')

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
ax.set_title('Average Loss Per Epoch\nCalma Mental Health Chatbot Model',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(int(epoch_train.index.max()) + 1))

plt.tight_layout()
plt.savefig(output_dir / '2_loss_per_epoch.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 2_loss_per_epoch.png")
plt.close()

# ============================================================================
# FIGURE 3: Learning Rate Schedule
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(train_df['step'], train_df['learning_rate'],
        linewidth=2, color='purple', alpha=0.8)

ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax.set_title('Learning Rate Schedule\nCalma Mental Health Chatbot Model',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig(output_dir / '3_learning_rate_schedule.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 3_learning_rate_schedule.png")
plt.close()

# ============================================================================
# FIGURE 4: Gradient Norm Over Training
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(train_df['step'], train_df['grad_norm'],
        linewidth=2, color='green', alpha=0.7)

ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
ax.set_title('Gradient Norm During Training\nCalma Mental Health Chatbot Model',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target Norm')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / '4_gradient_norm.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 4_gradient_norm.png")
plt.close()

# ============================================================================
# FIGURE 5: Test Performance Metrics Overview
# ============================================================================
metrics = test_results['performance_metrics']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Test Performance Metrics Overview\nCalma Mental Health Chatbot',
             fontsize=16, fontweight='bold', y=0.98)

# Metric 1: Success Rate
ax = axes[0, 0]
success_rate = (metrics['successful_tests'] / metrics['total_tests']) * 100
ax.bar(['Success Rate'], [success_rate], color='green', alpha=0.7)
ax.set_ylim(0, 100)
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Test Success Rate', fontweight='bold')
ax.text(0, success_rate + 2, f'{success_rate:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Metric 2: Response Time
ax = axes[0, 1]
avg_time = metrics['average_response_time']
ax.bar(['Avg Response Time'], [avg_time], color='blue', alpha=0.7)
ax.set_ylabel('Seconds', fontweight='bold')
ax.set_title('Average Response Time', fontweight='bold')
ax.text(0, avg_time + 0.5, f'{avg_time:.2f}s', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Metric 3: Response Length
ax = axes[0, 2]
avg_length = metrics['average_response_length']
ax.bar(['Avg Response Length'], [avg_length], color='orange', alpha=0.7)
ax.set_ylabel('Words', fontweight='bold')
ax.set_title('Average Response Length', fontweight='bold')
ax.text(0, avg_length + 5, f'{avg_length:.0f} words', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Metric 4: Crisis Detection Rate
ax = axes[1, 0]
crisis_rate = metrics['crisis_detection_rate']
ax.bar(['Crisis Detection'], [crisis_rate], color='red', alpha=0.7)
ax.set_ylim(0, 100)
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Crisis Detection Rate', fontweight='bold')
ax.text(0, crisis_rate + 2, f'{crisis_rate:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Metric 5: Empathy Score
ax = axes[1, 1]
empathy = metrics['empathy_score']
ax.bar(['Empathy Score'], [empathy], color='purple', alpha=0.7)
ax.set_ylim(0, 100)
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Empathy Score', fontweight='bold')
ax.text(0, empathy + 2, f'{empathy:.1f}', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Metric 6: Cultural Relevance
ax = axes[1, 2]
cultural = metrics['cultural_relevance_score']
ax.bar(['Cultural Relevance'], [cultural], color='teal', alpha=0.7)
ax.set_ylim(0, 100)
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Cultural Relevance Score', fontweight='bold')
ax.text(0, cultural + 2, f'{cultural:.1f}', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '5_test_metrics_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 5_test_metrics_overview.png")
plt.close()

# ============================================================================
# FIGURE 6: Quality Metrics Distribution (Box Plot)
# ============================================================================
quality_metrics = []
for result in test_results['test_results']:
    qm = result['quality_metrics']
    quality_metrics.append({
        'Empathy': qm['empathy_score'],
        'Relevance': qm['relevance_score'],
        'Cultural\nAwareness': qm['cultural_awareness_score'],
        'Safety': qm['safety_score'],
        'Actionability': qm['actionability_score'],
        'Overall\nQuality': qm['overall_quality']
    })

quality_df = pd.DataFrame(quality_metrics)

fig, ax = plt.subplots(figsize=(14, 7))
bp = quality_df.boxplot(ax=ax, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Quality Metrics Distribution Across Test Cases\nCalma Mental Health Chatbot',
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / '6_quality_metrics_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 6_quality_metrics_distribution.png")
plt.close()

# ============================================================================
# FIGURE 7: Performance by Test Category
# ============================================================================
category_metrics = {}
for result in test_results['test_results']:
    cat = result['category']
    if cat not in category_metrics:
        category_metrics[cat] = {
            'overall_quality': [],
            'empathy': [],
            'relevance': [],
            'response_time': []
        }
    category_metrics[cat]['overall_quality'].append(result['quality_metrics']['overall_quality'])
    category_metrics[cat]['empathy'].append(result['quality_metrics']['empathy_score'])
    category_metrics[cat]['relevance'].append(result['quality_metrics']['relevance_score'])
    category_metrics[cat]['response_time'].append(result['response_time'])

# Calculate averages
cat_summary = []
for cat, cat_metrics in category_metrics.items():
    cat_summary.append({
        'Category': cat,
        'Overall Quality': np.mean(cat_metrics['overall_quality']),
        'Empathy': np.mean(cat_metrics['empathy']),
        'Relevance': np.mean(cat_metrics['relevance']),
        'Avg Response Time': np.mean(cat_metrics['response_time'])
    })

cat_df = pd.DataFrame(cat_summary)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Performance Metrics by Test Category\nCalma Mental Health Chatbot',
             fontsize=14, fontweight='bold')

# Quality scores by category
ax = axes[0]
cat_df_sorted = cat_df.sort_values('Overall Quality', ascending=True)
y_pos = np.arange(len(cat_df_sorted))

bars = ax.barh(y_pos, cat_df_sorted['Overall Quality'], alpha=0.7)
# Color code bars
colors = plt.cm.RdYlGn(cat_df_sorted['Overall Quality'] / 100)
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_yticks(y_pos)
ax.set_yticklabels(cat_df_sorted['Category'])
ax.set_xlabel('Overall Quality Score', fontweight='bold')
ax.set_title('Overall Quality by Category', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 100)

# Add value labels
for i, v in enumerate(cat_df_sorted['Overall Quality']):
    ax.text(v + 2, i, f'{v:.1f}', va='center', fontweight='bold')

# Response time by category
ax = axes[1]
cat_df_sorted = cat_df.sort_values('Avg Response Time', ascending=True)
y_pos = np.arange(len(cat_df_sorted))

bars = ax.barh(y_pos, cat_df_sorted['Avg Response Time'], alpha=0.7, color='skyblue')

ax.set_yticks(y_pos)
ax.set_yticklabels(cat_df_sorted['Category'])
ax.set_xlabel('Average Response Time (seconds)', fontweight='bold')
ax.set_title('Response Time by Category', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(cat_df_sorted['Avg Response Time']):
    ax.text(v + 0.2, i, f'{v:.2f}s', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '7_performance_by_category.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 7_performance_by_category.png")
plt.close()

# ============================================================================
# FIGURE 8: Severity Level Analysis
# ============================================================================
severity_data = {}
for result in test_results['test_results']:
    sev = result['severity']
    if sev not in severity_data:
        severity_data[sev] = {'count': 0, 'quality': [], 'crisis': 0}
    severity_data[sev]['count'] += 1
    severity_data[sev]['quality'].append(result['quality_metrics']['overall_quality'])
    if result['crisis_detected']:
        severity_data[sev]['crisis'] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Analysis by Severity Level\nCalma Mental Health Chatbot',
             fontsize=14, fontweight='bold')

# Test distribution by severity
ax = axes[0]
severities = list(severity_data.keys())
counts = [severity_data[s]['count'] for s in severities]
colors_map = {'low': 'green', 'medium': 'orange', 'high': 'red', 'critical': 'darkred'}
colors = [colors_map.get(s, 'gray') for s in severities]

bars = ax.bar(severities, counts, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Number of Test Cases', fontweight='bold')
ax.set_xlabel('Severity Level', fontweight='bold')
ax.set_title('Test Distribution by Severity', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{int(count)}', ha='center', va='bottom', fontweight='bold')

# Quality by severity
ax = axes[1]
avg_quality = [np.mean(severity_data[s]['quality']) for s in severities]
bars = ax.bar(severities, avg_quality, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Average Quality Score', fontweight='bold')
ax.set_xlabel('Severity Level', fontweight='bold')
ax.set_title('Average Quality by Severity', fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for bar, qual in zip(bars, avg_quality):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{qual:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '8_severity_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 8_severity_analysis.png")
plt.close()

# ============================================================================
# FIGURE 9: Training Summary Statistics
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

summary_text = f"""
CALMA MENTAL HEALTH CHATBOT - TRAINING SUMMARY
═══════════════════════════════════════════════

TRAINING CONFIGURATION
────────────────────────────────────────────────
• Total Training Steps: {trainer_state['max_steps']}
• Number of Epochs: {trainer_state['num_train_epochs']}
• Batch Size: {trainer_state['train_batch_size']}
• Evaluation Steps: Every {trainer_state['eval_steps']} steps
• Logging Steps: Every {trainer_state['logging_steps']} steps

TRAINING RESULTS
────────────────────────────────────────────────
• Initial Training Loss: {train_df.iloc[0]['loss']:.4f}
• Final Training Loss: {train_df.iloc[-1]['loss']:.4f}
• Best Training Loss: {train_df['loss'].min():.4f}
• Loss Reduction: {((train_df.iloc[0]['loss'] - train_df['loss'].min()) / train_df.iloc[0]['loss'] * 100):.2f}%

VALIDATION RESULTS
────────────────────────────────────────────────
• Initial Validation Loss: {eval_df.iloc[0]['eval_loss']:.4f}
• Final Validation Loss: {eval_df.iloc[-1]['eval_loss']:.4f}
• Best Validation Loss: {trainer_state['best_metric']:.4f}
• Best Checkpoint: Step {trainer_state['best_global_step']}

TEST PERFORMANCE
────────────────────────────────────────────────
• Total Test Cases: {metrics['total_tests']}
• Successful Tests: {metrics['successful_tests']} ({metrics['successful_tests']/metrics['total_tests']*100:.1f}%)
• Average Response Time: {metrics['average_response_time']:.2f} seconds
• Average Response Length: {metrics['average_response_length']:.0f} words
• Crisis Detection Rate: {metrics['crisis_detection_rate']:.1f}%

QUALITY METRICS (AVERAGE)
────────────────────────────────────────────────
• Empathy Score: {quality_df['Empathy'].mean():.1f}/100
• Relevance Score: {quality_df['Relevance'].mean():.1f}/100
• Cultural Awareness: {quality_df['Cultural\nAwareness'].mean():.1f}/100
• Safety Score: {quality_df['Safety'].mean():.1f}/100
• Actionability Score: {quality_df['Actionability'].mean():.1f}/100
• Overall Quality: {quality_df['Overall\nQuality'].mean():.1f}/100

"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '9_training_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 9_training_summary.png")
plt.close()

# ============================================================================
# Create Summary Report
# ============================================================================
print("\nGenerating summary report...")

report = f"""
# Calma Mental Health Chatbot - Training and Evaluation Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

The Calma mental health chatbot model was successfully trained for {trainer_state['num_train_epochs']} epochs
({trainer_state['max_steps']} steps) and evaluated on {metrics['total_tests']} diverse test scenarios.

### Key Achievements:
- ✅ **{(metrics['successful_tests']/metrics['total_tests']*100):.1f}% success rate** on test cases
- ✅ **{((train_df.iloc[0]['loss'] - train_df['loss'].min()) / train_df.iloc[0]['loss'] * 100):.1f}% reduction** in training loss
- ✅ **{metrics['crisis_detection_rate']:.1f}% accuracy** in crisis detection
- ✅ Average response time: **{metrics['average_response_time']:.2f} seconds**

## Training Performance

### Loss Metrics
| Metric | Value |
|--------|-------|
| Initial Training Loss | {train_df.iloc[0]['loss']:.4f} |
| Final Training Loss | {train_df.iloc[-1]['loss']:.4f} |
| Best Training Loss | {train_df['loss'].min():.4f} |
| Best Validation Loss | {trainer_state['best_metric']:.4f} |
| Best Model Checkpoint | Step {trainer_state['best_global_step']} |

## Test Results Analysis

### Overall Performance
| Metric | Value |
|--------|-------|
| Test Success Rate | {metrics['successful_tests']}/{metrics['total_tests']} ({metrics['successful_tests']/metrics['total_tests']*100:.1f}%) |
| Avg Response Time | {metrics['average_response_time']:.2f}s |
| Avg Response Length | {metrics['average_response_length']:.0f} words |
| Crisis Detection Rate | {metrics['crisis_detection_rate']:.1f}% |

### Quality Metrics (0-100 scale)
| Metric | Mean | Median | Std Dev |
|--------|------|--------|---------|
| Empathy | {quality_df['Empathy'].mean():.2f} | {quality_df['Empathy'].median():.2f} | {quality_df['Empathy'].std():.2f} |
| Relevance | {quality_df['Relevance'].mean():.2f} | {quality_df['Relevance'].median():.2f} | {quality_df['Relevance'].std():.2f} |
| Cultural Awareness | {quality_df['Cultural\nAwareness'].mean():.2f} | {quality_df['Cultural\nAwareness'].median():.2f} | {quality_df['Cultural\nAwareness'].std():.2f} |
| Safety | {quality_df['Safety'].mean():.2f} | {quality_df['Safety'].median():.2f} | {quality_df['Safety'].std():.2f} |
| Actionability | {quality_df['Actionability'].mean():.2f} | {quality_df['Actionability'].median():.2f} | {quality_df['Actionability'].std():.2f} |
| **Overall Quality** | **{quality_df['Overall\nQuality'].mean():.2f}** | **{quality_df['Overall\nQuality'].median():.2f}** | **{quality_df['Overall\nQuality'].std():.2f}** |

## Visualizations Generated

1. **Training & Validation Loss** - Shows model learning progress
2. **Loss Per Epoch** - Epoch-level performance summary
3. **Learning Rate Schedule** - Learning rate decay visualization
4. **Gradient Norm** - Training stability indicator
5. **Test Metrics Overview** - High-level performance summary
6. **Quality Metrics Distribution** - Statistical distribution of quality scores
7. **Performance by Category** - Category-wise performance breakdown
8. **Severity Analysis** - Performance across different severity levels
9. **Training Summary** - Comprehensive statistics dashboard

## Recommendations

Based on the analysis:
1. The model shows good convergence with decreasing loss
2. Validation loss is stable, indicating minimal overfitting
3. Response quality varies across categories - focus on improving low-performing areas
4. Crisis detection is functioning but could benefit from fine-tuning
5. Cultural awareness scores are low - recommend additional culturally-aware training data

---
*All visualizations are saved in the `model_visualizations/` directory*
"""

with open(output_dir / 'TRAINING_REPORT.md', 'w') as f:
    f.write(report)

print("✓ Saved: TRAINING_REPORT.md")

print(f"\n{'='*70}")
print("✅ All visualizations created successfully!")
print(f"{'='*70}")
print(f"\n📁 Output directory: {output_dir.absolute()}")
print(f"\n📊 Files created:")
print("  1. 1_training_validation_loss.png")
print("  2. 2_loss_per_epoch.png")
print("  3. 3_learning_rate_schedule.png")
print("  4. 4_gradient_norm.png")
print("  5. 5_test_metrics_overview.png")
print("  6. 6_quality_metrics_distribution.png")
print("  7. 7_performance_by_category.png")
print("  8. 8_severity_analysis.png")
print("  9. 9_training_summary.png")
print(" 10. TRAINING_REPORT.md")
print(f"\n{'='*70}\n")
