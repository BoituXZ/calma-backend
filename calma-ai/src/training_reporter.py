#!/usr/bin/env python3
"""
Training Reporter - Automatic thesis reporting for Chapter 4

This module automatically logs and generates comprehensive reports for your thesis:
- Training metrics (loss, learning rate, time, memory)
- Evaluation results (test scenarios, response quality)
- Performance metrics (success rate, averages)
- Visualization data (CSV for plotting)
- Thesis-ready Markdown summary

Usage:
    from training_reporter import TrainingReporter

    reporter = TrainingReporter(output_dir="results")
    reporter.log_training_start(model_info, hyperparameters)
    reporter.log_epoch(epoch, metrics)
    reporter.log_evaluation_scenario(scenario_name, input_text, output_text, metrics)
    reporter.save_training_report()
    reporter.save_evaluation_report()
    reporter.generate_chapter4_summary()
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import time


class TrainingReporter:
    """Automatic thesis reporting for model training and evaluation."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize reporter with output directory.

        Args:
            output_dir: Base directory for saving reports (default: "results")
        """
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Training metrics storage
        self.training_metrics = {
            "start_time": None,
            "end_time": None,
            "total_training_time_seconds": 0,
            "epochs": [],
            "model_info": {},
            "hyperparameters": {}
        }

        # Evaluation results storage
        self.evaluation_results = {
            "scenarios": [],
            "summary": {
                "total_scenarios": 0,
                "successful_responses": 0,
                "errors": 0,
                "average_response_length_words": 0,
                "average_response_time_seconds": 0
            }
        }

        print(f"📊 TrainingReporter initialized")
        print(f"📁 Reports will be saved to: {self.run_dir}")

    def log_training_start(self, model_info: Dict[str, Any], hyperparameters: Dict[str, Any]):
        """
        Log training start with model info and hyperparameters.

        Args:
            model_info: Model architecture details (name, params, etc.)
            hyperparameters: Training hyperparameters (lr, epochs, batch_size, etc.)
        """
        self.training_metrics["start_time"] = datetime.now().isoformat()
        self.training_metrics["model_info"] = model_info
        self.training_metrics["hyperparameters"] = hyperparameters

        print(f"✓ Training started: {self.training_metrics['start_time']}")

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for a single epoch.

        Args:
            epoch: Epoch number (1-indexed)
            metrics: Dictionary with keys:
                - train_loss: Training loss
                - val_loss: Validation loss
                - learning_rate: Current learning rate
                - epoch_time_seconds: Time taken for epoch
                - gpu_memory_mb: GPU memory used (optional)
        """
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }

        self.training_metrics["epochs"].append(epoch_data)

        # Auto-save after each epoch (protects against interruptions)
        self._save_training_json()

        print(f"✓ Epoch {epoch} logged: train_loss={metrics.get('train_loss', 'N/A'):.4f}, "
              f"val_loss={metrics.get('val_loss', 'N/A'):.4f}")

    def log_training_end(self):
        """Log training completion time."""
        self.training_metrics["end_time"] = datetime.now().isoformat()

        # Calculate total training time
        if self.training_metrics["start_time"]:
            start = datetime.fromisoformat(self.training_metrics["start_time"])
            end = datetime.fromisoformat(self.training_metrics["end_time"])
            self.training_metrics["total_training_time_seconds"] = (end - start).total_seconds()

        print(f"✓ Training completed: {self.training_metrics['end_time']}")

    def log_evaluation_scenario(
        self,
        scenario_name: str,
        category: str,
        input_text: str,
        output_text: str,
        response_time_seconds: float,
        error: Optional[str] = None
    ):
        """
        Log a single evaluation scenario.

        Args:
            scenario_name: Name of the test scenario
            category: Category (e.g., "greeting", "mental_health", "crisis")
            input_text: User input
            output_text: Model response
            response_time_seconds: Time taken to generate response
            error: Error message if scenario failed (optional)
        """
        word_count = len(output_text.split()) if output_text else 0

        scenario_result = {
            "name": scenario_name,
            "category": category,
            "input": input_text,
            "output": output_text,
            "word_count": word_count,
            "response_time_seconds": response_time_seconds,
            "success": error is None,
            "error": error
        }

        self.evaluation_results["scenarios"].append(scenario_result)

        # Update summary
        self.evaluation_results["summary"]["total_scenarios"] += 1
        if error is None:
            self.evaluation_results["summary"]["successful_responses"] += 1
        else:
            self.evaluation_results["summary"]["errors"] += 1

        status = "✓" if error is None else "✗"
        print(f"{status} Evaluated: {scenario_name} ({word_count} words, {response_time_seconds:.2f}s)")

    def save_training_report(self):
        """Generate and save comprehensive training report (TXT + JSON)."""
        self._save_training_json()
        self._write_training_txt_report()
        print(f"✓ Training reports saved to: {self.run_dir}")

    def save_evaluation_report(self):
        """Generate and save evaluation report (TXT + JSON)."""
        # Calculate averages
        successful = [s for s in self.evaluation_results["scenarios"] if s["success"]]
        if successful:
            avg_words = sum(s["word_count"] for s in successful) / len(successful)
            avg_time = sum(s["response_time_seconds"] for s in successful) / len(successful)
            self.evaluation_results["summary"]["average_response_length_words"] = avg_words
            self.evaluation_results["summary"]["average_response_time_seconds"] = avg_time

        # Calculate success rate
        total = self.evaluation_results["summary"]["total_scenarios"]
        success = self.evaluation_results["summary"]["successful_responses"]
        success_rate = (success / total * 100) if total > 0 else 0
        self.evaluation_results["summary"]["success_rate_percent"] = success_rate

        self._save_evaluation_json()
        self._write_evaluation_txt_report()
        print(f"✓ Evaluation reports saved to: {self.run_dir}")

    def save_performance_metrics(self):
        """Save simplified performance metrics JSON."""
        metrics_file = self.run_dir / "performance_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.evaluation_results["summary"], f, indent=2)
        print(f"✓ Performance metrics saved: {metrics_file}")

    def save_loss_curves_csv(self):
        """Export loss curves as CSV for visualization."""
        csv_file = self.run_dir / "loss_curves.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])

            for epoch_data in self.training_metrics["epochs"]:
                writer.writerow([
                    epoch_data["epoch"],
                    epoch_data.get("train_loss", ""),
                    epoch_data.get("val_loss", ""),
                    epoch_data.get("learning_rate", "")
                ])

        print(f"✓ Loss curves CSV saved: {csv_file}")

    def generate_chapter4_summary(self):
        """Generate thesis-ready Markdown summary (CHAPTER4_SUMMARY.md)."""
        self._write_chapter4_summary()
        print(f"✓ Chapter 4 summary saved: {self.run_dir / 'CHAPTER4_SUMMARY.md'}")

    def save_all_reports(self):
        """Generate and save all report types."""
        print("\n📊 Generating all thesis reports...")
        self.save_training_report()
        self.save_evaluation_report()
        self.save_performance_metrics()
        self.save_loss_curves_csv()
        self.generate_chapter4_summary()
        print(f"\n✅ All reports saved to: {self.run_dir}")
        print(f"\n📖 Open CHAPTER4_SUMMARY.md for thesis-ready content!")

    # Private methods for file writing

    def _save_training_json(self):
        """Save training metrics as JSON."""
        json_file = self.run_dir / "training_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)

    def _save_evaluation_json(self):
        """Save evaluation results as JSON."""
        json_file = self.run_dir / "evaluation_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)

    def _write_training_txt_report(self):
        """Write human-readable training report."""
        report_file = self.run_dir / "training_report.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CALMA AI - TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Model Information
            f.write("1. MODEL INFORMATION\n")
            f.write("-" * 80 + "\n")
            for key, value in self.training_metrics["model_info"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # Hyperparameters
            f.write("2. HYPERPARAMETERS\n")
            f.write("-" * 80 + "\n")
            for key, value in self.training_metrics["hyperparameters"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # Training Timeline
            f.write("3. TRAINING TIMELINE\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Start Time: {self.training_metrics.get('start_time', 'N/A')}\n")
            f.write(f"  End Time: {self.training_metrics.get('end_time', 'N/A')}\n")
            total_time = self.training_metrics.get('total_training_time_seconds', 0)
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            f.write(f"  Total Time: {hours}h {minutes}m\n\n")

            # Epoch-by-Epoch Results
            f.write("4. EPOCH-BY-EPOCH RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'LR':<12} {'Time(s)':<10} {'GPU(MB)':<10}\n")
            f.write("-" * 80 + "\n")

            for epoch_data in self.training_metrics["epochs"]:
                f.write(f"{epoch_data['epoch']:<8} "
                       f"{epoch_data.get('train_loss', 'N/A'):<12.4f} "
                       f"{epoch_data.get('val_loss', 'N/A'):<12.4f} "
                       f"{epoch_data.get('learning_rate', 'N/A'):<12.6f} "
                       f"{epoch_data.get('epoch_time_seconds', 'N/A'):<10.1f} "
                       f"{epoch_data.get('gpu_memory_mb', 'N/A'):<10.1f}\n")

            f.write("\n")

            # Final Results
            if self.training_metrics["epochs"]:
                last_epoch = self.training_metrics["epochs"][-1]
                f.write("5. FINAL EVALUATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Final Training Loss: {last_epoch.get('train_loss', 'N/A'):.4f}\n")
                f.write(f"  Final Validation Loss: {last_epoch.get('val_loss', 'N/A'):.4f}\n")
                if 'train_loss' in last_epoch and 'val_loss' in last_epoch:
                    diff = abs(last_epoch['val_loss'] - last_epoch['train_loss'])
                    f.write(f"  Loss Difference: {diff:.4f}\n")
                    if diff < 0.05:
                        f.write("  ✓ Excellent generalization\n")
                    elif diff < 0.1:
                        f.write("  ✓ Good generalization\n")
                    elif diff < 0.2:
                        f.write("  ⚠ Fair generalization\n")
                    else:
                        f.write("  ⚠ Possible overfitting\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    def _write_evaluation_txt_report(self):
        """Write human-readable evaluation report."""
        report_file = self.run_dir / "evaluation_report.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CALMA AI - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            summary = self.evaluation_results["summary"]
            f.write("1. PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Scenarios: {summary['total_scenarios']}\n")
            f.write(f"  Successful Responses: {summary['successful_responses']}\n")
            f.write(f"  Errors: {summary['errors']}\n")
            f.write(f"  Success Rate: {summary.get('success_rate_percent', 0):.1f}%\n")
            f.write(f"  Average Response Length: {summary.get('average_response_length_words', 0):.1f} words\n")
            f.write(f"  Average Response Time: {summary.get('average_response_time_seconds', 0):.2f} seconds\n")
            f.write("\n")

            # Detailed Results
            f.write("2. DETAILED TEST SCENARIOS\n")
            f.write("-" * 80 + "\n\n")

            for i, scenario in enumerate(self.evaluation_results["scenarios"], 1):
                f.write(f"Scenario {i}: {scenario['name']}\n")
                f.write(f"Category: {scenario['category']}\n")
                f.write(f"Status: {'✓ Success' if scenario['success'] else '✗ Failed'}\n")
                f.write(f"\nUser Input:\n{scenario['input']}\n")
                f.write(f"\nAI Response:\n{scenario['output']}\n")
                f.write(f"\nMetrics:\n")
                f.write(f"  - Word Count: {scenario['word_count']}\n")
                f.write(f"  - Response Time: {scenario['response_time_seconds']:.2f}s\n")
                if scenario['error']:
                    f.write(f"  - Error: {scenario['error']}\n")
                f.write("\n" + "-" * 80 + "\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    def _write_chapter4_summary(self):
        """Write thesis-ready Markdown summary."""
        md_file = self.run_dir / "CHAPTER4_SUMMARY.md"

        with open(md_file, 'w') as f:
            f.write("# Chapter 4: Results and Discussion\n\n")

            # Model Configuration
            f.write("## 4.1 Model Configuration\n\n")
            f.write("### Model Architecture\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for key, value in self.training_metrics["model_info"].items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")

            # Hyperparameters
            f.write("### Training Hyperparameters\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for key, value in self.training_metrics["hyperparameters"].items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")

            # Training Results
            f.write("## 4.2 Training Results\n\n")
            f.write("### Loss Progression\n\n")
            f.write("| Epoch | Training Loss | Validation Loss | Learning Rate |\n")
            f.write("|-------|--------------|-----------------|---------------|\n")
            for epoch_data in self.training_metrics["epochs"]:
                f.write(f"| {epoch_data['epoch']} | "
                       f"{epoch_data.get('train_loss', 'N/A'):.4f} | "
                       f"{epoch_data.get('val_loss', 'N/A'):.4f} | "
                       f"{epoch_data.get('learning_rate', 'N/A'):.6f} |\n")
            f.write("\n")

            # Training Timeline
            f.write("### Training Timeline\n\n")
            f.write(f"- **Start Time**: {self.training_metrics.get('start_time', 'N/A')}\n")
            f.write(f"- **End Time**: {self.training_metrics.get('end_time', 'N/A')}\n")
            total_time = self.training_metrics.get('total_training_time_seconds', 0)
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            f.write(f"- **Total Training Time**: {hours} hours {minutes} minutes\n\n")

            # Final Performance
            if self.training_metrics["epochs"]:
                last_epoch = self.training_metrics["epochs"][-1]
                f.write("### Final Model Performance\n\n")
                f.write(f"- **Final Training Loss**: {last_epoch.get('train_loss', 'N/A'):.4f}\n")
                f.write(f"- **Final Validation Loss**: {last_epoch.get('val_loss', 'N/A'):.4f}\n")
                if 'train_loss' in last_epoch and 'val_loss' in last_epoch:
                    diff = abs(last_epoch['val_loss'] - last_epoch['train_loss'])
                    f.write(f"- **Loss Difference**: {diff:.4f} ")
                    if diff < 0.05:
                        f.write("(Excellent generalization)\n")
                    elif diff < 0.1:
                        f.write("(Good generalization)\n")
                    elif diff < 0.2:
                        f.write("(Fair generalization)\n")
                    else:
                        f.write("(Possible overfitting)\n")
                f.write("\n")

            # Model Evaluation
            f.write("## 4.3 Model Evaluation\n\n")
            summary = self.evaluation_results["summary"]
            f.write("### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Test Scenarios | {summary['total_scenarios']} |\n")
            f.write(f"| Success Rate | {summary.get('success_rate_percent', 0):.1f}% |\n")
            f.write(f"| Average Response Length | {summary.get('average_response_length_words', 0):.1f} words |\n")
            f.write(f"| Average Response Time | {summary.get('average_response_time_seconds', 0):.2f} seconds |\n")
            f.write(f"| Errors Encountered | {summary['errors']} |\n")
            f.write("\n")

            # Sample Responses
            f.write("### Sample Model Responses\n\n")
            # Show first 5 successful scenarios
            successful = [s for s in self.evaluation_results["scenarios"] if s["success"]][:5]
            for scenario in successful:
                f.write(f"**{scenario['name']}** ({scenario['category']})\n\n")
                f.write(f"*User Input*: \"{scenario['input']}\"\n\n")
                f.write(f"*AI Response*: \"{scenario['output'][:200]}{'...' if len(scenario['output']) > 200 else ''}\"\n\n")
                f.write(f"- Word Count: {scenario['word_count']}\n")
                f.write(f"- Response Time: {scenario['response_time_seconds']:.2f}s\n\n")

            # Discussion section (for user to fill in)
            f.write("## 4.4 Discussion\n\n")
            f.write("### Training Performance Analysis\n\n")
            f.write("[Your analysis of training results, loss curves, and convergence behavior]\n\n")

            f.write("### Model Quality Assessment\n\n")
            f.write("[Your analysis of response quality, coherence, and cultural appropriateness]\n\n")

            f.write("### Limitations and Future Work\n\n")
            f.write("[Discuss limitations observed and potential improvements]\n\n")

            f.write("---\n\n")
            f.write(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


if __name__ == "__main__":
    # Example usage
    print("TrainingReporter - Thesis Reporting Module")
    print("=" * 50)
    print("This module is imported by training scripts.")
    print("See THESIS_REPORTING_GUIDE.md for usage examples.")
