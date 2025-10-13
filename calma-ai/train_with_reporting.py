#!/usr/bin/env python3
"""
Enhanced training script with comprehensive reporting for thesis Chapter 4.

This script wraps the existing training pipeline and adds:
- Automatic training metrics logging
- Evaluation scenario testing
- Performance metrics calculation
- Visualization data export (CSV)
- Thesis-ready summary reports

Usage:
    python3 train_with_reporting.py [--dataset tcabanski] [--epochs 3]
"""

import argparse
import os
import sys
import time
import torch
import psutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_training_improved import ImprovedCalmaTrainer
from training_reporter import TrainingReporter
from transformers import TrainerCallback


class ReportingCallback(TrainerCallback):
    """Custom callback to capture training metrics for reporting."""

    def __init__(self, reporter: TrainingReporter):
        self.reporter = reporter
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch."""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch."""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        # Get GPU memory if available
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        # Extract metrics from state
        train_loss = None
        val_loss = None
        learning_rate = None

        if state.log_history:
            # Find most recent train and eval losses
            for log in reversed(state.log_history):
                if 'loss' in log and train_loss is None:
                    train_loss = log['loss']
                if 'eval_loss' in log and val_loss is None:
                    val_loss = log['eval_loss']
                if 'learning_rate' in log and learning_rate is None:
                    learning_rate = log['learning_rate']

        # Log epoch metrics
        self.reporter.log_epoch(
            epoch=state.epoch,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "epoch_time_seconds": epoch_time,
                "gpu_memory_mb": gpu_memory_mb,
                "global_step": state.global_step
            }
        )


def evaluate_test_scenarios(model, tokenizer, reporter: TrainingReporter, device):
    """
    Evaluate model on predefined test scenarios.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        reporter: TrainingReporter instance
        device: Device (cuda/cpu)
    """
    print("\n" + "=" * 80)
    print("EVALUATING TEST SCENARIOS")
    print("=" * 80 + "\n")

    # Define test scenarios (add your own scenarios here)
    test_scenarios = [
        {
            "scenario_name": "Greeting",
            "user_input": "Hi, how are you?",
            "category": "casual"
        },
        {
            "scenario_name": "Fatigue/Tiredness",
            "user_input": "I'm feeling very tired lately",
            "category": "symptom"
        },
        {
            "scenario_name": "Relationship Issues",
            "user_input": "I'm having problems with my partner. He said he doesn't trust me.",
            "category": "relationship"
        },
        {
            "scenario_name": "Family Conflict",
            "user_input": "My family doesn't understand what I'm going through",
            "category": "family"
        },
        {
            "scenario_name": "Stress/Anxiety",
            "user_input": "I feel overwhelmed and anxious about everything",
            "category": "mental_health"
        },
        {
            "scenario_name": "Work Pressure",
            "user_input": "The pressure at work is getting to me",
            "category": "work"
        },
        {
            "scenario_name": "Loneliness",
            "user_input": "I feel so alone and isolated",
            "category": "emotional"
        },
        {
            "scenario_name": "Cultural Context",
            "user_input": "My elders don't approve of my choices",
            "category": "cultural"
        }
    ]

    model.eval()

    for scenario in test_scenarios:
        try:
            # Prepare input
            system_prompt = """You are Calma, a culturally-aware mental health support assistant for Zimbabwean communities."""

            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario["user_input"]}
            ]

            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            # Measure response time
            start_time = time.time()

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            response_time = time.time() - start_time

            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()

            # Calculate word count
            word_count = len(response.split())

            # Log scenario results
            reporter.log_evaluation_scenario({
                "scenario_name": scenario["scenario_name"],
                "category": scenario["category"],
                "user_input": scenario["user_input"],
                "response": response,
                "response_word_count": word_count,
                "response_time_seconds": response_time,
                "error": None
            })

        except Exception as e:
            # Log error
            reporter.log_evaluation_scenario({
                "scenario_name": scenario["scenario_name"],
                "category": scenario["category"],
                "user_input": scenario["user_input"],
                "response": None,
                "response_word_count": 0,
                "response_time_seconds": 0,
                "error": str(e)
            })
            print(f"⚠️  Error in scenario '{scenario['scenario_name']}': {e}")

    print("\n✓ Test scenarios completed\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Calma AI with comprehensive reporting for thesis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tcabanski",
        help="Dataset to use (tcabanski, family, combined)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/calma-reported",
        help="Output directory for model"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory for reports and metrics"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (no training)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CALMA AI TRAINING WITH COMPREHENSIVE REPORTING")
    print("=" * 80)
    print()

    # Initialize reporter
    reporter = TrainingReporter(output_dir=args.results_dir)

    # Determine dataset path
    if args.dataset == "tcabanski":
        dataset_path = "data/processed/tcabanski_mental_health"
    elif args.dataset == "family":
        dataset_path = "data/processed/tokenized_dataset"
    else:
        dataset_path = args.dataset

    if not os.path.exists(dataset_path) and not args.eval_only:
        print(f"✗ Dataset not found: {dataset_path}")
        print("Please run data processing first")
        sys.exit(1)

    # Initialize trainer
    trainer_obj = ImprovedCalmaTrainer()

    # Get model info for reporting
    model_info = {
        "model_name": trainer_obj.model_name,
        "device": str(trainer_obj.device),
        "cuda_available": torch.cuda.is_available()
    }

    if torch.cuda.is_available():
        model_info["gpu_name"] = torch.cuda.get_device_name(0)
        model_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Hyperparameters
    hyperparameters = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lora_dropout": args.lora_dropout,
        "lora_rank": 8,
        "epochs": args.epochs,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_sequence_length": 384,
        "dataset": args.dataset,
        "dataset_path": dataset_path
    }

    if not args.eval_only:
        # Setup model and get parameter counts
        model = trainer_obj.setup_model(lora_dropout=args.lora_dropout)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        model_info["trainable_parameters"] = trainable_params
        model_info["total_parameters"] = total_params
        model_info["trainable_percentage"] = round(trainable_params / total_params * 100, 2)

        # Log training start
        reporter.log_training_start(model_info, hyperparameters)

        # Train with reporting callback
        print("\n" + "=" * 80)
        print("TRAINING MODEL")
        print("=" * 80 + "\n")

        try:
            from datasets import load_from_disk
            from transformers import EarlyStoppingCallback

            dataset = load_from_disk(dataset_path)

            # Setup training args
            training_args = trainer_obj.setup_training_args(
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay
            )

            # Create trainer with reporting callback
            from transformers import Trainer

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                tokenizer=trainer_obj.tokenizer,
                data_collator=trainer_obj.setup_data_collator(),
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=5),
                    ReportingCallback(reporter)
                ]
            )

            # Train
            trainer.train()

            # Save model
            print("\n💾 Saving final model...")
            final_output = f"{args.output_dir}/final"
            trainer.save_model(final_output)
            trainer_obj.tokenizer.save_pretrained(final_output)
            print(f"✓ Model saved to: {final_output}\n")

            # Final evaluation
            print("📊 Running final evaluation...")
            val_metrics = trainer.evaluate(eval_dataset=dataset["validation"])
            test_metrics = trainer.evaluate(eval_dataset=dataset["test"])

            final_metrics = {
                "validation_loss": val_metrics["eval_loss"],
                "test_loss": test_metrics["eval_loss"],
                "loss_difference": abs(val_metrics["eval_loss"] - test_metrics["eval_loss"])
            }

            # Log training end
            reporter.log_training_end(final_metrics)

            # Evaluate test scenarios
            evaluate_test_scenarios(model, trainer_obj.tokenizer, reporter, trainer_obj.device)

        except Exception as e:
            print(f"\n✗ Training error: {e}")
            import traceback
            traceback.print_exc()

            # Still log what we have
            reporter.log_training_end()
            sys.exit(1)

    else:
        # Eval-only mode
        print("\n" + "=" * 80)
        print("EVALUATION ONLY MODE")
        print("=" * 80 + "\n")

        # Load existing model
        from peft import PeftModel

        model = trainer_obj.setup_model(lora_dropout=args.lora_dropout)

        # Log as evaluation run
        reporter.log_training_start(model_info, {"mode": "evaluation_only"})

        # Evaluate test scenarios
        evaluate_test_scenarios(model, trainer_obj.tokenizer, reporter, trainer_obj.device)

        reporter.log_training_end()

    # Save all reports
    additional_info = {
        "Dataset Information": f"Dataset: {args.dataset}\nPath: {dataset_path}",
        "Hardware": f"Device: {model_info.get('device')}\nGPU: {model_info.get('gpu_name', 'N/A')}",
        "Training Configuration": "\n".join([f"{k}: {v}" for k, v in hyperparameters.items()])
    }

    reporter.save_all_reports(additional_info)

    print("\n" + "=" * 80)
    print("✅ TRAINING AND REPORTING COMPLETE")
    print("=" * 80)
    print(f"\nAll reports saved to: {reporter.run_dir}")
    print("\nGenerated files:")
    print("  📄 training_report.txt        - Human-readable training log")
    print("  📄 training_metrics.json      - Structured training data")
    print("  📄 evaluation_report.txt      - Test scenario results")
    print("  📄 evaluation_results.json    - Structured evaluation data")
    print("  📄 performance_metrics.json   - Performance summary")
    print("  📄 loss_curves.csv            - Data for plotting")
    print("  📄 CHAPTER4_SUMMARY.md        - Thesis-ready summary")
    print()


if __name__ == "__main__":
    main()
