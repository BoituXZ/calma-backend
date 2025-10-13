#!/usr/bin/env python3
"""
Complete training pipeline for Calma AI using tcabanski dataset.

This script:
1. Loads pre-processed tcabanski dataset (20k+ quality-filtered examples)
2. Trains from scratch using base Llama model (NO previous training)
3. Uses anti-overfitting measures (early stopping, regularization)
4. Saves model for deployment

Usage:
    python3 train_tcabanski.py [--max-samples N] [--epochs N]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_training_improved import ImprovedCalmaTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Calma AI model with tcabanski quality-filtered dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5, conservative for quality)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for L2 regularization"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for LoRA layers"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Early stopping patience (evaluations)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/calma-tcabanski-final",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run evaluation on existing model (no training)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Calma AI Training - tcabanski Quality Dataset")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LoRA dropout: {args.lora_dropout}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  Output directory: {args.output_dir}")
    print()
    print("Training Strategy:")
    print("  ✓ Fresh start from base Meta Llama (no previous training)")
    print("  ✓ 20k+ quality-filtered counseling conversations")
    print("  ✓ Empathy, appropriateness, relevance ≥ 4/5")
    print("  ✓ Zimbabwe cultural context injection")
    print("  ✓ Memory-optimized for 5.64GB GPU")
    print()

    dataset_path = "data/processed/tcabanski_mental_health"

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print("=" * 80)
        print("✗ Dataset not found!")
        print("=" * 80)
        print()
        print(f"Expected location: {dataset_path}")
        print()
        print("Please run data processing first:")
        print("  python3 src/data_processing_tcabanski.py")
        print()
        sys.exit(1)

    # Load dataset info
    try:
        import json
        with open(f"{dataset_path}/stats.json", 'r') as f:
            stats = json.load(f)
        print("Dataset Info:")
        print(f"  Training:   {stats['train_size']:,} examples")
        print(f"  Validation: {stats['validation_size']:,} examples")
        print(f"  Test:       {stats['test_size']:,} examples")
        print(f"  Total:      {stats['total_size']:,} examples")
        print(f"  Source:     {stats['dataset_source']}")
        print(f"  Quality filtered: {stats['quality_filtered']}")
        print()
    except Exception as e:
        print(f"Warning: Could not load dataset stats: {e}")
        print()

    # Initialize trainer
    trainer_obj = ImprovedCalmaTrainer()

    if args.test_only:
        # Just evaluate existing model
        print("=" * 80)
        print("EVALUATION ONLY MODE")
        print("=" * 80)
        print()

        if not os.path.exists(args.output_dir):
            print(f"✗ Model not found at: {args.output_dir}")
            sys.exit(1)

        # Load and evaluate
        from transformers import Trainer
        model = trainer_obj.setup_model(lora_dropout=args.lora_dropout)
        training_args = trainer_obj.setup_training_args(
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )

        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=dataset["validation"],
            tokenizer=trainer_obj.tokenizer,
            data_collator=trainer_obj.setup_data_collator()
        )

        results = trainer_obj.evaluate(trainer, dataset_path)
        print()
        print("=" * 80)
        print("Evaluation Complete")
        print("=" * 80)
        sys.exit(0)

    # Full Training Pipeline
    print("=" * 80)
    print("Starting Training from Base Model")
    print("=" * 80)
    print()

    try:
        trained_model = trainer_obj.train(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lora_dropout=args.lora_dropout,
            early_stopping_patience=args.early_stopping_patience
        )

        print()
        print("✓ Training completed successfully!")
        print()

    except Exception as e:
        print()
        print(f"✗ Error during training: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Evaluation
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print()

    try:
        results = trainer_obj.evaluate(trained_model, dataset_path)

        print()
        print("=" * 80)
        print("Training Pipeline Completed Successfully!")
        print("=" * 80)
        print()
        print("Results Summary:")
        print(f"  Validation Loss: {results['validation']['eval_loss']:.4f}")
        print(f"  Test Loss: {results['test']['eval_loss']:.4f}")
        print(f"  Loss Difference: {results['loss_difference']:.4f}")
        print()
        print(f"Model saved to: {args.output_dir}/final")
        print()

        # Provide feedback
        if results['loss_difference'] < 0.05:
            print("✓ Excellent generalization! Loss difference < 0.05")
        elif results['loss_difference'] < 0.1:
            print("✓ Good generalization! Loss difference < 0.1")
        elif results['loss_difference'] < 0.2:
            print("⚠ Fair generalization. Loss difference 0.1-0.2")
            print("  Consider: More training data or stronger regularization")
        else:
            print("⚠ Possible overfitting detected. Loss difference > 0.2")
            print("  Next steps:")
            print("  - Increase weight decay: --weight-decay 0.02")
            print("  - Increase LoRA dropout: --lora-dropout 0.15")
            print("  - Reduce epochs: --epochs 2")

        print()
        print("Next Steps:")
        print("  1. Update config.py model_path to point to new model")
        print(f"     model_path: str = \"../{args.output_dir}/final\"")
        print("  2. Restart AI service: ./start-calma.sh")
        print("  3. Test with relationship conversation")
        print()

    except Exception as e:
        print()
        print(f"✗ Error during evaluation: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
