#!/usr/bin/env python3
"""
Complete training pipeline for Calma AI using Hugging Face dataset.

This script:
1. Downloads and processes the mental health counseling dataset
2. Creates proper train/validation/test splits
3. Trains the model with anti-overfitting measures
4. Evaluates on test set

Usage:
    python train_with_hf_dataset.py [--max-samples N] [--epochs N] [--skip-data-prep]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_processing_hf import HuggingFaceDataPreprocessor
from model_training_improved import ImprovedCalmaTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Calma AI model with Hugging Face mental health dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None = all)"
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
        default=1e-4,
        help="Learning rate"
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
        default=0.15,
        help="Dropout rate for LoRA layers"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience (evaluations)"
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation (use existing processed dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/calma-hf-trained",
        help="Output directory for trained model"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Calma AI Training Pipeline - Hugging Face Dataset")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Max samples: {args.max_samples or 'All'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LoRA dropout: {args.lora_dropout}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Skip data prep: {args.skip_data_prep}")
    print()

    dataset_path = "data/processed/hf_mental_health_dataset"

    # Step 1: Data Preparation
    if not args.skip_data_prep:
        print("=" * 80)
        print("STEP 1: Data Preparation")
        print("=" * 80)
        print()

        try:
            preprocessor = HuggingFaceDataPreprocessor()
            dataset = preprocessor.prepare_dataset(
                train_size=0.8,
                val_size=0.1,
                test_size=0.1,
                max_samples=args.max_samples,
                min_response_length=20
            )
            preprocessor.save_dataset(dataset, dataset_path)

            print()
            print("✓ Data preparation completed successfully!")
            print()

        except Exception as e:
            print()
            print(f"✗ Error during data preparation: {e}")
            print()
            print("If you're having authentication issues, try:")
            print("  1. Run: huggingface-cli login")
            print("  2. Enter your Hugging Face token")
            print("  3. Re-run this script")
            print()
            sys.exit(1)
    else:
        print("=" * 80)
        print("STEP 1: Skipping Data Preparation (using existing dataset)")
        print("=" * 80)
        print()

        if not os.path.exists(dataset_path):
            print(f"✗ Error: Dataset not found at {dataset_path}")
            print("Run without --skip-data-prep to create the dataset first.")
            sys.exit(1)

        print(f"✓ Using existing dataset at {dataset_path}")
        print()

    # Step 2: Model Training
    print("=" * 80)
    print("STEP 2: Model Training with Anti-Overfitting Measures")
    print("=" * 80)
    print()

    try:
        trainer_obj = ImprovedCalmaTrainer()
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

    # Step 3: Evaluation
    print("=" * 80)
    print("STEP 3: Model Evaluation")
    print("=" * 80)
    print()

    try:
        results = trainer_obj.evaluate(trained_model, dataset_path)

        print()
        print("=" * 80)
        print("Training Pipeline Completed Successfully!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  Validation Loss: {results['validation']['eval_loss']:.4f}")
        print(f"  Test Loss: {results['test']['eval_loss']:.4f}")
        print(f"  Loss Difference: {results['loss_difference']:.4f}")
        print()
        print(f"Model saved to: {args.output_dir}/final")
        print()

        if results['loss_difference'] > 0.1:
            print("⚠️  Note: Large validation-test loss difference detected.")
            print("   Consider:")
            print("   - Reducing epochs (--epochs 2)")
            print("   - Increasing weight decay (--weight-decay 0.02)")
            print("   - Increasing LoRA dropout (--lora-dropout 0.2)")
        else:
            print("✓ Good generalization achieved!")

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
