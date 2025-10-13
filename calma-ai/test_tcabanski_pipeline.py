#!/usr/bin/env python3
"""
Quick test of the tcabanski training pipeline with a small sample.
This verifies everything works before running full training (which takes hours).

Usage:
    python3 test_tcabanski_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_processing():
    """Test data processing with small sample."""
    print("=" * 70)
    print("TEST 1: Data Processing (100 samples)")
    print("=" * 70)
    print()

    try:
        from data_processing_tcabanski import TcabanskiDataPreprocessor

        # Initialize
        preprocessor = TcabanskiDataPreprocessor()
        print("✓ Preprocessor initialized")

        # Load and process small sample
        dataset = preprocessor.prepare_dataset(
            quality_threshold=4,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            max_samples=100  # Small sample for testing
        )

        print(f"✓ Dataset processed: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")

        # Save
        output_path = "data/processed/tcabanski_test"
        preprocessor.save_dataset(dataset, output_path)
        print(f"✓ Dataset saved to: {output_path}")

        return True, output_path

    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_setup():
    """Test model initialization."""
    print()
    print("=" * 70)
    print("TEST 2: Model Initialization")
    print("=" * 70)
    print()

    try:
        from model_training_improved import ImprovedCalmaTrainer

        # Initialize trainer
        trainer = ImprovedCalmaTrainer()
        print("✓ Trainer initialized")

        # Setup model
        model = trainer.setup_model(lora_dropout=0.1)
        print("✓ Model initialized with LoRA")

        # Check trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        return True

    except Exception as e:
        print(f"✗ Model setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training(dataset_path):
    """Test training with tiny dataset for 1 step."""
    print()
    print("=" * 70)
    print("TEST 3: Training (1 epoch, test run)")
    print("=" * 70)
    print()

    try:
        from model_training_improved import ImprovedCalmaTrainer

        trainer = ImprovedCalmaTrainer()

        # Quick training test (1 epoch)
        print("Running 1 epoch training test (this may take 5-10 minutes)...")
        print()

        trained_model = trainer.train(
            dataset_path=dataset_path,
            output_dir="./models/calma-test",
            num_epochs=1,  # Just 1 epoch for testing
            learning_rate=5e-5,
            weight_decay=0.01,
            lora_dropout=0.1,
            early_stopping_patience=1
        )

        print()
        print("✓ Training completed successfully!")

        # Quick evaluation
        print()
        print("Running evaluation...")
        results = trainer.evaluate(trained_model, dataset_path)

        print()
        print(f"✓ Evaluation completed:")
        print(f"  Validation Loss: {results['validation']['eval_loss']:.4f}")
        print(f"  Test Loss: {results['test']['eval_loss']:.4f}")

        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print()
    print("=" * 70)
    print("Calma AI - tcabanski Pipeline Test")
    print("=" * 70)
    print()
    print("This script tests the complete pipeline with a small sample (100 examples).")
    print("It verifies:")
    print("  1. Data can be downloaded and processed from HuggingFace")
    print("  2. Model can be initialized with LoRA")
    print("  3. Training runs without errors")
    print()
    print("Expected time: ~10-15 minutes")
    print()
    input("Press Enter to continue...")
    print()

    # Test 1: Data Processing
    success, dataset_path = test_data_processing()
    if not success:
        print()
        print("=" * 70)
        print("✗ Pipeline test FAILED at data processing")
        print("=" * 70)
        sys.exit(1)

    # Test 2: Model Setup
    success = test_model_setup()
    if not success:
        print()
        print("=" * 70)
        print("✗ Pipeline test FAILED at model setup")
        print("=" * 70)
        sys.exit(1)

    # Test 3: Training
    print()
    print("WARNING: The next test will train for 1 epoch (~5-10 minutes).")
    response = input("Continue with training test? [y/N]: ")
    if response.lower() != 'y':
        print()
        print("=" * 70)
        print("✓ Pipeline tests 1-2 passed! Skipping training test.")
        print("=" * 70)
        print()
        print("When ready for full training, run:")
        print("  python3 train_tcabanski.py")
        sys.exit(0)

    success = test_training(dataset_path)
    if not success:
        print()
        print("=" * 70)
        print("✗ Pipeline test FAILED at training")
        print("=" * 70)
        sys.exit(1)

    # All tests passed
    print()
    print("=" * 70)
    print("✓ ALL PIPELINE TESTS PASSED!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Delete test data:")
    print("     rm -rf data/processed/tcabanski_test models/calma-test")
    print()
    print("  2. Process full dataset:")
    print("     python3 src/data_processing_tcabanski.py")
    print()
    print("  3. Train model:")
    print("     python3 train_tcabanski.py")
    print()


if __name__ == "__main__":
    main()
