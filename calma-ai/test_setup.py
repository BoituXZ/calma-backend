#!/usr/bin/env python3
"""
Quick test script to verify the setup before full training.
Tests: imports, dataset loading, tokenization, model loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        import peft
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ Transformers: {transformers.__version__}")
        print(f"  ✓ Datasets: {datasets.__version__}")
        print(f"  ✓ PEFT installed")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_huggingface_auth():
    """Test Hugging Face authentication."""
    print("\nTesting Hugging Face authentication...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"  ✓ Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"  ✗ Not logged in or error: {e}")
        print("  Run: huggingface-cli login")
        return False


def test_dataset_loading():
    """Test loading a small sample from the HF dataset."""
    print("\nTesting dataset loading...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Amod/mental_health_counseling_conversations", split="train[:10]")
        print(f"  ✓ Dataset loaded: {len(ds)} samples")
        print(f"  ✓ Features: {list(ds.features.keys())}")

        # Show first example
        if len(ds) > 0:
            print(f"\n  Sample entry keys: {list(ds[0].keys())}")
            for key, value in ds[0].items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"    {key}: {preview}")

        return True
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False


def test_tokenizer():
    """Test tokenizer setup."""
    print("\nTesting tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test tokenization
        text = "Hello, this is a test message."
        tokens = tokenizer(text, return_tensors="pt")

        print(f"  ✓ Tokenizer loaded")
        print(f"  ✓ Vocab size: {len(tokenizer)}")
        print(f"  ✓ Test tokenization successful: {tokens['input_ids'].shape}")
        return True
    except Exception as e:
        print(f"  ✗ Error with tokenizer: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA/GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")

            # Check memory
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"  ✓ GPU memory: {mem_allocated:.2f}GB / {mem_total:.2f}GB")
        else:
            print(f"  ⚠ CUDA not available, will use CPU (slower)")
            print(f"  CPU training is supported but will be slow")
        return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_data_processor():
    """Test the data processor with a tiny sample."""
    print("\nTesting data processor...")
    try:
        from data_processing_hf import HuggingFaceDataPreprocessor

        processor = HuggingFaceDataPreprocessor()
        print("  ✓ Data processor initialized")

        # Test formatting
        test_context = "I'm feeling anxious about my exams."
        test_response = "It's completely normal to feel anxious before exams. Let's talk about some strategies that might help you manage this anxiety."

        formatted = processor.format_conversation(test_context, test_response)
        print(f"  ✓ Conversation formatting works")
        print(f"  ✓ Formatted length: {len(formatted)} characters")

        return True
    except Exception as e:
        print(f"  ✗ Error with data processor: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("Calma AI Setup Test")
    print("=" * 80)
    print()

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("HuggingFace Auth", test_huggingface_auth()))
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Tokenizer", test_tokenizer()))
    results.append(("CUDA/GPU", test_cuda()))
    results.append(("Data Processor", test_data_processor()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Passed: {passed}/{total}")
    print()

    if passed == total:
        print("✓ All tests passed! You're ready to train.")
        print()
        print("Run the training pipeline with:")
        print("  python train_with_hf_dataset.py --max-samples 100")
        print()
        print("Or for full training:")
        print("  python train_with_hf_dataset.py")
    else:
        print("⚠ Some tests failed. Please fix the issues above before training.")
        if not results[1][1]:  # HuggingFace auth failed
            print()
            print("Quick fix for authentication:")
            print("  1. pip install huggingface_hub")
            print("  2. huggingface-cli login")
            print("  3. Enter your token from: https://huggingface.co/settings/tokens")

    print()
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
