import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import json
import numpy as np

def debug_dataset():
    """Comprehensive dataset debugging."""
    print("🚀 Calma Data Format Validation")
    print("=" * 50)
    
    # Test raw data
    print("\n📁 Testing raw data format...")
    try:
        with open("/raw/training_data.json", 'r') as f:
            raw_data = json.load(f)
        print(f"  ✅ Raw data loaded: {len(raw_data)} examples")
        
        # Check structure of first item
        if raw_data:
            first_item = raw_data[0]
            for key in ["instruction", "context", "response", "prompt"]:
                if key in first_item:
                    print(f"  ✅ Found key: {key}")
                    print(f"  📝 Sample {key}: {first_item[key][:100]}...")
    except Exception as e:
        print(f"  ❌ Error loading raw data: {e}")
    
    # Test processed dataset
    print("\n🔍 Testing processed dataset...")
    try:
        dataset = load_from_disk("data/processed/tokenized_dataset")
        print("✅ Dataset loaded successfully")
        
        print("📊 Dataset info:")
        print(f"  Training samples: {len(dataset['train'])}")
        print(f"  Validation samples: {len(dataset['test'])}")
        print(f"  Columns: {dataset['train'].column_names}")
        
        # Check first sample
        sample = dataset['train'][0]
        print("\n🔍 Sample data inspection:")
        print(f"  Keys: {sample.keys()}")
        
        for key in sample.keys():
            value = sample[key]
            if isinstance(value, list):
                print(f"  {key}: type={type(value)}, length={len(value)}")
                print(f"    First few values: {value[:5]}")
                print(f"    Contains padding (-100): {-100 in value if key == 'labels' else 'N/A'}")
            else:
                print(f"  {key}: type={type(value)}, value={value}")
        
        # Check consistency across samples
        print("\n📐 Checking data consistency...")
        lengths = []
        for i in range(min(10, len(dataset['train']))):
            sample = dataset['train'][i]
            length = len(sample['input_ids'])
            lengths.append(length)
        
        if len(set(lengths)) == 1:
            print(f"  ✅ All samples have consistent length: {lengths[0]}")
        else:
            print(f"  ⚠️  Variable lengths found: {set(lengths)}")
        
        # Test data collator
        print("\n🧪 Testing data collator...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            try:
                batch = [dataset['train'][i] for i in range(batch_size)]
                collated = data_collator(batch)
                print(f"  ✅ Batch size {batch_size}: input_ids shape = {collated['input_ids'].shape}")
            except Exception as e:
                print(f"  ❌ Batch size {batch_size} failed: {e}")
        
        print("\n✨ Dataset validation complete!")
        
    except Exception as e:
        print(f"❌ Error loading processed dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()