"""
Data preprocessing for Hugging Face mental health counseling dataset.
This script addresses overfitting by using a larger, more diverse dataset
with proper train/validation/test splits.
"""

import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import random
from typing import List, Dict
import numpy as np

class HuggingFaceDataPreprocessor:
    """
    Data preprocessing class for Hugging Face mental health counseling dataset.
    Designed to prevent overfitting with proper data handling and splits.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # CRITICAL: Set padding token for Llama models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Add padding side for causal LM
        self.tokenizer.padding_side = "left"

        # Cultural context mappings matching your schema
        self.age_groups = ["YOUTH", "ADULT", "ELDER"]
        self.locations = ["URBAN", "RURAL", "PERI_URBAN"]
        self.education_levels = ["PRIMARY", "SECONDARY", "TERTIARY", "POSTGRADUATE"]
        self.family_types = ["NUCLEAR", "EXTENDED", "SINGLE_PARENT", "GUARDIAN"]
        self.respect_levels = ["HIGH", "MODERATE", "RELAXED"]
        self.economic_levels = ["LOW", "MIDDLE", "HIGH"]

    def load_huggingface_dataset(self):
        """
        Load the mental health counseling dataset from Hugging Face.

        Note: You may need to login first using:
        huggingface-cli login
        """
        print("Loading dataset from Hugging Face...")
        try:
            ds = load_dataset("Amod/mental_health_counseling_conversations")
            print(f"Dataset loaded successfully!")
            print(f"Dataset structure: {ds}")
            return ds
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure you're logged in with: huggingface-cli login")
            raise

    def extract_cultural_context_random(self) -> Dict:
        """
        Generate random cultural context for training diversity.
        Since the HF dataset may not have explicit cultural markers,
        we'll add varied cultural contexts to teach the model flexibility.
        """
        cultural_profile = {
            "ageGroup": random.choice(self.age_groups),
            "location": random.choice(self.locations),
            "educationLevel": random.choice(self.education_levels),
            "ethnicBackground": random.choice(["Shona", "Ndebele", "Other"]),
            "familyStructure": random.choice(self.family_types),
            "respectLevel": random.choice(self.respect_levels)
        }
        return cultural_profile

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove very short or very long texts (likely noise)
        if len(text) < 10 or len(text) > 2000:
            return None

        return text

    def format_conversation(self, context: str, response: str) -> str:
        """
        Format conversation according to Llama chat template with cultural awareness.

        Args:
            context: The user's message/question from the counseling conversation
            response: The counselor's response
        """
        # Generate random cultural context for diversity
        cultural_context = self.extract_cultural_context_random()

        # Enhanced system prompt with cultural awareness
        system_prompt = f"""You are Calma, a culturally-aware psychological health assistant designed for African users, specifically Zimbabwean contexts. You understand and respect:

- Ubuntu philosophy and community-oriented approaches
- Traditional African family structures and respect for elders
- Zimbabwean cultural values and communication styles
- The importance of family, community, and spiritual well-being
- Economic realities and challenges faced by Zimbabweans

You provide empathetic, practical advice that incorporates these cultural values. Your responses should be supportive, non-judgmental, and culturally sensitive. When appropriate, reference community support, family wisdom, traditional coping mechanisms, and Ubuntu principles.

User Cultural Profile: {json.dumps(cultural_context, indent=2)}"""

        # Format using Llama chat template
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
            {"role": "assistant", "content": response}
        ]

        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )

    def filter_quality_examples(self, examples: List[Dict]) -> List[Dict]:
        """
        Filter out low-quality examples to prevent overfitting on noise.

        Criteria:
        - Minimum length requirements
        - No repetitive patterns
        - Coherent responses
        """
        filtered = []

        for ex in examples:
            # Check if required fields exist
            if 'text' not in ex or not ex['text']:
                continue

            text = ex['text']

            # Skip if too short (likely low quality)
            if len(text) < 50:
                continue

            # Skip if too repetitive (check for repeated tokens)
            tokens = text.lower().split()
            if len(tokens) > 0:
                unique_ratio = len(set(tokens)) / len(tokens)
                if unique_ratio < 0.3:  # Too repetitive
                    continue

            filtered.append(ex)

        return filtered

    def tokenize_function(self, examples):
        """
        Tokenize the formatted text for training with proper padding.
        Uses dynamic padding to prevent overfitting on padding tokens.
        Memory-optimized with shorter max_length for small GPUs.
        """
        # Tokenize with truncation but NO padding (we'll pad dynamically during training)
        # Reduced max_length to 384 for memory efficiency on small GPUs (was 512)
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=384,
            truncation=True,
            padding=False,  # Dynamic padding during training
            return_tensors=None,
            add_special_tokens=True
        )

        # For causal language modeling, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def prepare_dataset(
        self,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        max_samples: int = None,
        min_response_length: int = 20
    ) -> DatasetDict:
        """
        Load and prepare the Hugging Face dataset with proper splits.

        Args:
            train_size: Fraction for training (default 0.8)
            val_size: Fraction for validation (default 0.1)
            test_size: Fraction for testing (default 0.1)
            max_samples: Maximum number of samples to use (None = all)
            min_response_length: Minimum response length to include
        """
        # Load dataset from HuggingFace
        hf_dataset = self.load_huggingface_dataset()

        # Get the train split (this dataset typically has only 'train')
        if 'train' in hf_dataset:
            raw_data = hf_dataset['train']
        else:
            # If it's not a DatasetDict, use it directly
            raw_data = hf_dataset

        print(f"Original dataset size: {len(raw_data)}")
        print(f"Dataset features: {raw_data.features}")

        # Limit samples if specified
        if max_samples and max_samples < len(raw_data):
            raw_data = raw_data.shuffle(seed=42).select(range(max_samples))
            print(f"Limited to {max_samples} samples")

        # Process and format the data
        formatted_data = []
        skipped = 0

        for item in raw_data:
            # Extract context and response from the dataset
            # Adjust these field names based on the actual dataset structure
            context = None
            response = None

            # Try different possible field names
            if 'Context' in item:
                context = item['Context']
                response = item.get('Response', '')
            elif 'questionText' in item:
                context = item['questionText']
                response = item.get('answerText', '')
            elif 'input' in item:
                context = item['input']
                response = item.get('output', '')
            elif 'prompt' in item:
                context = item['prompt']
                response = item.get('response', '')

            # Clean the text
            if context:
                context = self.clean_text(context)
            if response:
                response = self.clean_text(response)

            # Skip if cleaning failed or response too short
            if not context or not response:
                skipped += 1
                continue

            if len(response) < min_response_length:
                skipped += 1
                continue

            # Format with chat template
            try:
                formatted_text = self.format_conversation(context, response)
                formatted_data.append({"text": formatted_text})
            except Exception as e:
                print(f"Error formatting conversation: {e}")
                skipped += 1
                continue

        print(f"Formatted {len(formatted_data)} examples (skipped {skipped})")

        # Filter quality examples
        formatted_data = self.filter_quality_examples(formatted_data)
        print(f"After quality filtering: {len(formatted_data)} examples")

        # Convert to dataset
        dataset = Dataset.from_list(formatted_data)

        # Create proper train/val/test splits
        # First split: separate test set
        train_val_test = dataset.train_test_split(
            test_size=test_size,
            seed=42
        )

        # Second split: separate train and validation
        val_fraction = val_size / (train_size + val_size)
        train_val = train_val_test['train'].train_test_split(
            test_size=val_fraction,
            seed=42
        )

        # Create final dataset dict
        final_dataset = DatasetDict({
            'train': train_val['train'],
            'validation': train_val['test'],
            'test': train_val_test['test']
        })

        print(f"\nFinal splits:")
        print(f"  Training examples: {len(final_dataset['train'])}")
        print(f"  Validation examples: {len(final_dataset['validation'])}")
        print(f"  Test examples: {len(final_dataset['test'])}")

        # Tokenize all splits
        tokenized_dataset = final_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=final_dataset["train"].column_names,
            desc="Tokenizing dataset"
        )

        print(f"\nTokenized dataset columns: {tokenized_dataset['train'].column_names}")

        # Verify data format
        sample = tokenized_dataset['train'][0]
        print(f"Sample data shape - input_ids: {len(sample['input_ids'])}, labels: {len(sample['labels'])}")

        return tokenized_dataset

    def save_dataset(self, dataset: DatasetDict, output_path: str):
        """Save the processed dataset to disk."""
        dataset.save_to_disk(output_path)
        print(f"\nDataset saved to {output_path}")

        # Save some statistics
        stats = {
            "train_size": len(dataset['train']),
            "validation_size": len(dataset['validation']),
            "test_size": len(dataset['test']),
            "total_size": len(dataset['train']) + len(dataset['validation']) + len(dataset['test']),
            "tokenizer": self.tokenizer.name_or_path,
            "max_length": 512
        }

        with open(f"{output_path}/stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print("Dataset statistics saved to stats.json")


# Usage example
if __name__ == "__main__":
    print("=" * 60)
    print("Calma AI - Hugging Face Dataset Preprocessing")
    print("=" * 60)
    print()

    # Initialize preprocessor
    preprocessor = HuggingFaceDataPreprocessor()

    # Prepare dataset with proper splits
    # You can adjust max_samples for testing (e.g., max_samples=1000)
    dataset = preprocessor.prepare_dataset(
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        max_samples=None,  # Use all data, or set a limit like 10000
        min_response_length=20
    )

    # Save processed dataset
    output_path = "data/processed/hf_mental_health_dataset"
    preprocessor.save_dataset(dataset, output_path)

    print()
    print("=" * 60)
    print("Dataset preprocessing completed successfully!")
    print("=" * 60)
