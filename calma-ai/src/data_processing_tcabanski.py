"""
Data preprocessing for tcabanski mental health counseling dataset.
This script processes 26k+ counseling conversations with quality filtering.
"""

import json
import random
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import List, Dict

class TcabanskiDataPreprocessor:
    """
    Data preprocessing for tcabanski/mental_health_counseling_responses dataset.
    Features:
    - Quality filtering (empathy ≥ 4, appropriateness ≥ 4)
    - Zimbabwe cultural context injection
    - Professional counseling tone preservation
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "left"

        # Cultural context for Zimbabwe
        self.age_groups = ["YOUTH", "ADULT", "ELDER"]
        self.locations = ["URBAN", "RURAL", "PERI_URBAN"]
        self.education_levels = ["PRIMARY", "SECONDARY", "TERTIARY", "POSTGRADUATE"]
        self.family_types = ["NUCLEAR", "EXTENDED", "SINGLE_PARENT", "GUARDIAN"]
        self.respect_levels = ["HIGH", "MODERATE", "RELAXED"]

    def load_huggingface_dataset(self, quality_threshold: int = 4):
        """
        Load tcabanski dataset from Hugging Face with quality filtering.

        Args:
            quality_threshold: Minimum score for empathy and appropriateness (1-5 scale)

        Returns:
            Filtered dataset with high-quality responses
        """
        print(f"Loading tcabanski/mental_health_counseling_responses from Hugging Face...")
        print(f"Quality threshold: empathy ≥ {quality_threshold}, appropriateness ≥ {quality_threshold}")

        try:
            # Load full dataset
            ds = load_dataset("tcabanski/mental_health_counseling_responses", split="train")
            print(f"✓ Dataset loaded: {len(ds)} total examples")

            # Filter by quality scores
            filtered_ds = ds.filter(
                lambda x: (
                    isinstance(x['empathy'], (int, float)) and x['empathy'] >= quality_threshold and
                    isinstance(x['appropriateness'], (int, float)) and x['appropriateness'] >= quality_threshold and
                    isinstance(x['relevance'], (int, float)) and x['relevance'] >= quality_threshold
                )
            )

            print(f"✓ After quality filtering: {len(filtered_ds)} examples")
            print(f"  Filtered out {len(ds) - len(filtered_ds)} low-quality examples")

            return filtered_ds

        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            print("Make sure you're logged in with: huggingface-cli login")
            raise

    def generate_cultural_context(self) -> Dict[str, str]:
        """Generate random cultural context for diversity."""
        return {
            "ageGroup": random.choice(self.age_groups),
            "location": random.choice(self.locations),
            "educationLevel": random.choice(self.education_levels),
            "familyStructure": random.choice(self.family_types),
            "respectLevel": random.choice(self.respect_levels),
            "ethnicBackground": random.choice(["Shona", "Ndebele", "Other"])
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return None

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Basic length checks
        if len(text) < 10 or len(text) > 3000:
            return None

        return text

    def format_conversation(self, question_title: str, question_text: str, answer_text: str) -> str:
        """
        Format counseling Q&A into Llama chat template with cultural awareness.

        Args:
            question_title: Brief title of the question
            question_text: Full question from user
            answer_text: Counselor's response
        """
        # Generate random cultural context for diversity
        cultural_context = self.generate_cultural_context()

        # Build system prompt with cultural awareness
        system_prompt = f"""You are Calma, a culturally-aware mental health support assistant designed for African users, specifically Zimbabwean contexts. You understand and respect:

- Ubuntu philosophy and community-oriented approaches
- Traditional African family structures and respect for elders
- Zimbabwean cultural values and communication styles
- The importance of family, community, and spiritual well-being
- Economic realities and challenges faced by Zimbabweans

You provide empathetic, practical advice that incorporates these cultural values. Your responses should be supportive, non-judgmental, and culturally sensitive.

User Cultural Profile: {json.dumps(cultural_context, indent=2)}"""

        # Build user message (combine title and text if both exist)
        if question_title and question_title.strip() and question_title != question_text[:100]:
            user_message = f"{question_title}\n\n{question_text}"
        else:
            user_message = question_text

        # Format using Llama chat template
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_text}
        ]

        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )

    def tokenize_function(self, examples):
        """
        Tokenize with memory-optimized settings.
        Max length 384 tokens for 5.64GB GPU.
        """
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=384,
            truncation=True,
            padding=False,  # Dynamic padding during training
            return_tensors=None,
            add_special_tokens=True
        )

        # Labels are same as input_ids for causal LM
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    def prepare_dataset(
        self,
        quality_threshold: int = 4,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        max_samples: int = None
    ) -> DatasetDict:
        """
        Load, filter, and prepare the dataset.

        Args:
            quality_threshold: Min score for quality filtering (1-5)
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for testing
            max_samples: Maximum samples to use (None = all)
        """
        # Load and filter dataset
        hf_dataset = self.load_huggingface_dataset(quality_threshold=quality_threshold)

        # Limit samples if specified
        if max_samples and max_samples < len(hf_dataset):
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(max_samples))
            print(f"Limited to {max_samples} samples for testing")

        # Process and format the data
        formatted_data = []
        skipped = 0

        print(f"\nProcessing {len(hf_dataset)} examples...")

        for item in hf_dataset:
            # Extract fields
            question_title = item.get('questionTitle', '')
            question_text = item.get('questionText', '')
            answer_text = item.get('answerText', '')

            # Clean texts
            question_text = self.clean_text(question_text)
            answer_text = self.clean_text(answer_text)

            # Skip if cleaning failed or texts too short
            if not question_text or not answer_text:
                skipped += 1
                continue

            if len(answer_text) < 50:  # Skip very short answers
                skipped += 1
                continue

            # Format with chat template
            try:
                formatted_text = self.format_conversation(
                    question_title, question_text, answer_text
                )
                formatted_data.append({"text": formatted_text})
            except Exception as e:
                print(f"Error formatting conversation: {e}")
                skipped += 1
                continue

        print(f"✓ Formatted {len(formatted_data)} examples (skipped {skipped})")

        # Convert to dataset
        dataset = Dataset.from_list(formatted_data)

        # Create train/val/test splits
        train_val_test = dataset.train_test_split(test_size=test_size, seed=42)
        val_fraction = val_size / (train_size + val_size)
        train_val = train_val_test['train'].train_test_split(test_size=val_fraction, seed=42)

        final_dataset = DatasetDict({
            'train': train_val['train'],
            'validation': train_val['test'],
            'test': train_val_test['test']
        })

        print(f"\n{'='*60}")
        print("Final Dataset Splits:")
        print('='*60)
        print(f"  Training:   {len(final_dataset['train']):,} examples")
        print(f"  Validation: {len(final_dataset['validation']):,} examples")
        print(f"  Test:       {len(final_dataset['test']):,} examples")
        print(f"  Total:      {len(dataset):,} examples")
        print('='*60)

        # Tokenize all splits
        print("\nTokenizing dataset...")
        tokenized_dataset = final_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=final_dataset["train"].column_names,
            desc="Tokenizing"
        )

        print(f"✓ Tokenization complete")
        print(f"  Columns: {tokenized_dataset['train'].column_names}")

        # Verify sample
        sample = tokenized_dataset['train'][0]
        print(f"  Sample length: {len(sample['input_ids'])} tokens")

        return tokenized_dataset

    def save_dataset(self, dataset: DatasetDict, output_path: str):
        """Save processed dataset to disk."""
        dataset.save_to_disk(output_path)
        print(f"\n✓ Dataset saved to: {output_path}")

        # Save statistics
        stats = {
            "train_size": len(dataset['train']),
            "validation_size": len(dataset['validation']),
            "test_size": len(dataset['test']),
            "total_size": len(dataset['train']) + len(dataset['validation']) + len(dataset['test']),
            "tokenizer": self.tokenizer.name_or_path,
            "max_length": 384,
            "dataset_source": "tcabanski/mental_health_counseling_responses",
            "quality_filtered": True
        }

        with open(f"{output_path}/stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print("✓ Statistics saved to stats.json")


if __name__ == "__main__":
    print("=" * 70)
    print("Calma AI - tcabanski Dataset Preprocessing")
    print("=" * 70)
    print()

    # Initialize preprocessor
    preprocessor = TcabanskiDataPreprocessor()

    # Prepare dataset with quality filtering
    dataset = preprocessor.prepare_dataset(
        quality_threshold=4,  # Empathy, appropriateness, relevance ≥ 4
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        max_samples=None  # Use all high-quality data
    )

    # Save processed dataset
    output_path = "data/processed/tcabanski_mental_health"
    preprocessor.save_dataset(dataset, output_path)

    print()
    print("=" * 70)
    print("✓ Dataset preprocessing completed successfully!")
    print("=" * 70)
    print()
    print("Next step:")
    print("  python3 train_tcabanski.py")
    print()
