import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import random
from typing import List, Dict
import numpy as np

class CalmaDataPreprocessor:
    """
    Data preprocessing class for Calma chatbot training data.
    Integrates with the existing Prisma schema structure for cultural awareness.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # CRITICAL: Set padding token for Llama models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add padding side for causal LM
        self.tokenizer.padding_side = "left"  # Important for causal language models
        
        # Cultural context mappings matching your schema
        self.age_groups = ["YOUTH", "ADULT", "ELDER"]
        self.locations = ["URBAN", "RURAL", "PERI_URBAN"]
        self.education_levels = ["PRIMARY", "SECONDARY", "TERTIARY", "POSTGRADUATE"]
        self.family_types = ["NUCLEAR", "EXTENDED", "SINGLE_PARENT", "GUARDIAN"]
        self.respect_levels = ["HIGH", "MODERATE", "RELAXED"]
        self.economic_levels = ["LOW", "MIDDLE", "HIGH"]
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load JSON training data."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_cultural_context(self, context: str) -> Dict:
        """
        Extract cultural context from training data to match Prisma schema.
        This helps the model understand the cultural profile structure.
        """
        cultural_profile = {
            "ageGroup": "ADULT",  # Default
            "location": "URBAN",  # Default
            "educationLevel": "SECONDARY",  # Default
            "ethnicBackground": "Shona",  # Default for Zimbabwe
            "familyStructure": "NUCLEAR",  # Default
            "respectLevel": "MODERATE"  # Default
        }
        
        context_lower = context.lower()
        
        # Extract age information
        if any(word in context_lower for word in ["teen", "teenage", "17", "18", "19", "student"]):
            cultural_profile["ageGroup"] = "YOUTH"
        elif any(word in context_lower for word in ["elder", "elderly", "grandmother", "grandfather"]):
            cultural_profile["ageGroup"] = "ELDER"
        
        # Extract location information
        if any(word in context_lower for word in ["rural", "village", "countryside"]):
            cultural_profile["location"] = "RURAL"
        elif any(word in context_lower for word in ["town", "suburb"]):
            cultural_profile["location"] = "PERI_URBAN"
        
        # Extract family structure
        if any(word in context_lower for word in ["extended family", "grandmother", "grandfather", "aunt", "uncle"]):
            cultural_profile["familyStructure"] = "EXTENDED"
        elif any(word in context_lower for word in ["single mother", "single father", "single parent"]):
            cultural_profile["familyStructure"] = "SINGLE_PARENT"
        
        # Extract education level
        if any(word in context_lower for word in ["university", "college", "degree"]):
            cultural_profile["educationLevel"] = "TERTIARY"
        elif any(word in context_lower for word in ["primary school", "grade"]):
            cultural_profile["educationLevel"] = "PRIMARY"
        
        return cultural_profile
    
    def format_chat_template(self, instruction: str, context: str, response: str) -> str:
        """
        Format data according to Llama chat template with enhanced cultural context.
        Integrates with the database schema structure.
        """
        # Extract cultural context
        cultural_context = self.extract_cultural_context(context)
        
        # Enhanced system prompt with cultural awareness
        system_prompt = f"""You are Calma, a culturally-aware psychological health assistant designed for African users, specifically Zimbabwean contexts. You understand and respect:

- Ubuntu philosophy and community-oriented approaches
- Traditional African family structures and respect for elders
- Zimbabwean cultural values and communication styles
- The importance of family, community, and spiritual well-being
- Economic realities and challenges faced by Zimbabweans

You provide empathetic, practical advice that incorporates these cultural values. Your responses should be supportive, non-judgmental, and culturally sensitive. When appropriate, reference community support, family wisdom, traditional coping mechanisms, and Ubuntu principles.

User Cultural Profile: {json.dumps(cultural_context, indent=2)}"""
        
        # Enhanced user message with session context
        user_message = f"""Cultural Context: {context}

Current Issue: {instruction}

Please provide culturally-appropriate advice that considers the user's background and incorporates African values where relevant."""
        
        # Format using Llama chat template
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ]
        
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    
    def augment_with_memory_context(self, data: List[Dict]) -> List[Dict]:
        """
        Augment training data with conversation memory patterns.
        Simulates the memory system from your database schema.
        """
        augmented_data = []
        
        for i, item in enumerate(data):
            # Original example
            augmented_data.append(item)
            
            # Create follow-up conversation variants
            if i < len(data) - 1 and "instruction" in item and "instruction" in data[i+1]:
                # Simulate previous session memory
                memory_context = f"Previous conversation history: User previously discussed {item['instruction'][:50]}... Status: {random.choice(['ONGOING', 'IMPROVING'])}. "

                follow_up_item = {
                    "instruction": f"Following up on our previous conversation, {data[i+1]['instruction']}",
                    "context": memory_context + data[i+1].get("context", ""),
                    "response": f"I remember we discussed similar challenges before. {data[i+1]['response']}"
                }

                if len(augmented_data) < len(data) * 1.3:  # Limit augmentation
                    augmented_data.append(follow_up_item)
        
        return augmented_data
    
    def add_session_topics(self, examples: List[Dict]) -> List[Dict]:
        """Add session topic classification to training examples."""
        common_topics = [
            "family_conflict", "work_stress", "academic_pressure",
            "relationship_issues", "financial_worry", "health_anxiety",
            "career_uncertainty", "social_isolation", "cultural_identity"
        ]

        for example in examples:
            # Handle both data formats: instruction/context/response and prompt/response
            text_to_analyze = ""
            if "instruction" in example:
                text_to_analyze = example["instruction"].lower()
            elif "prompt" in example:
                text_to_analyze = example["prompt"].lower()
            else:
                print(f"Warning: Example missing both 'instruction' and 'prompt' keys: {example.keys()}")
                continue

            # Simple topic classification based on keywords
            detected_topics = []
            
            if any(word in text_to_analyze for word in ["family", "parent", "mother", "father"]):
                detected_topics.append("family_conflict")
            if any(word in text_to_analyze for word in ["work", "job", "boss", "colleague"]):
                detected_topics.append("work_stress")
            if any(word in text_to_analyze for word in ["school", "exam", "study", "university"]):
                detected_topics.append("academic_pressure")
            if any(word in text_to_analyze for word in ["money", "financial", "afford", "expensive"]):
                detected_topics.append("financial_worry")
            
            if not detected_topics:
                detected_topics = [random.choice(common_topics)]
            
            # Add topic context to the training example
            example["detected_topics"] = detected_topics
            
        return examples
    
    def tokenize_function(self, examples):
        """Tokenize the formatted text for training with proper padding."""
        # Tokenize with consistent max_length and padding strategy
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding="max_length",  # Use max_length padding for consistency
            return_tensors=None,
            add_special_tokens=True
        )
        
        # For causal language modeling, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        # Replace padding token id's in labels with -100 so they're ignored by loss
        labels = model_inputs["labels"]
        for i in range(len(labels)):
            labels[i] = [
                -100 if token == self.tokenizer.pad_token_id else token 
                for token in labels[i]
            ]
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def prepare_dataset(self, file_path: str, test_size: float = 0.2, augment_data: bool = True) -> DatasetDict:
        """
        Load, format, and split dataset for training with cultural awareness.
        """
        # Load raw data
        raw_data = self.load_data(file_path)
        print(f"Loaded {len(raw_data)} examples")
        
        # Add session topics
        raw_data = self.add_session_topics(raw_data)
        
        # Augment with memory context if requested
        if augment_data:
            raw_data = self.augment_with_memory_context(raw_data)
            print(f"Augmented to {len(raw_data)} examples with memory context")
        
        # Format data with enhanced cultural context
        formatted_data = []
        for item in raw_data:
            # Normalize data structure - handle both formats
            if "instruction" in item:
                instruction = item["instruction"]
                context = item.get("context", "")
                response = item["response"]
            elif "prompt" in item:
                instruction = item["prompt"]
                context = ""  # prompt format doesn't have separate context
                response = item["response"]
            else:
                print(f"Skipping item with unknown format: {item.keys()}")
                continue

            formatted_text = self.format_chat_template(
                instruction,
                context,
                response
            )
            formatted_data.append({
                "text": formatted_text,
                "topics": item.get("detected_topics", [])
            })
        
        # Convert to dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Split into train/validation
        dataset = dataset.train_test_split(test_size=test_size, seed=42)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=1,  # Use single process for stability
            remove_columns=dataset["train"].column_names
        )

        print(f"Training examples: {len(tokenized_dataset['train'])}")
        print(f"Validation examples: {len(tokenized_dataset['test'])}")
        print(f"Final dataset columns: {tokenized_dataset['train'].column_names}")
        
        # Verify data format
        sample = tokenized_dataset['train'][0]
        print(f"Sample data shape - input_ids: {len(sample['input_ids'])}, labels: {len(sample['labels'])}")

        return tokenized_dataset

    def save_dataset(self, dataset: DatasetDict, output_path: str):
        """Save the processed dataset to disk."""
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")


# Usage example
if __name__ == "__main__":
    preprocessor = CalmaDataPreprocessor()
    dataset = preprocessor.prepare_dataset("data/raw/training_data.json", augment_data=True)
    
    # Save processed dataset
    preprocessor.save_dataset(dataset, "data/processed/tokenized_dataset")
    print("Dataset preprocessing completed and saved!")