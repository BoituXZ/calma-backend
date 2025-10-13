"""
Improved model training script with anti-overfitting measures.
This version includes:
- Better regularization (weight decay, dropout)
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Proper validation monitoring
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

class ImprovedCalmaTrainer:
    """
    Improved training class for Calma chatbot with LoRA fine-tuning.
    Includes anti-overfitting measures and better monitoring.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # CRITICAL: Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "left"

        # Quantization config for memory efficiency (only if CUDA available)
        if torch.cuda.is_available():
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            self.bnb_config = None

    def setup_model(self, lora_dropout: float = 0.1):
        """
        Load and prepare model for training with improved LoRA configuration.

        Args:
            lora_dropout: Dropout rate for LoRA layers (higher = more regularization)
        """
        print("Loading model...")

        if torch.cuda.is_available() and self.bnb_config:
            # Load model with quantization for GPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )

            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
        else:
            # Load model without quantization for CPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": self.device},
                trust_remote_code=True,
                torch_dtype=torch.float32
            )

        # Configure LoRA with improved anti-overfitting settings
        # Memory-optimized: lower rank for smaller GPUs
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # Reduced rank for memory efficiency (was 16)
            lora_alpha=16,  # Scaling factor (proportional to rank)
            lora_dropout=lora_dropout,  # Increased dropout for regularization
            bias="none",
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            inference_mode=False,
            # Add these for better regularization
            use_rslora=True,  # Rank-stabilized LoRA
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model

    def setup_data_collator(self):
        """
        Create data collator with dynamic padding.
        Dynamic padding helps prevent overfitting on padding tokens.
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )

    def setup_training_args(
        self,
        output_dir: str = "./models/calma-finetuned-improved",
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1
    ):
        """
        Configure training arguments with anti-overfitting measures.

        Args:
            output_dir: Directory for model checkpoints
            num_epochs: Number of training epochs (lower = less overfitting risk)
            learning_rate: Learning rate (lower = more stable, less overfitting)
            weight_decay: L2 regularization strength (higher = more regularization)
            warmup_ratio: Fraction of steps for learning rate warmup
        """
        # Adjust settings based on device
        if torch.cuda.is_available():
            # Memory-optimized settings for small GPUs (< 8GB VRAM)
            batch_size = 1  # Minimum batch size for memory efficiency
            gradient_accumulation = 16  # Higher accumulation to simulate larger batches
            fp16 = False
            bf16 = True
            optim = "paged_adamw_8bit"
        else:
            batch_size = 1
            gradient_accumulation = 4
            fp16 = False
            bf16 = False
            optim = "adamw_torch"

        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            gradient_checkpointing=True,

            # Learning rate schedule
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",  # Cosine decay for smooth convergence
            warmup_ratio=warmup_ratio,

            # Regularization
            weight_decay=weight_decay,  # L2 regularization
            max_grad_norm=1.0,  # Gradient clipping (more conservative)

            # Precision
            fp16=fp16,
            bf16=bf16,

            # Logging and evaluation
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=50,  # Frequent evaluation for early stopping
            save_strategy="steps",
            save_steps=50,  # Save frequently
            save_total_limit=3,  # Keep best 3 checkpoints

            # Early stopping configuration
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Other settings
            report_to="none",  # or "wandb" if you want to use Weights & Biases
            push_to_hub=False,
            optim=optim,
            group_by_length=True,  # Group similar lengths for efficiency
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            label_names=["labels"],

            # Disable data augmentation during eval
            prediction_loss_only=True,
        )

    def train(
        self,
        dataset_path: str = "data/processed/hf_mental_health_dataset",
        output_dir: str = "./models/calma-finetuned-improved",
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        lora_dropout: float = 0.1,
        early_stopping_patience: int = 3
    ):
        """
        Main training function with early stopping.

        Args:
            dataset_path: Path to processed dataset
            output_dir: Output directory for checkpoints
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            lora_dropout: LoRA dropout rate
            early_stopping_patience: Stop after N evaluations without improvement
        """
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)

        print(f"Dataset splits: {dataset.keys()}")
        print(f"Training examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")

        # Setup model with specified dropout
        model = self.setup_model(lora_dropout=lora_dropout)

        # Setup data collator
        data_collator = self.setup_data_collator()

        # Setup training arguments
        training_args = self.setup_training_args(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize trainer with early stopping callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=0.0  # Any improvement counts
                )
            ]
        )

        # Start training
        print("\n" + "=" * 60)
        print("Starting training with anti-overfitting measures:")
        print(f"  - Early stopping patience: {early_stopping_patience}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - LoRA dropout: {lora_dropout}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Max epochs: {num_epochs}")
        print("=" * 60 + "\n")

        trainer.train()

        # Save final model
        print("\nSaving final model...")
        final_output = f"{output_dir}/final"
        trainer.save_model(final_output)
        self.tokenizer.save_pretrained(final_output)

        print(f"\nTraining completed!")
        print(f"Model saved to: {final_output}")

        return trainer

    def evaluate(self, trainer, dataset_path: str = "data/processed/hf_mental_health_dataset"):
        """
        Evaluate the trained model on validation and test sets.

        Args:
            trainer: Trained Trainer object
            dataset_path: Path to dataset
        """
        print("\n" + "=" * 60)
        print("Evaluating model...")
        print("=" * 60 + "\n")

        # Load dataset
        dataset = load_from_disk(dataset_path)

        # Evaluate on validation set
        print("Validation set evaluation:")
        val_results = trainer.evaluate(eval_dataset=dataset["validation"])
        print(f"Validation Loss: {val_results['eval_loss']:.4f}")
        print(f"Validation Perplexity: {torch.exp(torch.tensor(val_results['eval_loss'])):.4f}")

        # Evaluate on test set
        print("\nTest set evaluation:")
        test_results = trainer.evaluate(eval_dataset=dataset["test"])
        print(f"Test Loss: {test_results['eval_loss']:.4f}")
        print(f"Test Perplexity: {torch.exp(torch.tensor(test_results['eval_loss'])):.4f}")

        # Compare validation vs test loss
        loss_diff = abs(val_results['eval_loss'] - test_results['eval_loss'])
        print(f"\nValidation-Test Loss Difference: {loss_diff:.4f}")
        if loss_diff > 0.1:
            print("⚠️  Warning: Large difference between validation and test loss.")
            print("   This might indicate overfitting or domain shift.")
        else:
            print("✓ Good generalization: Similar validation and test performance.")

        return {
            'validation': val_results,
            'test': test_results,
            'loss_difference': loss_diff
        }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Calma AI - Improved Training with Anti-Overfitting")
    print("=" * 60 + "\n")

    # Create trainer
    trainer = ImprovedCalmaTrainer()

    # Train model with anti-overfitting measures
    trained_model = trainer.train(
        dataset_path="data/processed/hf_mental_health_dataset",
        output_dir="./models/calma-finetuned-improved",
        num_epochs=3,  # Can reduce to 2 if still overfitting
        learning_rate=1e-4,  # Slightly lower learning rate
        weight_decay=0.01,  # L2 regularization
        lora_dropout=0.15,  # Increased dropout
        early_stopping_patience=3  # Stop if no improvement for 3 evaluations
    )

    # Evaluate on both validation and test sets
    results = trainer.evaluate(
        trained_model,
        dataset_path="data/processed/hf_mental_health_dataset"
    )

    print("\n" + "=" * 60)
    print("Training and evaluation completed!")
    print("=" * 60)
