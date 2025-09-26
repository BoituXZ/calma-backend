import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft.config import LoraConfig
from peft.mapping import get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from datasets import load_from_disk, DatasetDict
from typing import Union, Optional
import wandb
from datetime import datetime

class CalmaModelTrainer:
    """
    Fine-tuning trainer for Calma psychological health chatbot.
    Uses QLoRA for memory-efficient training on consumer hardware.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model_and_tokenizer(self):
        """Load base model with quantization and tokenizer."""
        print("Loading model and tokenizer...")
        
        # Quantization config for memory efficiency
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return model, tokenizer
    
    def setup_lora(self, model):
        """Setup LoRA configuration for parameter-efficient fine-tuning."""
        print("Setting up LoRA...")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank of adaptation
            lora_alpha=32,  # LoRA scaling parameter
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def create_training_arguments(self, output_dir: str = "models/fine_tuned"):
        """Create training arguments for the Trainer."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"calma_llama3.2_3b_{timestamp}"
        
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,  # Small batch size for 6GB VRAM
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
            num_train_epochs=3,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,  # Save memory
            fp16=True,  # Mixed precision training
            optim="paged_adamw_8bit",  # Memory efficient optimizer
            lr_scheduler_type="cosine",
            report_to=["wandb"],
            run_name=run_name,
            save_total_limit=2,  # Keep only 2 best checkpoints
        )
    
    def train(self, dataset_path: str = "data/processed/tokenized_dataset"):
        """Main training function."""
        print("Starting training...")
        
        # Initialize wandb (optional)
        wandb.init(project="calma-chatbot", name="llama3.2-3b-finetuning")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Setup LoRA
        model = self.setup_lora(model)
        
        # Load dataset
        dataset = load_from_disk(dataset_path)

        # Prepare datasets
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"] if "train" in dataset else None
            eval_dataset = dataset["test"] if "test" in dataset else dataset.get("validation", None)
        else:
            # For single dataset, assume it's training data
            train_dataset = dataset
            eval_dataset = None

        if train_dataset is None:
            raise ValueError("No training dataset found in the loaded dataset")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Not masked language modeling
            pad_to_multiple_of=8  # For tensor cores efficiency
        )

        # Training arguments
        training_args = self.create_training_arguments()

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("Training started...")
        trainer.train()
        
        # Save the final model
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save training metrics
        metrics = trainer.state.log_history
        with open(f"{training_args.output_dir}/training_metrics.json", "w") as f:
            import json
            json.dump(metrics, f, indent=2)
        
        print(f"Training completed! Model saved to {training_args.output_dir}")
        
        return trainer

# Usage
if __name__ == "__main__":
    trainer = CalmaModelTrainer()
    trainer.train()