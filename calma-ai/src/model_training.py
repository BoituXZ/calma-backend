import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

class CalmaTrainer:
    """Training class for Calma chatbot with LoRA fine-tuning."""
    
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
    
    def setup_model(self):
        """Load and prepare model for training."""
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
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def setup_data_collator(self):
        """Create data collator for dynamic padding."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
    
    def setup_training_args(self, output_dir: str = "./models/calma-finetuned"):
        """Configure training arguments."""
        # Adjust settings based on device
        if torch.cuda.is_available():
            batch_size = 1
            gradient_accumulation = 16
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
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            gradient_checkpointing=True,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=fp16,
            bf16=bf16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            push_to_hub=False,
            optim=optim,
            max_grad_norm=0.3,
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            group_by_length=True,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            label_names=["labels"]
        )
    
    def train(self, dataset_path: str = "data/processed/tokenized_dataset"):
        """Main training function."""
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)
        
        # Setup model
        model = self.setup_model()
        
        # Setup data collator
        data_collator = self.setup_data_collator()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print("Saving model...")
        trainer.save_model("./models/calma-final")
        self.tokenizer.save_pretrained("./models/calma-final")
        
        print("Training completed!")
        
        return trainer
    
    def evaluate(self, trainer):
        """Evaluate the trained model."""
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        return eval_results


if __name__ == "__main__":
    # Create trainer
    trainer = CalmaTrainer()
    
    # Train model
    trained_model = trainer.train()
    
    # Evaluate
    trainer.evaluate(trained_model)