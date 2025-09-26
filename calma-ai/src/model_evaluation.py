import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from datasets import load_from_disk

class CalmaModelEvaluator:
    """
    Evaluation class for the fine-tuned Calma model.
    Includes perplexity, response quality, and cultural relevance metrics.
    """
    
    def __init__(self, base_model_name: str, fine_tuned_path: str):
        self.base_model_name = base_model_name
        self.fine_tuned_path = fine_tuned_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the fine-tuned model for evaluation."""
        print("Loading fine-tuned model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)
        model = model.merge_and_unload()  # Merge LoRA weights with base model
        model.eval()
        
        return model, tokenizer
    
    def calculate_perplexity(self, model, tokenizer, dataset_path: str):
        """Calculate perplexity on validation set."""
        print("Calculating perplexity...")
        
        dataset = load_from_disk(dataset_path)
        val_dataset = dataset["test"]
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataset):
                if i >= 100:  # Limit for speed
                    break

                # Handle different dataset formats
                if isinstance(batch, dict) and "text" in batch:
                    text = batch["text"]
                elif isinstance(batch, dict) and "input_ids" in batch:
                    # Skip if already tokenized
                    continue
                elif isinstance(batch, str):
                    text = batch
                else:
                    # Skip invalid batch
                    continue

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        print(f"Perplexity: {perplexity:.2f}")
        
        return perplexity.item()
    
    def generate_response(self, model, tokenizer, prompt: str, max_length: int = 512):
        """Generate response for a given prompt."""
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        return response.strip()
    
    def evaluate_sample_responses(self, model, tokenizer, test_prompts: list):
        """Evaluate model on sample prompts and return responses."""
        print("Generating sample responses...")
        
        results = []
        for prompt in test_prompts:
            response = self.generate_response(model, tokenizer, prompt)
            results.append({
                "prompt": prompt,
                "response": response
            })
            print(f"Prompt: {prompt[:100]}...")
            print(f"Response: {response[:200]}...\n")
        
        return results
    
    def cultural_relevance_check(self, responses: list):
        """
        Basic check for cultural relevance in responses.
        Looks for Ubuntu principles, community focus, etc.
        """
        ubuntu_keywords = [
            "community", "family", "together", "support",
            "ubuntu", "collective", "shared", "group",
            "elders", "wisdom", "traditional", "cultural"
        ]
        
        relevance_scores = []
        for response in responses:
            response_lower = response["response"].lower()
            score = sum(1 for keyword in ubuntu_keywords if keyword in response_lower)
            relevance_scores.append(score)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        print(f"Average cultural relevance score: {avg_relevance:.2f}")
        
        return relevance_scores
    
    def create_evaluation_report(self, model, tokenizer, dataset_path: str):
        """Create comprehensive evaluation report."""
        print("Creating evaluation report...")
        
        # Test prompts for evaluation
        test_prompts = [
            "I feel overwhelmed with university studies and don't know how to cope.",
            "My family doesn't understand my career choices and it's causing stress.",
            "I'm dealing with anxiety about my future and feel lost.",
            "I have trouble sleeping due to worrying about exams.",
            "I feel isolated and don't have many friends at university."
        ]
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(model, tokenizer, dataset_path)
        
        # Generate sample responses
        sample_responses = self.evaluate_sample_responses(model, tokenizer, test_prompts)
        
        # Check cultural relevance
        cultural_scores = self.cultural_relevance_check(sample_responses)
        
        # Create report
        report = {
            "model_performance": {
                "perplexity": perplexity,
                "average_cultural_relevance": sum(cultural_scores) / len(cultural_scores)
            },
            "sample_responses": sample_responses,
            "cultural_relevance_scores": cultural_scores
        }
        
        # Save report
        with open(f"{self.fine_tuned_path}/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {self.fine_tuned_path}/evaluation_report.json")
        
        return report

# Usage
if __name__ == "__main__":
    evaluator = CalmaModelEvaluator(
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        fine_tuned_path="models/fine_tuned"
    )
    
    model, tokenizer = evaluator.load_model()
    report = evaluator.create_evaluation_report(model, tokenizer, "data/processed/tokenized_dataset")