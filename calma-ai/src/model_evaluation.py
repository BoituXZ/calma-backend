import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os
import gc
from datasets import load_from_disk, Dataset

class MemoryEfficientCalmaEvaluator:
    """
    Memory-efficient evaluation for Calma model that handles GPU memory constraints.
    """
    
    def __init__(self, base_model_name: str, fine_tuned_path: str, force_cpu: bool = False):
        self.base_model_name = base_model_name
        self.fine_tuned_path = fine_tuned_path
        self.force_cpu = force_cpu
        
        # Clear GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        
        # Determine device with memory check
        if force_cpu:
            self.device = torch.device("cpu")
            print("🖥️  Forced to use CPU")
        elif torch.cuda.is_available():
            # Check available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_free = gpu_memory - gpu_allocated
            
            print(f"🔍 GPU Memory Status:")
            print(f"   Total: {gpu_memory / 1024**3:.2f} GB")
            print(f"   Allocated: {gpu_allocated / 1024**3:.2f} GB")
            print(f"   Free: {gpu_free / 1024**3:.2f} GB")
            
            # Need at least 4GB free for the model
            if gpu_free > 4 * 1024**3:
                self.device = torch.device("cuda")
                print("✅ Using CUDA")
            else:
                self.device = torch.device("cpu")
                print("⚠️  Insufficient GPU memory, falling back to CPU")
        else:
            self.device = torch.device("cpu")
            print("🖥️  CUDA not available, using CPU")
            
        print(f"🎯 Final device: {self.device}")
        
    def clear_memory(self):
        """Clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        
    def load_model(self):
        """Load the fine-tuned model with memory optimization."""
        print("📚 Loading fine-tuned model...")
        
        # Clear memory first
        self.clear_memory()
        
        # Load tokenizer
        print("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_path)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        try:
            # Load base model with memory optimization
            print("🧠 Loading base model...")
            
            if self.device.type == "cuda":
                # For GPU: use float16 and optimize memory
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    use_cache=False  # Disable KV cache to save memory
                )
            else:
                # For CPU: use float32 but with memory optimization
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    use_cache=False
                )
            
            # Load LoRA weights
            print("🔧 Loading LoRA weights...")
            model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)
            model = model.merge_and_unload()  # Merge LoRA weights
            
            # Move to device
            print(f"🚀 Moving model to {self.device}...")
            model = model.to(self.device)
            model.eval()
            
            # Clear memory after loading
            self.clear_memory()
            
            print("✅ Model loaded successfully!")
            return model, tokenizer
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"💥 GPU out of memory: {e}")
            print("🔄 Retrying with CPU...")
            
            # Force CPU and retry
            self.device = torch.device("cpu")
            return self.load_model_cpu_fallback(tokenizer)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_model_cpu_fallback(self, tokenizer):
        """Fallback to CPU loading."""
        print("🖥️  Loading model on CPU...")
        
        # Clear any GPU memory
        self.clear_memory()
        
        # Load on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)
        model = model.merge_and_unload()
        model = model.to("cpu")
        model.eval()
        
        print("✅ Model loaded on CPU!")
        return model, tokenizer
    
    def generate_response(self, model, tokenizer, prompt: str):
        """Generate response with memory management."""
        try:
            # Prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with memory optimization
            with torch.no_grad():
                if self.device.type == "cuda":
                    # GPU generation with memory management
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,  # Reduced for memory
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        use_cache=False  # Disable KV cache
                    )
                else:
                    # CPU generation (can be more generous with tokens)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            # Clear memory after generation
            del outputs
            self.clear_memory()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def test_basic_functionality(self, model, tokenizer):
        """Test basic model functionality."""
        print("\n🧪 TESTING BASIC FUNCTIONALITY")
        print("=" * 50)
        
        simple_prompts = [
            "Hello, how are you?",
            "I need help with something",
            "Can you assist me?"
        ]
        
        for i, prompt in enumerate(simple_prompts, 1):
            print(f"\n📝 Test {i}/3")
            print(f"User: {prompt}")
            
            response = self.generate_response(model, tokenizer, prompt)
            print(f"Calma: {response}")
            
            if "Error" not in response:
                print("✅ Success")
            else:
                print("❌ Failed")
        
    def evaluate_psychological_responses(self, model, tokenizer):
        """Evaluate psychological health responses."""
        print("\n🧠 PSYCHOLOGICAL HEALTH EVALUATION")
        print("=" * 50)
        
        # Test scenarios relevant to Zimbabwe/Africa context
        test_scenarios = [
            {
                "prompt": "I'm feeling overwhelmed with my university studies and the pressure from my family",
                "category": "Academic & Family Pressure"
            },
            {
                "prompt": "I'm struggling financially and it's affecting my mental health",
                "category": "Financial Stress"
            },
            {
                "prompt": "I feel anxious about my future and finding a job after graduation",
                "category": "Career Anxiety"
            },
            {
                "prompt": "I feel isolated and don't have many friends to talk to",
                "category": "Social Isolation"
            },
            {
                "prompt": "I'm having trouble sleeping because I worry too much about everything",
                "category": "Sleep & Anxiety"
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Scenario {i}/{len(test_scenarios)}: {scenario['category']}")
            print("-" * 30)
            print(f"User: {scenario['prompt']}")
            
            response = self.generate_response(model, tokenizer, scenario['prompt'])
            print(f"Calma: {response}")
            
            # Evaluate response
            evaluation = self.evaluate_response_quality(response)
            
            print(f"📊 Evaluation:")
            print(f"   Words: {evaluation['word_count']}")
            print(f"   Empathy: {evaluation['empathy_score']}/5")
            print(f"   Cultural: {evaluation['cultural_score']}/5") 
            print(f"   Helpful: {evaluation['action_score']}/5")
            print(f"   Overall: {evaluation['overall_score']}")
            
            results.append({
                "category": scenario['category'],
                "prompt": scenario['prompt'],
                "response": response,
                "evaluation": evaluation
            })
        
        return results
    
    def evaluate_response_quality(self, response: str):
        """Evaluate response quality with detailed metrics."""
        if "Error" in response:
            return {
                "error": True,
                "word_count": 0,
                "empathy_score": 0,
                "cultural_score": 0,
                "action_score": 0,
                "overall_score": 0
            }
            
        response_lower = response.lower()
        word_count = len(response.split())
        
        # Empathy indicators
        empathy_words = [
            "understand", "feel", "hear", "difficult", "challenging",
            "sorry", "tough", "hard", "overwhelming", "struggle"
        ]
        
        # Cultural relevance (Ubuntu/African values)
        cultural_words = [
            "family", "community", "together", "support", "ubuntu",
            "elders", "wisdom", "traditional", "ancestors", "collective",
            "shared", "group", "connection"
        ]
        
        # Helpful/actionable advice
        action_words = [
            "try", "consider", "might", "could", "suggest", "recommend",
            "talk", "speak", "reach out", "professional", "counselor",
            "therapy", "help", "support", "steps", "practice"
        ]
        
        # Count occurrences
        empathy_score = min(5, sum(1 for word in empathy_words if word in response_lower))
        cultural_score = min(5, sum(1 for word in cultural_words if word in response_lower))
        action_score = min(5, sum(1 for word in action_words if word in response_lower))
        
        # Overall score
        overall_score = empathy_score + cultural_score + action_score
        
        return {
            "word_count": word_count,
            "empathy_score": empathy_score,
            "cultural_score": cultural_score,
            "action_score": action_score,
            "overall_score": overall_score,
            "length_appropriate": 10 <= word_count <= 200
        }
    
    def create_evaluation_report(self, model, tokenizer):
        """Create comprehensive evaluation report."""
        print("\n📊 CREATING EVALUATION REPORT")
        print("=" * 50)
        
        # Test basic functionality
        self.test_basic_functionality(model, tokenizer)
        
        # Test psychological responses
        psych_results = self.evaluate_psychological_responses(model, tokenizer)
        
        # Calculate statistics
        valid_results = [r for r in psych_results if not r['evaluation'].get('error', False)]
        
        if valid_results:
            stats = {
                "avg_empathy": sum(r['evaluation']['empathy_score'] for r in valid_results) / len(valid_results),
                "avg_cultural": sum(r['evaluation']['cultural_score'] for r in valid_results) / len(valid_results),
                "avg_action": sum(r['evaluation']['action_score'] for r in valid_results) / len(valid_results),
                "avg_overall": sum(r['evaluation']['overall_score'] for r in valid_results) / len(valid_results),
                "avg_length": sum(r['evaluation']['word_count'] for r in valid_results) / len(valid_results)
            }
        else:
            stats = {"avg_empathy": 0, "avg_cultural": 0, "avg_action": 0, "avg_overall": 0, "avg_length": 0}
        
        # Create report
        report = {
            "model_info": {
                "base_model": self.base_model_name,
                "fine_tuned_path": self.fine_tuned_path,
                "device": str(self.device),
                "evaluation_date": torch.backends.cudnn.version() if torch.cuda.is_available() else "CPU"
            },
            "evaluation_summary": {
                "total_tests": len(psych_results),
                "successful_responses": len(valid_results),
                "success_rate": len(valid_results) / len(psych_results) if psych_results else 0,
                **stats
            },
            "detailed_results": psych_results
        }
        
        # Save report
        os.makedirs(self.fine_tuned_path, exist_ok=True)
        report_path = os.path.join(self.fine_tuned_path, "calma_evaluation_report.json")
        
        with open(report_path, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print final summary
        print("\n" + "🎯 FINAL EVALUATION SUMMARY")
        print("=" * 50)
        print(f"🤖 Model: Calma (based on {self.base_model_name})")
        print(f"💻 Device: {self.device}")
        print(f"📈 Success Rate: {report['evaluation_summary']['success_rate']:.1%}")
        print(f"💝 Empathy Score: {stats['avg_empathy']:.1f}/5")
        print(f"🌍 Cultural Relevance: {stats['avg_cultural']:.1f}/5")
        print(f"🎯 Helpfulness: {stats['avg_action']:.1f}/5")
        print(f"⭐ Overall Score: {stats['avg_overall']:.1f}/15")
        print(f"📝 Avg Response Length: {stats['avg_length']:.0f} words")
        print(f"💾 Report saved: {report_path}")
        print("=" * 50)
        
        return report

# Usage
if __name__ == "__main__":
    print("🚀 Starting Memory-Efficient Calma Evaluation...")
    print("=" * 60)
    
    # You can force CPU mode by setting force_cpu=True
    evaluator = MemoryEfficientCalmaEvaluator(
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        fine_tuned_path="models/calma-final",
        force_cpu=False  # Set to True if you want to force CPU
    )
    
    try:
        # Load model
        model, tokenizer = evaluator.load_model()
        
        # Run evaluation
        report = evaluator.create_evaluation_report(model, tokenizer)
        
        print("\n🎉 Evaluation completed successfully!")
        print("📄 Check the generated report for detailed results.")
        
    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        print("💡 Try running with force_cpu=True in the constructor")
        import traceback
        traceback.print_exc()