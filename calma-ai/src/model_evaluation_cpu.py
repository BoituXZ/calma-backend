import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os
import gc

class CPUOnlyCalmaEvaluator:
    """
    CPU-only evaluation for Calma model to avoid GPU memory issues.
    This will be slower but guaranteed to work.
    """
    
    def __init__(self, base_model_name: str, fine_tuned_path: str):
        self.base_model_name = base_model_name
        self.fine_tuned_path = fine_tuned_path
        
        # Force CPU device and disable CUDA completely
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.device = torch.device("cpu")
        
        print("🖥️  CPU-Only Mode Enabled")
        print(f"🎯 Device: {self.device}")
        
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def load_model(self):
        """Load the fine-tuned model on CPU only."""
        print("📚 Loading fine-tuned model on CPU...")
        
        # Load tokenizer
        print("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_path)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model on CPU
        print("🧠 Loading base model on CPU (this may take a few minutes)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        
        # Ensure model is on CPU
        base_model = base_model.to("cpu")
        
        # Load LoRA weights
        print("🔧 Loading LoRA weights...")
        model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)
        model = model.merge_and_unload()
        
        # Ensure final model is on CPU
        model = model.to("cpu")
        model.eval()
        
        print("✅ Model loaded successfully on CPU!")
        return model, tokenizer
    
    def generate_response(self, model, tokenizer, prompt: str):
        """Generate response using CPU."""
        try:
            print(f"🤔 Thinking about: '{prompt[:50]}...'")
            
            # Prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,
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
            
            print("✅ Response generated")
            return response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return f"Error generating response: {str(e)}"
    
    def evaluate_response_quality(self, response: str):
        """Evaluate response quality with simple metrics."""
        if "Error" in response:
            return {
                "error": True,
                "word_count": 0,
                "empathy_score": 0,
                "cultural_score": 0,
                "action_score": 0,
                "total_score": 0
            }
            
        response_lower = response.lower()
        words = response.split()
        word_count = len(words)
        
        # Define keyword categories
        empathy_keywords = [
            "understand", "feel", "hear", "know", "difficult", "hard", 
            "tough", "challenging", "struggle", "overwhelming", "sorry"
        ]
        
        cultural_keywords = [
            "family", "community", "together", "support", "ubuntu",
            "elders", "wisdom", "traditional", "connection", "shared",
            "collective", "group", "ancestors", "cultural"
        ]
        
        action_keywords = [
            "try", "consider", "might", "could", "suggest", "help",
            "talk", "speak", "reach", "professional", "counselor",
            "therapy", "practice", "steps", "plan", "support"
        ]
        
        # Count keyword matches
        empathy_score = sum(1 for word in empathy_keywords if word in response_lower)
        cultural_score = sum(1 for word in cultural_keywords if word in response_lower)
        action_score = sum(1 for word in action_keywords if word in response_lower)
        
        # Cap scores at 5
        empathy_score = min(5, empathy_score)
        cultural_score = min(5, cultural_score)
        action_score = min(5, action_score)
        
        total_score = empathy_score + cultural_score + action_score
        
        return {
            "word_count": word_count,
            "empathy_score": empathy_score,
            "cultural_score": cultural_score,
            "action_score": action_score,
            "total_score": total_score,
            "length_ok": 10 <= word_count <= 200
        }
    
    def run_basic_tests(self, model, tokenizer):
        """Run basic functionality tests."""
        print("\n🧪 BASIC FUNCTIONALITY TESTS")
        print("=" * 50)
        
        basic_prompts = [
            "Hello, how are you today?",
            "I need some help",
            "Can you assist me with something?"
        ]
        
        for i, prompt in enumerate(basic_prompts, 1):
            print(f"\n📝 Test {i}: Basic Response")
            print(f"Input: {prompt}")
            
            response = self.generate_response(model, tokenizer, prompt)
            print(f"Output: {response}")
            
            if "Error" not in response and len(response.strip()) > 0:
                print("✅ PASS")
            else:
                print("❌ FAIL")
    
    def run_psychological_evaluation(self, model, tokenizer):
        """Run comprehensive psychological health evaluation."""
        print("\n🧠 PSYCHOLOGICAL HEALTH EVALUATION")
        print("=" * 50)
        
        # Test scenarios for mental health chatbot
        scenarios = [
            {
                "id": 1,
                "category": "Academic Stress",
                "prompt": "I'm feeling overwhelmed with my university studies and the pressure is getting to me"
            },
            {
                "id": 2,
                "category": "Family Pressure", 
                "prompt": "My family expects me to support them financially but I'm still a student struggling myself"
            },
            {
                "id": 3,
                "category": "Career Anxiety",
                "prompt": "I'm worried about finding a job after graduation and I feel anxious about my future"
            },
            {
                "id": 4,
                "category": "Social Isolation",
                "prompt": "I feel lonely and isolated, I don't have many close friends to talk to"
            },
            {
                "id": 5,
                "category": "Sleep & Anxiety",
                "prompt": "I can't sleep properly because I keep worrying about everything in my life"
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\n📋 Scenario {scenario['id']}: {scenario['category']}")
            print("-" * 40)
            print(f"User: {scenario['prompt']}")
            
            # Generate response
            response = self.generate_response(model, tokenizer, scenario['prompt'])
            print(f"Calma: {response}")
            
            # Evaluate response
            evaluation = self.evaluate_response_quality(response)
            
            # Display evaluation
            print(f"\n📊 Evaluation:")
            print(f"   Word Count: {evaluation['word_count']}")
            print(f"   Empathy: {evaluation['empathy_score']}/5")
            print(f"   Cultural: {evaluation['cultural_score']}/5")
            print(f"   Helpful: {evaluation['action_score']}/5")
            print(f"   Total: {evaluation['total_score']}/15")
            print(f"   Length OK: {'✅' if evaluation['length_ok'] else '❌'}")
            
            results.append({
                "scenario": scenario,
                "response": response,
                "evaluation": evaluation
            })
        
        return results
    
    def generate_report(self, results):
        """Generate final evaluation report."""
        print("\n📄 GENERATING EVALUATION REPORT")
        print("=" * 50)
        
        # Calculate statistics
        valid_results = [r for r in results if not r['evaluation'].get('error', False)]
        
        if valid_results:
            total_tests = len(results)
            successful = len(valid_results)
            success_rate = successful / total_tests
            
            avg_empathy = sum(r['evaluation']['empathy_score'] for r in valid_results) / len(valid_results)
            avg_cultural = sum(r['evaluation']['cultural_score'] for r in valid_results) / len(valid_results)
            avg_action = sum(r['evaluation']['action_score'] for r in valid_results) / len(valid_results)
            avg_total = sum(r['evaluation']['total_score'] for r in valid_results) / len(valid_results)
            avg_length = sum(r['evaluation']['word_count'] for r in valid_results) / len(valid_results)
            
            # Create report data
            report = {
                "model_info": {
                    "name": "Calma Chatbot",
                    "base_model": self.base_model_name,
                    "fine_tuned_path": self.fine_tuned_path,
                    "device": "CPU",
                    "evaluation_date": "2025"
                },
                "summary": {
                    "total_scenarios": total_tests,
                    "successful_responses": successful,
                    "success_rate": success_rate,
                    "average_empathy_score": avg_empathy,
                    "average_cultural_score": avg_cultural,
                    "average_helpfulness_score": avg_action,
                    "average_total_score": avg_total,
                    "average_response_length": avg_length
                },
                "detailed_results": [
                    {
                        "scenario_id": r['scenario']['id'],
                        "category": r['scenario']['category'],
                        "prompt": r['scenario']['prompt'],
                        "response": r['response'],
                        "scores": r['evaluation']
                    }
                    for r in results
                ]
            }
            
            # Save report
            os.makedirs(self.fine_tuned_path, exist_ok=True)
            report_path = os.path.join(self.fine_tuned_path, "calma_evaluation_final.json")
            
            with open(report_path, "w", encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Print summary
            print("\n🎯 FINAL EVALUATION RESULTS")
            print("=" * 50)
            print(f"🤖 Model: Calma (Zimbabwean Mental Health Chatbot)")
            print(f"💻 Device: CPU")
            print(f"📊 Test Scenarios: {total_tests}")
            print(f"✅ Successful Responses: {successful}/{total_tests} ({success_rate:.1%})")
            print(f"💝 Average Empathy Score: {avg_empathy:.1f}/5")
            print(f"🌍 Average Cultural Relevance: {avg_cultural:.1f}/5")
            print(f"🎯 Average Helpfulness: {avg_action:.1f}/5")
            print(f"⭐ Overall Average Score: {avg_total:.1f}/15")
            print(f"📝 Average Response Length: {avg_length:.0f} words")
            print(f"💾 Report saved to: {report_path}")
            print("=" * 50)
            
            return report
        
        else:
            print("❌ No valid results to report")
            return None
    
    def run_full_evaluation(self):
        """Run complete evaluation process."""
        try:
            # Load model
            model, tokenizer = self.load_model()
            
            # Run basic tests
            self.run_basic_tests(model, tokenizer)
            
            # Run psychological evaluation
            results = self.run_psychological_evaluation(model, tokenizer)
            
            # Generate report
            report = self.generate_report(results)
            
            print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
            return report
            
        except Exception as e:
            print(f"\n💥 Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Main execution
if __name__ == "__main__":
    print("🚀 Starting CPU-Only Calma Evaluation")
    print("⚠️  This will be slower but guaranteed to work")
    print("=" * 60)
    
    evaluator = CPUOnlyCalmaEvaluator(
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        fine_tuned_path="models/calma-final"
    )
    
    report = evaluator.run_full_evaluation()
    
    if report:
        print("\n✨ Success! Check the generated report for your Chapter 4 documentation.")
    else:
        print("\n💥 Evaluation failed. Check the error messages above.")