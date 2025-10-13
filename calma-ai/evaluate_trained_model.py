#!/usr/bin/env python3
"""
Evaluate the trained tcabanski model and generate thesis reports.

This script:
- Loads your trained model
- Runs comprehensive test scenarios
- Generates all thesis-ready reports

Usage:
    python3 evaluate_trained_model.py
"""

import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training_reporter import TrainingReporter


def load_model(model_path: str, base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """Load the trained model."""
    print(f"Loading model from: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config for GPU
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float32
        )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    print("✓ Model loaded successfully\n")
    return model, tokenizer, device


def run_test_scenarios(model, tokenizer, reporter: TrainingReporter, device):
    """Run comprehensive test scenarios."""
    print("=" * 80)
    print("RUNNING TEST SCENARIOS")
    print("=" * 80 + "\n")

    # Comprehensive test scenarios for thesis
    scenarios = [
        # 1. Greetings / Casual
        {"name": "Greeting - Simple", "input": "Hi", "category": "greeting"},
        {"name": "Greeting - How are you", "input": "Hello, how are you doing?", "category": "greeting"},

        # 2. Fatigue / Physical Symptoms
        {"name": "Fatigue - Simple", "input": "I'm tired", "category": "physical"},
        {"name": "Fatigue - Detailed", "input": "I'm feeling very tired lately and don't have energy for anything", "category": "physical"},

        # 3. Relationship Issues
        {"name": "Relationship - Trust", "input": "My partner said he doesn't trust me", "category": "relationship"},
        {"name": "Relationship - Fighting", "input": "We're fighting all the time and I don't know what to do", "category": "relationship"},
        {"name": "Relationship - Communication", "input": "We can't seem to communicate properly anymore", "category": "relationship"},

        # 4. Family Issues
        {"name": "Family - Conflict", "input": "My family doesn't understand me", "category": "family"},
        {"name": "Family - Parents", "input": "My parents are putting too much pressure on me", "category": "family"},
        {"name": "Family - Cultural", "input": "My elders don't approve of my life choices", "category": "family"},

        # 5. Mental Health / Emotional
        {"name": "Anxiety - General", "input": "I feel anxious all the time", "category": "mental_health"},
        {"name": "Stress - Overwhelmed", "input": "I feel overwhelmed by everything", "category": "mental_health"},
        {"name": "Depression - Hopeless", "input": "I feel hopeless and don't see the point anymore", "category": "mental_health"},
        {"name": "Loneliness", "input": "I feel so alone and isolated", "category": "mental_health"},

        # 6. Work / Academic
        {"name": "Work - Pressure", "input": "The pressure at work is getting to me", "category": "work"},
        {"name": "Academic - Exams", "input": "I'm stressed about my upcoming exams", "category": "academic"},

        # 7. Zimbabwe-specific Cultural Context
        {"name": "Cultural - Ubuntu", "input": "How can I balance modern life with traditional values?", "category": "cultural"},
        {"name": "Cultural - Community", "input": "My community expects certain things from me", "category": "cultural"},

        # 8. Follow-up / Conversational
        {"name": "Follow-up - Simple", "input": "Yes", "category": "conversational"},
        {"name": "Follow-up - Elaborate", "input": "Can you tell me more about that?", "category": "conversational"},
    ]

    model.eval()

    system_prompt = """You are Calma, a culturally-aware mental health support assistant designed for African users, specifically Zimbabwean contexts. You understand and respect:

- Ubuntu philosophy and community-oriented approaches
- Traditional African family structures and respect for elders
- Zimbabwean cultural values and communication styles
- The importance of family, community, and spiritual well-being

You provide empathetic, practical advice that incorporates these cultural values."""

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] Testing: {scenario['name']}")

        try:
            # Build prompt
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario["input"]}
            ]

            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            # Measure time
            start_time = time.time()

            # Generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )

            response_time = time.time() - start_time

            # Decode
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant response
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()

            # Clean up
            response = response.replace("<|eot_id|>", "").strip()

            word_count = len(response.split())

            # Log to reporter
            reporter.log_evaluation_scenario({
                "scenario_name": scenario["name"],
                "category": scenario["category"],
                "user_input": scenario["input"],
                "response": response,
                "response_word_count": word_count,
                "response_time_seconds": response_time,
                "error": None
            })

            print(f"  ✓ Response: {word_count} words in {response_time:.2f}s\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            reporter.log_evaluation_scenario({
                "scenario_name": scenario["name"],
                "category": scenario["category"],
                "user_input": scenario["input"],
                "response": None,
                "response_word_count": 0,
                "response_time_seconds": 0,
                "error": str(e)
            })

    print("=" * 80)
    print("✓ ALL SCENARIOS COMPLETED")
    print("=" * 80 + "\n")


def main():
    print("\n" + "=" * 80)
    print("CALMA AI MODEL EVALUATION - THESIS CHAPTER 4")
    print("=" * 80 + "\n")

    # Model path (use your trained model)
    model_path = "models/calma-tcabanski-final/checkpoint-400"

    # Initialize reporter
    reporter = TrainingReporter(output_dir="results")

    # Model info
    model_info = {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "adapter_path": model_path,
        "evaluation_mode": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    if torch.cuda.is_available():
        model_info["gpu_name"] = torch.cuda.get_device_name(0)

    # Log start
    reporter.log_training_start(model_info, {"mode": "evaluation_only"})

    # Load model
    model, tokenizer, device = load_model(model_path)

    # Run test scenarios
    run_test_scenarios(model, tokenizer, reporter, device)

    # Log end
    reporter.log_training_end()

    # Save all reports
    additional_info = {
        "Model": f"Llama 3.2-3B with LoRA fine-tuning\nAdapter: {model_path}",
        "Dataset": "tcabanski/mental_health_counseling_responses (quality-filtered)",
        "Evaluation": f"Comprehensive test with {len(reporter.evaluation_results['test_scenarios'])} scenarios",
        "Purpose": "Thesis Chapter 4 - Results and Discussion"
    }

    reporter.save_all_reports(additional_info)

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE - REPORTS GENERATED")
    print("=" * 80)
    print(f"\nResults saved to: {reporter.run_dir}\n")
    print("Generated files:")
    print("  📄 evaluation_report.txt      - Detailed scenario results")
    print("  📄 evaluation_results.json    - Structured data")
    print("  📄 performance_metrics.json   - Performance summary")
    print("  📄 CHAPTER4_SUMMARY.md        - Thesis-ready report")
    print("\nYou can now copy these reports into your thesis!\n")


if __name__ == "__main__":
    main()
