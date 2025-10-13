# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calma AI is a culturally-aware psychological health chatbot built on Llama-3.2-3B-Instruct with LoRA fine-tuning. The project is designed to provide culturally sensitive mental health support with conversation memory and psychological evaluation capabilities.

## Architecture

### Core Components

- **FastAPI Service** (`src/inference.py`): Main API endpoint with cultural profile management and conversation memory
- **Training Pipeline** (`src/model_training.py`): LoRA fine-tuning implementation for Llama models
- **Data Processing** (`src/data_processing.py`): Cultural context-aware data preprocessing with tokenization
- **Model Evaluation** (`src/model_evaluation.py`, `src/model_evaluation_cpu.py`): Performance evaluation with psychological health scenarios

### Key Data Models

- `CulturalProfile`: Age group, location, education, ethnicity, religion, family structure, respect level, economic status
- `ConversationMemory`: Previous topics, relationship strength, trust level, conversation history
- `ChatRequest/ChatResponse`: Main API request/response schemas with cultural context integration

### Directory Structure

- `src/`: Core Python modules for training, inference, and evaluation
- `configs/`: YAML configuration files for training parameters
- `data/raw/`: Training data in JSON format
- `data/processed/`: Tokenized and processed datasets
- `models/`: Trained model checkpoints and adapters
- `wandb/`: Weights & Biases experiment tracking logs

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (if using one)
source calma/bin/activate

# Install dependencies (use requirements from wandb logs as reference)
pip install torch transformers datasets peft accelerate wandb bitsandbytes
```

### Training
```bash
# Train the model with LoRA fine-tuning
python src/model_training.py

# Process training data
python src/data_processing.py
```

### Inference & Evaluation
```bash
# Start the FastAPI server
python src/inference.py

# Run model evaluation
python src/model_evaluation.py

# Run CPU-based evaluation (for systems without GPU)
python src/model_evaluation_cpu.py
```

### Configuration

Training parameters are managed in `configs/training_config.yaml`:
- Base model: meta-llama/Llama-3.2-3B-Instruct
- LoRA config: r=16, alpha=32, dropout=0.1
- Target modules: All attention and MLP projections
- Training: 3 epochs, batch size 2, gradient accumulation 8 steps

## Key Implementation Details

### Cultural Awareness
The system integrates cultural profiles matching a Prisma schema with enums for:
- Age groups: YOUTH, ADULT, ELDER
- Locations: URBAN, RURAL, PERI_URBAN
- Education levels: PRIMARY, SECONDARY, TERTIARY, POSTGRADUATE
- Family structures: NUCLEAR, EXTENDED, SINGLE_PARENT, GUARDIAN

### Memory Management
Conversation memory tracks relationship strength, trust levels, and previous topics to maintain context across sessions.

### Technical Considerations
- Llama models require explicit padding token setup (uses EOS token)
- Left padding for causal language models
- LoRA fine-tuning for memory-efficient training
- BitsAndBytesConfig for 4-bit quantization support
- Weights & Biases integration for experiment tracking