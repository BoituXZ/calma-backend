# Two-Model System Implementation

## Overview

Calma now uses TWO separate AI models for optimal conversation quality:

1. **Casual Model** (Base Llama 3.2-3B) - For natural, friendly conversation
2. **Therapeutic Model** (Fine-tuned with LoRA) - For mental health support

## How It Works

### Automatic Model Selection

The system automatically chooses the right model based on:

**Use Casual Model When:**
- Conversation stage is early (messages 1-8)
- No distress detected
- Greeting or light chat detected
- Explicit "stay casual" instruction from context

**Use Therapeutic Model When:**
- User expresses moderate or severe distress
- Conversation is established (9+ messages) and user seeks support
- User explicitly asks for help with problems

### Model Routing Logic

```
Message → NestJS analyzes (stage + distress)
       → Builds context with markers
       → AI service reads markers
       → Selects appropriate model
       → Generates response
```

## Architecture Changes

### Model Service (`model_service.py`)
- Loads BOTH models simultaneously
- `casual_model`: Base Llama (no LoRA)
- `therapeutic_model`: Base Llama + LoRA adapters
- Shared tokenizer for efficiency
- `get_pipeline(mode)` returns appropriate pipeline

### Inference Service (`inference_service.py`)
- `_determine_conversation_mode()`: Reads context markers
- Routes to correct model based on conversation needs
- Returns which model was used in metadata

### Chat Service (`chat.service.ts`)
- Detects distress level (none/mild/moderate/severe)
- Tracks conversation stage (early 1-8 vs established 9+)
- Injects clear markers into context for AI service

## Memory Requirements

- **CPU**: ~12-16GB RAM (2x base model + LoRA)
- **GPU**: ~8-10GB VRAM with 4-bit quantization
- Models share same base weights when possible

## Benefits

1. **Natural Conversation**: Base model excels at casual chat
2. **No Therapy Creep**: Casual model isn't trained to find problems
3. **Expert Support**: Therapeutic model provides quality mental health support
4. **Clear Boundaries**: Explicit switching based on user needs
5. **Permanent Fix**: No prompt engineering workarounds needed

## Testing

**Casual Mode Test (Messages 1-8):**
- "hi" → Base model
- "I'm bored" → Base model
- "Music sounds good" → Base model
- Should get natural, friendly responses

**Therapeutic Mode Test:**
- "I'm really struggling with anxiety" → Therapeutic model (distress detected)
- Message 10+ with problem discussion → Therapeutic model (established)

## Monitoring

Check which model was used in API response:
```json
{
  "response": "...",
  "model_used": "casual"  // or "therapeutic"
}
```

## Fallback

If LoRA adapters aren't found, both modes use base model (graceful degradation).
