# Calma AI Inference Service

A lightweight FastAPI microservice for AI inference with cultural awareness, designed specifically for Zimbabwean mental health support. This service provides pure AI inference capabilities while your NestJS backend handles all business logic, database operations, and user management.

## Features

- **Pure AI Inference**: Stateless service focused solely on model inference
- **Cultural Awareness**: Built-in Zimbabwean cultural context and sensitivity
- **Mood Detection**: Automatic emotional tone analysis with confidence scoring
- **Resource Recommendations**: Context-aware mental health resource suggestions
- **Performance Optimized**: Efficient model loading with GPU/CPU fallback
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to the service directory
cd calma-ai-service

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (ensure MODEL_PATH points to your fine-tuned model)
nano .env
```

Key configurations:
- `MODEL_PATH`: Path to your fine-tuned Calma model (default: `models/calma-final`)
- `PORT`: Service port (default: 8000)
- `CORS_ORIGINS`: Allowed origins for your NestJS backend

### 3. Run the Service

```bash
# Development mode with auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
python -m app.main
```

### 4. Verify Installation

```bash
# Check health status
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

## API Endpoints

### Main Endpoints

- **POST `/infer`** - AI inference with cultural analysis
- **GET `/health`** - Service and model health check
- **GET `/model-info`** - Detailed model information
- **GET `/cultural-guidelines`** - Cultural awareness information

### Example Request

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I am feeling overwhelmed with my studies and family pressure",
    "parameters": {
      "temperature": 0.8,
      "max_tokens": 200
    }
  }'
```

### Example Response

```json
{
  "response": "I understand you're feeling overwhelmed with both studies and family expectations...",
  "metadata": {
    "mood_detected": "negative",
    "confidence": 0.85,
    "emotional_intensity": 7,
    "response_time_ms": 1250.5,
    "model_version": "calma-v1"
  },
  "suggested_resources": ["stress_management", "study_techniques", "family_counseling"],
  "cultural_elements_detected": ["family_support", "respect_hierarchy"],
  "quality_metrics": {
    "cultural_awareness_score": 0.8,
    "empathy_score": 0.9,
    "response_length_category": "medium"
  }
}
```

## Integration with NestJS

This service is designed to integrate seamlessly with your NestJS backend:

### NestJS Service Example

```typescript
// In your NestJS chat service
async generateAIResponse(message: string, context?: string) {
  const response = await this.httpService.post('http://localhost:8000/infer', {
    message,
    context,
    parameters: {
      temperature: 0.8,
      max_tokens: 200
    }
  }).toPromise();

  return response.data;
}
```

### Architecture Benefits

- **Clean Separation**: NestJS handles business logic, this service handles AI inference
- **Scalability**: Can be scaled independently based on inference load
- **Stateless**: No database connections or session management
- **Fast**: Optimized for quick inference responses

## Model Requirements

### Supported Models
- Fine-tuned Llama 3.2-3B models with LoRA adapters
- Models trained on cultural context for Zimbabwean mental health

### File Structure
Your model directory should contain:
```
models/calma-final/
├── adapter_config.json
├── adapter_model.safetensors
└── README.md
```

### Performance Requirements
- **GPU**: 6GB+ VRAM recommended for optimal performance
- **CPU**: Fallback mode available (slower inference)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 3GB+ for model files

## Configuration Options

### Model Settings
- `MODEL_PATH`: Path to fine-tuned model
- `BASE_MODEL_NAME`: Base Llama model identifier
- `DEVICE_MAP`: Device allocation strategy
- `TORCH_DTYPE`: Model precision (float16/float32)

### Inference Settings
- `MAX_TOKENS`: Maximum response length (10-1024)
- `TEMPERATURE`: Response randomness (0.1-2.0)
- `TOP_P`: Nucleus sampling parameter (0.1-1.0)
- `INFERENCE_TIMEOUT`: Request timeout in seconds

### Performance Settings
- `MAX_MEMORY_MB`: Memory usage limit
- Auto-fallback from GPU to CPU
- 4-bit quantization for GPU inference
- Efficient model caching

## Monitoring and Health

### Health Checks
- **Service Status**: Overall service health
- **Model Status**: Model loading and readiness
- **Memory Usage**: Current memory consumption
- **Uptime**: Service availability metrics

### Logging
- Structured logging with timestamps
- Request/response tracking
- Error tracking with stack traces
- Performance metrics

### Error Handling
- Graceful model loading failures
- Timeout handling for long inference
- Structured error responses
- Request ID tracking

## Cultural Awareness Features

### Built-in Cultural Context
- **Ubuntu Philosophy**: Interconnectedness and community support
- **Family Dynamics**: Respect for elders and family hierarchy
- **Language Sensitivity**: English, Shona, and Ndebele considerations
- **Economic Context**: Resource limitations and practical solutions

### Cultural Element Detection
- Family support patterns
- Community help references
- Traditional healing mentions
- Economic stress indicators
- Ubuntu values
- Respect hierarchy

### Resource Recommendations
- Stress management techniques
- Family counseling approaches
- Financial guidance
- Study techniques
- Cultural and spiritual guidance
- Community-based support

## Development

### Code Structure
```
app/
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── models/
│   └── schemas.py       # Pydantic request/response models
└── services/
    ├── model_service.py     # Model loading and caching
    ├── inference_service.py # AI inference with cultural prompts
    └── analysis_service.py  # Mood and cultural analysis
```

### Key Services
- **ModelService**: Handles model loading, caching, and memory management
- **InferenceService**: Manages AI generation with cultural context
- **AnalysisService**: Analyzes mood, cultural elements, and resource needs

### Adding Features
1. **New Analysis Patterns**: Add to `analysis_service.py`
2. **Cultural Guidelines**: Modify system prompt in `inference_service.py`
3. **API Endpoints**: Add to `main.py` with proper schemas

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "app.main"]
```

### Production Considerations
- Use ASGI server (Gunicorn + Uvicorn workers)
- Configure proper logging and monitoring
- Set up health check endpoints for load balancers
- Use environment variables for all configuration
- Implement proper secret management

## Troubleshooting

### Model Loading Issues
- Check `MODEL_PATH` in `.env` file
- Verify model files exist and are accessible
- Check available GPU memory
- Review logs for specific error messages

### Performance Issues
- Monitor memory usage via `/health` endpoint
- Adjust `MAX_TOKENS` and `TEMPERATURE` parameters
- Consider CPU fallback if GPU memory insufficient
- Check inference timeout settings

### Integration Issues
- Verify CORS settings for NestJS integration
- Check network connectivity between services
- Review request/response format compatibility
- Monitor response times and adjust timeouts

## License

This project is part of the Calma mental health platform.