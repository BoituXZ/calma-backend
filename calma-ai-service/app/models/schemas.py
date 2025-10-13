"""Pydantic schemas for request/response validation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class InferenceParameters(BaseModel):
    """Parameters for AI inference."""
    temperature: Optional[float] = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Controls randomness in generation (0.1-2.0)"
    )
    max_tokens: Optional[int] = Field(
        default=256,
        ge=10,
        le=1024,
        description="Maximum number of tokens to generate (10-1024)"
    )
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Nucleus sampling parameter (0.1-1.0)"
    )


class InferenceRequest(BaseModel):
    """Request schema for AI inference endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message for AI to respond to"
    )
    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Previous conversation context if needed"
    )
    parameters: Optional[InferenceParameters] = Field(
        default_factory=InferenceParameters,
        description="Inference parameters"
    )

    @validator('message')
    def validate_message(cls, v):
        """Validate message content."""
        if not v.strip():
            raise ValueError("Message cannot be empty or only whitespace")
        return v.strip()

    @validator('context')
    def validate_context(cls, v):
        """Validate context content."""
        if v is not None:
            return v.strip() if v.strip() else None
        return v


class ResponseMetadata(BaseModel):
    """Metadata about the AI response."""
    mood_detected: str = Field(description="Detected mood: negative, neutral, positive")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in mood detection (0.0-1.0)"
    )
    emotional_intensity: int = Field(
        ge=1,
        le=10,
        description="Emotional intensity on 1-10 scale"
    )
    response_time_ms: float = Field(description="Response generation time in milliseconds")
    model_version: str = Field(description="Version of the model used")
    emotional_indicators: List[str] = Field(
        default_factory=list,
        description="Detected emotional indicators in the message"
    )


class ResponseQualityMetrics(BaseModel):
    """Quality metrics for the AI response."""
    word_count: int = Field(description="Number of words in response")
    sentence_count: int = Field(description="Number of sentences in response")
    cultural_awareness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cultural awareness score (0.0-1.0)"
    )
    empathy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Empathy score (0.0-1.0)"
    )
    response_length_category: str = Field(
        description="Response length category: very_short, short, medium, long, very_long"
    )


class InferenceResponse(BaseModel):
    """Response schema for AI inference endpoint."""
    response: str = Field(description="Generated AI response")
    metadata: ResponseMetadata = Field(description="Response metadata and analysis")
    suggested_resources: List[str] = Field(
        default_factory=list,
        description="Suggested resource types based on analysis"
    )
    cultural_elements_detected: List[str] = Field(
        default_factory=list,
        description="Detected cultural elements in the message"
    )
    quality_metrics: ResponseQualityMetrics = Field(
        description="Quality metrics for the response"
    )
    parameters_used: InferenceParameters = Field(
        description="Parameters actually used for inference"
    )


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(description="Service status: healthy, unhealthy, loading")
    timestamp: datetime = Field(default_factory=datetime.now)
    service_name: str = Field(description="Name of the service")
    service_version: str = Field(description="Version of the service")
    model_status: str = Field(description="Model loading status")
    model_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model information if loaded"
    )
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="Service uptime in seconds"
    )


class ModelInfoResponse(BaseModel):
    """Response schema for model info endpoint."""
    model_name: str = Field(description="Name of the base model")
    model_path: str = Field(description="Path to the fine-tuned model")
    model_version: str = Field(description="Version identifier")
    device: str = Field(description="Device being used (cpu/cuda)")
    load_time_seconds: Optional[float] = Field(description="Time taken to load model")
    memory_usage_mb: Optional[int] = Field(description="Current memory usage in MB")
    quantization: Optional[bool] = Field(description="Whether quantization is enabled")
    lora_enabled: Optional[bool] = Field(description="Whether LoRA adapters are loaded")
    cultural_guidelines: Dict[str, Any] = Field(
        description="Information about cultural guidelines"
    )
    supported_features: List[str] = Field(
        default_factory=lambda: [
            "mood_detection",
            "cultural_awareness",
            "resource_recommendations",
            "response_quality_analysis"
        ],
        description="List of supported features"
    )


class ErrorResponse(BaseModel):
    """Response schema for error cases."""
    error: str = Field(description="Error type")
    message: str = Field(description="Detailed error message")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(default=None, description="Request ID if available")


class CulturalGuidelinesResponse(BaseModel):
    """Response schema for cultural guidelines endpoint."""
    cultural_focus: str = Field(description="Primary cultural focus")
    languages_considered: List[str] = Field(description="Languages taken into account")
    key_values: List[str] = Field(description="Key cultural values considered")
    contexts: List[str] = Field(description="Different contexts considered")
    approach: str = Field(description="Overall approach description")
    system_prompt_length: int = Field(description="Length of cultural system prompt")
    cultural_patterns: Dict[str, List[str]] = Field(
        description="Patterns used for cultural element detection"
    )


# Request/Response examples for documentation
class SchemaExamples:
    """Example data for API documentation."""

    INFERENCE_REQUEST_EXAMPLE = {
        "message": "I'm feeling overwhelmed with my studies and family pressure",
        "context": "User previously mentioned stress about university exams",
        "parameters": {
            "temperature": 0.8,
            "max_tokens": 200
        }
    }

    INFERENCE_RESPONSE_EXAMPLE = {
        "response": "I understand you're feeling overwhelmed with both studies and family expectations. This is very common, especially in our culture where family support comes with high expectations. Let's think about some practical ways to manage this stress...",
        "metadata": {
            "mood_detected": "negative",
            "confidence": 0.85,
            "emotional_intensity": 7,
            "response_time_ms": 1250.5,
            "model_version": "calma-v1",
            "emotional_indicators": ["overwhelmed", "pressure", "stress"]
        },
        "suggested_resources": ["stress_management", "study_techniques", "family_counseling"],
        "cultural_elements_detected": ["family_support", "respect_hierarchy"],
        "quality_metrics": {
            "word_count": 85,
            "sentence_count": 4,
            "cultural_awareness_score": 0.8,
            "empathy_score": 0.9,
            "response_length_category": "medium"
        },
        "parameters_used": {
            "temperature": 0.8,
            "max_tokens": 200,
            "top_p": 0.9
        }
    }

    HEALTH_CHECK_EXAMPLE = {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z",
        "service_name": "Calma AI Inference Service",
        "service_version": "1.0.0",
        "model_status": "loaded",
        "uptime_seconds": 3600.5
    }

    MODEL_INFO_EXAMPLE = {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "model_path": "models/calma-final",
        "model_version": "calma-v1",
        "device": "cuda",
        "load_time_seconds": 45.2,
        "memory_usage_mb": 6144,
        "quantization": True,
        "lora_enabled": True,
        "cultural_guidelines": {
            "cultural_focus": "Zimbabwean communities",
            "languages_considered": ["English", "Shona", "Ndebele"]
        },
        "supported_features": [
            "mood_detection",
            "cultural_awareness",
            "resource_recommendations"
        ]
    }