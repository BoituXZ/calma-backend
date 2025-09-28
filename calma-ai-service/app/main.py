"""Main FastAPI application for Calma AI inference service."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .models.schemas import (
    InferenceRequest,
    InferenceResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    CulturalGuidelinesResponse,
    ErrorResponse,
    ResponseMetadata,
    ResponseQualityMetrics,
    SchemaExamples,
)
from .services.model_service import model_service
from .services.inference_service import inference_service
from .services.analysis_service import analysis_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Track service startup time
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    global startup_time
    startup_time = time.time()

    logger.info("Starting Calma AI Inference Service...")

    # Load model on startup
    logger.info("Loading AI model...")
    model_loaded = await model_service.load_model()

    if model_loaded:
        logger.info("Model loaded successfully")
        logger.info(f"Service ready on {settings.host}:{settings.port}")
    else:
        logger.error("Failed to load model - service will be unhealthy")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Calma AI Inference Service...")
    await model_service.cleanup()


# Create FastAPI app with lifespan management
app = FastAPI(
    title=settings.service_name,
    description="Cultural-aware AI inference service for Zimbabwean mental health support",
    version=settings.service_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail,
            request_id=getattr(request.state, "request_id", None),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured error response."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            request_id=getattr(request.state, "request_id", None),
        ).dict(),
    )


# Middleware to add request ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check service and model health status",
)
async def health_check():
    """Check service and model health status."""
    global startup_time

    # Determine service status
    model_ready = model_service.is_ready()
    service_status = "healthy" if model_ready else "unhealthy"

    # Calculate uptime
    uptime = time.time() - startup_time if startup_time else None

    response_data = {
        "status": service_status,
        "timestamp": datetime.now(),
        "service_name": settings.service_name,
        "service_version": settings.service_version,
        "model_status": "loaded" if model_ready else "not_loaded",
        "uptime_seconds": round(uptime, 2) if uptime else None,
    }

    # Add model info if available
    if model_ready:
        response_data["model_info"] = model_service.get_model_info()

    return HealthCheckResponse(**response_data)


# Model information endpoint
@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description="Get detailed information about the loaded model",
)
async def get_model_info():
    """Get detailed information about the loaded model."""
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    model_info = model_service.get_model_info()
    cultural_guidelines = inference_service.get_cultural_guidelines()

    return ModelInfoResponse(
        **model_info,
        cultural_guidelines=cultural_guidelines,
    )


# Cultural guidelines endpoint
@app.get(
    "/cultural-guidelines",
    response_model=CulturalGuidelinesResponse,
    summary="Cultural guidelines",
    description="Get information about cultural awareness and guidelines",
)
async def get_cultural_guidelines():
    """Get information about cultural awareness and guidelines."""
    guidelines = inference_service.get_cultural_guidelines()

    return CulturalGuidelinesResponse(
        **guidelines,
        cultural_patterns={
            key: [pattern.pattern for pattern in patterns]
            for key, patterns in analysis_service.cultural_patterns.items()
        }
    )


# Main inference endpoint
@app.post(
    "/infer",
    response_model=InferenceResponse,
    summary="AI inference",
    description="Generate AI response with cultural awareness and analysis",
)
async def inference_endpoint(request_data: InferenceRequest):
    """Generate AI response with comprehensive analysis."""
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded and ready for inference"
        )

    try:
        # Generate AI response
        inference_result = await inference_service.generate_response(
            message=request_data.message,
            context=request_data.context,
            temperature=request_data.parameters.temperature,
            max_tokens=request_data.parameters.max_tokens,
        )

        # Analyze the message and response
        analysis_result = analysis_service.analyze_message(
            message=request_data.message,
            ai_response=inference_result["response"]
        )

        # Build response metadata
        metadata = ResponseMetadata(
            mood_detected=analysis_result["mood_detected"],
            confidence=analysis_result["confidence"],
            emotional_intensity=analysis_result["emotional_intensity"],
            response_time_ms=inference_result["inference_time_ms"],
            model_version=inference_result["model_version"],
            emotional_indicators=analysis_result["emotional_indicators"],
        )

        # Build quality metrics
        quality_metrics = ResponseQualityMetrics(
            **analysis_result["response_metrics"]
        )

        # Build complete response
        response = InferenceResponse(
            response=inference_result["response"],
            metadata=metadata,
            suggested_resources=analysis_result["suggested_resources"],
            cultural_elements_detected=analysis_result["cultural_elements_detected"],
            quality_metrics=quality_metrics,
            parameters_used=request_data.parameters,
        )

        return response

    except RuntimeError as e:
        if "timeout" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed due to an unexpected error"
        )


# Root endpoint
@app.get(
    "/",
    summary="Service information",
    description="Get basic service information",
)
async def root():
    """Get basic service information."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
    }


# Add example responses to OpenAPI schema
def add_examples_to_schema():
    """Add example responses to OpenAPI documentation."""
    if hasattr(app, "openapi_schema") and app.openapi_schema:
        # Add examples to inference endpoint
        inference_path = app.openapi_schema["paths"]["/infer"]["post"]

        inference_path["requestBody"]["content"]["application/json"]["example"] = (
            SchemaExamples.INFERENCE_REQUEST_EXAMPLE
        )

        inference_path["responses"]["200"]["content"]["application/json"]["example"] = (
            SchemaExamples.INFERENCE_RESPONSE_EXAMPLE
        )

        # Add examples to health endpoint
        health_path = app.openapi_schema["paths"]["/health"]["get"]
        health_path["responses"]["200"]["content"]["application/json"]["example"] = (
            SchemaExamples.HEALTH_CHECK_EXAMPLE
        )

        # Add examples to model-info endpoint
        model_info_path = app.openapi_schema["paths"]["/model-info"]["get"]
        model_info_path["responses"]["200"]["content"]["application/json"]["example"] = (
            SchemaExamples.MODEL_INFO_EXAMPLE
        )


# Hook to add examples after OpenAPI schema generation
@app.middleware("http")
async def add_examples_middleware(request: Request, call_next):
    """Middleware to ensure examples are added to OpenAPI schema."""
    response = await call_next(request)

    # Add examples on first request to /docs or /openapi.json
    if request.url.path in ["/docs", "/openapi.json"]:
        add_examples_to_schema()

    return response


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )