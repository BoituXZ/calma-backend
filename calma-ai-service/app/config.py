"""Configuration settings for Calma AI inference service."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = {
        "protected_namespaces": ("settings_",),
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

    # Model settings
    model_path: str = "/home/boitu/Desktop/Coding/Calma/calma-backend/calma-ai/models/calma-final"
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Inference settings
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    do_sample: bool = True

    # Performance settings
    max_memory_mb: int = 5800  # RTX 4050 memory
    device_map: str = "cuda:0"  # Force GPU usage
    torch_dtype: str = "float16"
    use_4bit: bool = True  # Enable 4-bit quantization for GPU memory efficiency

    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # CORS settings
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://localhost:5173"
    ]

    # Service settings
    service_name: str = "Calma AI Inference Service"
    service_version: str = "1.0.0"
    model_version: str = "calma-v1"

    # Timeout settings
    inference_timeout: int = 180  # seconds (increased for CPU inference)
    model_load_timeout: int = 120  # seconds


# Global settings instance
settings = Settings()