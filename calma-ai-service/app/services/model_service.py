"""Model loading and caching service for Calma AI inference."""

import gc
import logging
import time
import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
import psutil

from ..config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """Handles dual model loading: casual (base) and therapeutic (fine-tuned)."""

    def __init__(self):
        # Casual model (base Llama for natural conversation)
        self.casual_model = None
        self.casual_pipeline = None

        # Therapeutic model (fine-tuned with LoRA for mental health support)
        self.therapeutic_model = None
        self.therapeutic_pipeline = None

        # Shared tokenizer (same for both models)
        self.tokenizer = None

        self.device = None
        self.models_loaded = False
        self.load_time = None
        self.model_info = {}

    async def load_model(self) -> bool:
        """Load both casual (base) and therapeutic (fine-tuned) models."""
        try:
            logger.info("Starting dual model loading process...")
            start_time = time.time()

            # Clear any existing models from memory
            self._clear_memory()

            # Determine device
            self.device = self._get_optimal_device()
            logger.info(f"Using device: {self.device}")

            # Configure quantization for GPU if available
            quantization_config = None
            if self.device != "cpu":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            # Load tokenizer (shared between both models)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.base_model_name,
                trust_remote_code=True
            )

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": settings.device_map if self.device != "cpu" else None,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            # Load CASUAL model (base Llama without LoRA)
            logger.info("Loading casual conversation model (base Llama)...")
            self.casual_model = AutoModelForCausalLM.from_pretrained(
                settings.base_model_name,
                **model_kwargs
            )

            if self.device == "cpu":
                self.casual_model = self.casual_model.to(self.device)

            # Create casual pipeline
            self.casual_pipeline = pipeline(
                "text-generation",
                model=self.casual_model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=settings.device_map if self.device != "cpu" else None,
            )
            logger.info("✓ Casual model loaded")

            # Load THERAPEUTIC model (base + LoRA adapters)
            if self._has_lora_adapters():
                logger.info("Loading therapeutic support model (fine-tuned with LoRA)...")

                # Load base model again for therapeutic
                therapeutic_base = AutoModelForCausalLM.from_pretrained(
                    settings.base_model_name,
                    **model_kwargs
                )

                # Apply LoRA adapters
                self.therapeutic_model = PeftModel.from_pretrained(
                    therapeutic_base,
                    settings.model_path,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
                )

                if self.device == "cpu":
                    self.therapeutic_model = self.therapeutic_model.to(self.device)

                # Create therapeutic pipeline
                self.therapeutic_pipeline = pipeline(
                    "text-generation",
                    model=self.therapeutic_model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map=settings.device_map if self.device != "cpu" else None,
                )
                logger.info("✓ Therapeutic model loaded")
            else:
                logger.warning("No LoRA adapters found - using base model for both modes")
                self.therapeutic_model = self.casual_model
                self.therapeutic_pipeline = self.casual_pipeline

            self.load_time = time.time() - start_time
            self.models_loaded = True

            # Store model info
            self.model_info = {
                "model_name": settings.base_model_name,
                "model_path": settings.model_path,
                "device": str(self.device),
                "load_time_seconds": round(self.load_time, 2),
                "memory_usage_mb": self._get_memory_usage(),
                "quantization": quantization_config is not None,
                "lora_enabled": self._has_lora_adapters(),
                "dual_model_system": True,
                "model_version": settings.model_version
            }

            logger.info(f"Both models loaded successfully in {self.load_time:.2f} seconds")
            logger.info(f"Memory usage: {self.model_info['memory_usage_mb']} MB")

            return True

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            self.models_loaded = False
            self._clear_memory()
            return False

    def _get_optimal_device(self) -> str:
        """Determine the best device for inference."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            logger.info(f"GPU available with {gpu_memory_gb:.1f} GB memory")

            if gpu_memory_gb >= 4:  # Minimum for 3B model with 4-bit quantization
                return "cuda"
            else:
                logger.warning("GPU memory insufficient, falling back to CPU")
                return "cpu"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"

    def _has_lora_adapters(self) -> bool:
        """Check if LoRA adapters exist in the model path."""
        import os
        adapter_config_path = os.path.join(settings.model_path, "adapter_config.json")
        return os.path.exists(adapter_config_path)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return int(process.memory_info().rss / 1024 / 1024)

    def get_pipeline(self, mode: str = "casual") -> Optional[Any]:
        """Get the appropriate pipeline based on conversation mode."""
        if mode == "casual":
            return self.casual_pipeline
        elif mode == "therapeutic":
            return self.therapeutic_pipeline
        elif mode == "crisis":
            # Use base model for crisis - the fine-tuned model deflects from serious issues
            logger.info("Using base model for crisis response (bypassing fine-tuned deflection)")
            return self.casual_pipeline
        else:
            logger.warning(f"Unknown mode '{mode}', defaulting to casual")
            return self.casual_pipeline

    def _clear_memory(self):
        """Clear GPU and system memory."""
        if self.casual_model is not None:
            del self.casual_model
        if self.therapeutic_model is not None:
            del self.therapeutic_model
        if self.casual_pipeline is not None:
            del self.casual_pipeline
        if self.therapeutic_pipeline is not None:
            del self.therapeutic_pipeline
        if self.tokenizer is not None:
            del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            **self.model_info,
            "status": "loaded" if self.models_loaded else "not_loaded",
            "current_memory_mb": self._get_memory_usage(),
            "device_available": torch.cuda.is_available(),
        }

    def is_ready(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return (self.models_loaded and
                self.casual_pipeline is not None and
                self.therapeutic_pipeline is not None)

    async def cleanup(self):
        """Cleanup resources when shutting down."""
        logger.info("Cleaning up model resources...")
        self._clear_memory()
        self.models_loaded = False


# Global model service instance
model_service = ModelService()