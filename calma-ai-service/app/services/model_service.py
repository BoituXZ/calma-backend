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
    """Handles loading the fine-tuned Calma model (single model system)."""

    def __init__(self):
        # Single fine-tuned model for all interactions
        self.model = None
        self.pipeline = None
        self.tokenizer = None

        self.device = None
        self.models_loaded = False
        self.load_time = None
        self.model_info = {}

    async def load_model(self) -> bool:
        """Load the fine-tuned Calma model (single model system)."""
        try:
            logger.info("Loading fine-tuned Calma model (single model system)...")
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

            # Load tokenizer
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

            # Check if we have LoRA adapters
            if self._has_lora_adapters():
                logger.info(f"Loading base model with LoRA adapters from {settings.model_path}...")

                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    settings.base_model_name,
                    **model_kwargs
                )

                # Apply LoRA adapters
                self.model = PeftModel.from_pretrained(
                    base_model,
                    settings.model_path,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
                )

                if self.device == "cpu":
                    self.model = self.model.to(self.device)

                logger.info("✓ Fine-tuned model with LoRA loaded")
            else:
                # Fallback to base model if no adapters found
                logger.warning("No LoRA adapters found - loading base model only")
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.base_model_name,
                    **model_kwargs
                )

                if self.device == "cpu":
                    self.model = self.model.to(self.device)

                logger.info("✓ Base model loaded (no LoRA adapters)")

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=settings.device_map if self.device != "cpu" else None,
            )

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
                "single_model_system": True,
                "model_version": settings.model_version
            }

            logger.info(f"Model loaded successfully in {self.load_time:.2f} seconds")
            logger.info(f"Memory usage: {self.model_info['memory_usage_mb']} MB")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
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
        """Get the pipeline (single model system - mode parameter kept for compatibility)."""
        # All modes use the same fine-tuned model now
        return self.pipeline

    def _clear_memory(self):
        """Clear GPU and system memory."""
        if self.model is not None:
            del self.model
        if self.pipeline is not None:
            del self.pipeline
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
        """Check if model is loaded and ready for inference."""
        return self.models_loaded and self.pipeline is not None

    async def cleanup(self):
        """Cleanup resources when shutting down."""
        logger.info("Cleaning up model resources...")
        self._clear_memory()
        self.models_loaded = False


# Global model service instance
model_service = ModelService()