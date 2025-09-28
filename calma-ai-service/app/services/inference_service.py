"""Inference service with cultural awareness for Calma AI."""

import logging
import time
import asyncio
from typing import Optional, Dict, Any, List

from .model_service import model_service
from ..config import settings

logger = logging.getLogger(__name__)


class InferenceService:
    """Handles AI inference with cultural context for Zimbabwean mental health support."""

    def __init__(self):
        self.cultural_system_prompt = self._build_cultural_system_prompt()

    def _build_cultural_system_prompt(self) -> str:
        """Build the cultural system prompt for Zimbabwean mental health context."""
        return """You are Calma, a culturally-aware mental health support assistant designed specifically for Zimbabwean communities. Your role is to provide empathetic, culturally sensitive psychological support.

CULTURAL GUIDELINES:
- Respect traditional Zimbabwean values including Ubuntu (humanness/interconnectedness)
- Understand the importance of family, elders, and community in Zimbabwean culture
- Be aware of Shona and Ndebele cultural contexts and customs
- Acknowledge the role of traditional healers alongside modern mental health approaches
- Consider economic challenges and resource limitations common in Zimbabwe
- Use appropriate language that shows respect for age and social hierarchy when indicated
- Understand rural vs urban cultural differences in Zimbabwe

COMMUNICATION STYLE:
- Be warm, empathetic, and respectful
- Use simple, clear language accessible to different education levels
- Incorporate culturally relevant metaphors and examples when appropriate
- Show understanding of family dynamics and community expectations
- Be sensitive to stigma around mental health in traditional communities

MENTAL HEALTH APPROACH:
- Provide practical, actionable advice suitable for the Zimbabwean context
- Suggest community-based support and family involvement when appropriate
- Offer coping strategies that work within cultural and economic constraints
- Be aware of limited access to formal mental health services
- Encourage seeking professional help when needed while acknowledging barriers

Remember: You are a supportive companion, not a replacement for professional medical or psychological treatment. Always encourage seeking professional help for serious concerns."""

    async def generate_response(
        self,
        message: str,
        context: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate AI response with cultural awareness."""

        if not model_service.is_ready():
            raise RuntimeError("Model is not loaded and ready for inference")

        try:
            start_time = time.time()

            # Use provided parameters or defaults
            temp = temperature if temperature is not None else settings.temperature
            max_toks = max_tokens if max_tokens is not None else settings.max_tokens

            # Build the full prompt with cultural context
            full_prompt = self._build_prompt(message, context)

            # Generate response using the pipeline
            response = await self._generate_with_timeout(
                full_prompt,
                temperature=temp,
                max_new_tokens=max_toks,
                do_sample=settings.do_sample,
                top_p=settings.top_p,
                pad_token_id=model_service.tokenizer.eos_token_id,
                eos_token_id=model_service.tokenizer.eos_token_id,
            )

            # Extract generated text
            generated_text = response[0]['generated_text']

            # Clean up the response to extract only the assistant's reply
            ai_response = self._extract_response(generated_text, full_prompt)

            inference_time = time.time() - start_time

            return {
                "response": ai_response,
                "inference_time_ms": round(inference_time * 1000, 2),
                "model_version": settings.model_version,
                "parameters": {
                    "temperature": temp,
                    "max_tokens": max_toks,
                    "top_p": settings.top_p,
                }
            }

        except asyncio.TimeoutError:
            logger.error(f"Inference timeout after {settings.inference_timeout} seconds")
            raise RuntimeError(f"Inference timeout after {settings.inference_timeout} seconds")
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")

    def _build_prompt(self, message: str, context: Optional[str] = None) -> str:
        """Build the complete prompt with system instructions and context."""

        # Start with the cultural system prompt
        prompt_parts = [
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
            self.cultural_system_prompt,
            "<|eot_id|>"
        ]

        # Add conversation context if provided
        if context:
            prompt_parts.extend([
                "<|start_header_id|>user<|end_header_id|>",
                f"Previous conversation context:\n{context}",
                "<|eot_id|>"
            ])

        # Add the current user message
        prompt_parts.extend([
            "<|start_header_id|>user<|end_header_id|>",
            message,
            "<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>"
        ])

        return "\n".join(prompt_parts)

    def _extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract the AI response from the generated text."""

        # Remove the original prompt from the generated text
        if generated_text.startswith(original_prompt):
            response = generated_text[len(original_prompt):].strip()
        else:
            response = generated_text.strip()

        # Clean up any residual formatting tokens
        response = response.replace("<|eot_id|>", "").strip()
        response = response.replace("<|end_of_text|>", "").strip()

        # Remove any system or formatting artifacts
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip().startswith('<|') and not line.strip().endswith('|>'):
                cleaned_lines.append(line)

        response = '\n'.join(cleaned_lines).strip()

        # Ensure we have a meaningful response
        if not response or len(response.strip()) < 5:
            return "I understand you're reaching out. Could you tell me more about what's on your mind? I'm here to listen and support you."

        return response

    async def _generate_with_timeout(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Generate response with timeout handling."""

        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None,
                lambda: model_service.pipeline(
                    prompt,
                    return_full_text=True,
                    **kwargs
                )
            )

            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=settings.inference_timeout)
            return result

        except asyncio.TimeoutError:
            logger.error("Inference timeout occurred")
            raise

    def get_cultural_guidelines(self) -> Dict[str, Any]:
        """Get information about cultural guidelines and context."""
        return {
            "cultural_focus": "Zimbabwean communities",
            "languages_considered": ["English", "Shona", "Ndebele"],
            "key_values": ["Ubuntu", "Family unity", "Respect for elders", "Community support"],
            "contexts": ["Urban", "Rural", "Peri-urban"],
            "approach": "Culturally-sensitive mental health support",
            "system_prompt_length": len(self.cultural_system_prompt),
        }


# Global inference service instance
inference_service = InferenceService()