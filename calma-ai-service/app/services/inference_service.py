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
        self.therapeutic_system_prompt = self._build_therapeutic_system_prompt()
        self.casual_system_prompt = self._build_casual_system_prompt()
        self.crisis_system_prompt = self._build_crisis_system_prompt()

    def _build_crisis_system_prompt(self) -> str:
        """Build crisis response prompt for severe distress situations - uses base model."""
        return """You are Calma, a mental health crisis support assistant for Zimbabwean communities. A person in severe distress is reaching out to you.

🚨 CRITICAL CRISIS RESPONSE PROTOCOL:

The user has expressed SEVERE DISTRESS (suicidal thoughts, self-harm, or crisis). You MUST respond appropriately.

IMMEDIATE ACTIONS REQUIRED:
1. ACKNOWLEDGE their pain seriously and directly: "I hear you, and I'm really concerned about what you're sharing. Thank you for trusting me with this."

2. ASSESS IMMEDIATE SAFETY: "Are you safe right now? Are you currently having thoughts of hurting yourself?"

3. PROVIDE CRISIS RESOURCES:
   - Zimbabwe National Crisis Line: 0800 123 456 (toll-free, 24/7)
   - Friendship Bench counselors at local clinics
   - Encourage contacting a trusted family member, elder, or community leader

4. VALIDATE and STAY ENGAGED: "What you're feeling is real and painful. You don't have to go through this alone. I'm here to listen. Can you tell me what's been happening?"

5. DO NOT:
   - Minimize their feelings
   - Change the subject
   - Suggest distractions or deflect
   - Offer tea, chips, or casual conversation
   - Dismiss their crisis

CULTURAL SENSITIVITY:
- Acknowledge Ubuntu (we are connected as community)
- Respect family and elder involvement
- Be aware of mental health stigma
- Offer community-based support options

Remember: Safety first. Stay focused on their crisis. Provide support and resources. Encourage professional help."""

    def _build_casual_system_prompt(self) -> str:
        """Build a casual, friendly system prompt for greetings and light conversation."""
        return """You are Calma, a warm and friendly companion for people in Zimbabwean communities.

RIGHT NOW, the user is having a casual conversation with you. Respond naturally and warmly, like a good friend would.

CRITICAL RULES FOR EARLY CONVERSATION (Messages 1-8):
- This is the RELATIONSHIP BUILDING phase
- Have normal, friendly conversations
- Talk about everyday things naturally (music, hobbies, work, interests)
- If someone says "I'm bored" → suggest fun activities, NOT therapy
- If someone mentions music → talk about music, NOT "calming down from stress"
- If someone says "yes please" → continue the current topic casually
- Build trust through genuine conversation FIRST

WHAT TO DO:
✓ Respond naturally to what they're actually saying
✓ Keep responses short and conversational (1-2 sentences)
✓ Ask about their interests, hobbies, day
✓ Be warm, authentic, and human
✓ Match their energy level

WHAT NOT TO DO:
✗ DON'T assume every statement is a cry for help
✗ DON'T interpret "I'm bored" as deep emotional distress
✗ DON'T turn every topic into mental health discussion
✗ DON'T use therapeutic language unless explicitly needed
✗ DON'T bring up past problems unprompted
✗ DON'T be overly formal or clinical

REMEMBER: Building rapport naturally is MORE therapeutic than forcing therapy mode. Let them lead.

CULTURAL AWARENESS:
- Respect Zimbabwean values of Ubuntu (community and connection)
- Use warm, accessible language
- Be respectful but friendly"""

    def _build_therapeutic_system_prompt(self) -> str:
        """Build the full therapeutic system prompt for deeper conversations."""
        return """You are Calma, a culturally-aware mental health support assistant designed specifically for Zimbabwean communities. Your role is to provide empathetic, culturally sensitive psychological support.

CRITICAL CRISIS RESPONSE PROTOCOLS:
⚠️ IMMEDIATE SAFETY FIRST - If user expresses:
- Suicidal thoughts ("want to die", "kill myself", "end my life", "suicidal")
- Self-harm intentions ("hurt myself", "harm myself")
- Severe crisis ("can't take it anymore", "unbearable")

YOU MUST:
1. ACKNOWLEDGE their pain directly and seriously: "I hear you, and I'm really concerned about what you're sharing."
2. VALIDATE their courage in reaching out: "It takes strength to share this."
3. ASK about immediate safety: "Are you safe right now? Are you having thoughts of hurting yourself?"
4. PROVIDE crisis resources:
   - Zimbabwe National Crisis Line: 0800 123 456 (toll-free)
   - Friendship Bench counselors available at local clinics
   - Encourage reaching out to trusted family member or elder
5. STAY engaged: "I'm here with you. Let's talk through this together."
6. DO NOT dismiss, minimize, or deflect from their distress
7. DO NOT change the subject or suggest distractions

CONVERSATION PRINCIPLES:
- Have natural, flowing conversations - don't rush to recall or reference past topics unless the user brings them up
- Respond authentically to what the user is saying RIGHT NOW
- Build rapport gradually through genuine interaction, not forced context recall
- Let the user lead the conversation - they'll mention what matters to them
- Save deeper discussions for when the user is ready, not when you think they should be

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
- Match the user's energy and pace - if they're casual, be conversational; if serious, be thoughtful

MENTAL HEALTH APPROACH:
- Provide practical, actionable advice suitable for the Zimbabwean context
- Suggest community-based support and family involvement when appropriate
- Offer coping strategies that work within cultural and economic constraints
- Be aware of limited access to formal mental health services
- Encourage seeking professional help when needed while acknowledging barriers

Remember: You are a supportive companion, not a replacement for professional medical or psychological treatment. Always encourage seeking professional help for serious concerns."""

    def _determine_conversation_mode(self, context: Optional[str]) -> str:
        """Determine which model to use based on context markers."""
        if not context:
            return "casual"

        # Check for explicit mode markers from NestJS
        is_early_stage = "[Conversation stage: early" in context
        is_forced_casual = "[INSTRUCTION: This is early conversation. Stay casual" in context
        is_distress = "[INSTRUCTION: User is expressing" in context and "distress" in context

        # CRITICAL: CRISIS MODE for severe distress - use base model to avoid fine-tuned deflection
        if is_distress and "severe" in context:
            return "crisis"
        # Moderate distress - use therapeutic (fine-tuned)
        elif is_distress and "moderate" in context:
            return "therapeutic"
        # Only use casual mode if explicitly forced AND no distress
        elif is_forced_casual and not is_distress:
            return "casual"
        # Early stage defaults to casual unless distress detected
        elif is_early_stage and not is_distress:
            return "casual"
        elif "Last interaction:" in context or "Recent exchange:" in context:
            return "casual"
        else:
            return "therapeutic"

    async def generate_response(
        self,
        message: str,
        context: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate AI response with cultural awareness using the fine-tuned model."""

        if not model_service.is_ready():
            raise RuntimeError("Model is not loaded and ready for inference")

        try:
            start_time = time.time()

            # Determine conversation mode for context (still used for prompt building)
            conversation_mode = self._determine_conversation_mode(context)
            logger.info(f"Using conversation mode: {conversation_mode}")

            # Get the pipeline (single model system)
            active_pipeline = model_service.get_pipeline(conversation_mode)

            # Use provided parameters or defaults
            temp = temperature if temperature is not None else settings.temperature
            max_toks = max_tokens if max_tokens is not None else settings.max_tokens

            # Build the full prompt with cultural context
            full_prompt = self._build_prompt(message, context)

            # Generate response using the selected pipeline
            response = await self._generate_with_timeout(
                full_prompt,
                pipeline=active_pipeline,
                temperature=temp,
                max_new_tokens=max_toks,
                do_sample=settings.do_sample,
                top_p=settings.top_p,
                repetition_penalty=1.3,  # Prevent regurgitating training data
                no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                pad_token_id=model_service.tokenizer.eos_token_id,
                eos_token_id=model_service.tokenizer.eos_token_id,
            )

            # Extract generated text
            generated_text = response[0]['generated_text']

            # Clean up the response to extract only the assistant's reply
            ai_response = self._extract_response(generated_text, full_prompt)

            # CRITICAL: Detect and override deflection in crisis situations
            if self._is_deflecting_from_crisis(ai_response, message):
                logger.warning(f"Deflection detected in crisis response. Using safety override.")
                ai_response = self._generate_crisis_override_response(message)
                conversation_mode = "crisis_override"

            inference_time = time.time() - start_time

            return {
                "response": ai_response,
                "inference_time_ms": round(inference_time * 1000, 2),
                "model_version": settings.model_version,
                "model_used": conversation_mode,
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

    def _get_few_shot_examples(self, conversation_mode: str) -> str:
        """Get few-shot examples based on conversation mode."""

        if conversation_mode == "casual":
            return """<|start_header_id|>user<|end_header_id|>
hi
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Hey! Good to see you. How's it going?
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
hello
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Hi there! What's on your mind today?
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
how are you
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I'm doing well, thanks for asking! How about you?
<|eot_id|>"""

        elif conversation_mode == "light":
            return """<|start_header_id|>user<|end_header_id|>
what's up
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Not much! Just here if you want to chat. What's going on with you?
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
just checking in
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I appreciate that! Glad you're here. Anything you want to talk about, or just saying hi?
<|eot_id|>"""

        elif conversation_mode == "crisis":
            # CRITICAL crisis response examples - direct, focused, no deflection
            return """<|start_header_id|>user<|end_header_id|>
I'm suicidal
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I hear you, and I'm really concerned about what you're sharing. Thank you for trusting me with this. Are you safe right now? Are you currently having thoughts of hurting yourself? Your safety is the most important thing. Please know that the Zimbabwe National Crisis Line is available 24/7 at 0800 123 456 - they can provide immediate support. I'm here to listen. What's been happening?
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
I want to die
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I'm so sorry you're in this much pain. Your life matters, and what you're feeling right now, even though it's overwhelming, can change with support. Are you in immediate danger? Please reach out to the Zimbabwe National Crisis Line at 0800 123 456 right now, or visit a Friendship Bench counselor at your local clinic. Is there a trusted family member or friend nearby who can be with you? You don't have to face this alone.
<|eot_id|>"""

        elif conversation_mode == "therapeutic":
            # Therapeutic response examples for moderate distress
            return """<|start_header_id|>user<|end_header_id|>
I'm struggling with my family issues
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I'm glad you feel comfortable sharing this with me. Family challenges can weigh heavily on us, especially when we value Ubuntu and our connections to others. Can you tell me more about what's been happening with your family? What's been the most difficult part for you?
<|eot_id|>"""

        return ""  # No examples for other modes

    def _build_prompt(self, message: str, context: Optional[str] = None) -> str:
        """Build the complete prompt with system instructions and context."""

        # Determine conversation mode based on context markers
        is_early_stage = "[Conversation stage: early" in (context or "")
        is_forced_casual = "[INSTRUCTION: This is early conversation. Stay casual" in (context or "")
        is_distress = "[INSTRUCTION: User is expressing" in (context or "") and "distress" in (context or "")
        is_severe_distress = is_distress and "severe" in (context or "")
        is_greeting = "Last interaction:" in (context or "")
        is_light_chat = "Recent exchange:" in (context or "")

        # Choose system prompt based on conversation mode
        # Priority: CRISIS (severe distress) > distress > forced casual > early stage > message type
        if is_severe_distress:
            system_prompt = self.crisis_system_prompt
            conversation_mode = "crisis"
        elif is_forced_casual or is_early_stage:
            system_prompt = self.casual_system_prompt
            conversation_mode = "casual"
        elif is_distress:
            system_prompt = self.therapeutic_system_prompt
            conversation_mode = "therapeutic"
        elif is_greeting:
            system_prompt = self.casual_system_prompt
            conversation_mode = "casual"
        elif is_light_chat:
            system_prompt = self.casual_system_prompt
            conversation_mode = "light"
        else:
            system_prompt = self.therapeutic_system_prompt
            conversation_mode = "therapeutic"

        # Start with the appropriate system prompt
        prompt_parts = [
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
            system_prompt,
            "<|eot_id|>"
        ]

        # Add few-shot examples for casual/light modes
        few_shot_examples = self._get_few_shot_examples(conversation_mode)
        if few_shot_examples:
            prompt_parts.append(few_shot_examples)

        # Add conversation context if provided
        if context and context.strip():
            if is_forced_casual or is_early_stage or is_greeting or is_light_chat:
                # For casual/early interactions, minimal context injection
                prompt_parts.extend([
                    "<|start_header_id|>user<|end_header_id|>",
                    f"[Context: {context}]",
                    "<|eot_id|>"
                ])
            else:
                # For substantive conversations, provide context normally
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

    def _is_deflecting_from_crisis(self, response: str, original_message: str) -> bool:
        """Detect if the response is deflecting from a crisis situation."""

        # Crisis indicators in user's message
        crisis_keywords = [
            'suicidal', 'suicide', 'kill myself', 'end my life', 'want to die',
            'harm myself', 'hurt myself', 'can\'t take it', 'cannot take it'
        ]

        is_crisis = any(keyword in original_message.lower() for keyword in crisis_keywords)

        if not is_crisis:
            return False

        # Deflection indicators in response
        deflection_patterns = [
            'tea', 'coffee', 'cup of', 'chips', 'snack', 'food',
            'movies', 'watch', 'tv', 'television',
            'birthday', 'weekend', 'holiday',
            'spar', 'shop', 'store',
            'calm down', 'relax', 'take a breath',
            'no worries', 'don\'t worry',
            'change the subject', 'talk about something else'
        ]

        response_lower = response.lower()
        has_deflection = any(pattern in response_lower for pattern in deflection_patterns)

        # Check if response lacks safety assessment
        safety_keywords = [
            'safe', 'danger', 'hurt', 'harm', 'crisis line', 'help',
            'counselor', 'therapist', 'professional', 'emergency'
        ]
        has_safety_check = any(keyword in response_lower for keyword in safety_keywords)

        # If crisis message gets deflection patterns and no safety check, it's deflecting
        return has_deflection or not has_safety_check

    def _generate_crisis_override_response(self, message: str) -> str:
        """Generate a safe, appropriate crisis response when deflection is detected."""

        # Analyze the specific crisis type
        message_lower = message.lower()

        if 'suicid' in message_lower or 'kill myself' in message_lower or 'want to die' in message_lower:
            return """I hear you, and I'm really concerned about what you're sharing. Thank you for trusting me with this.

Are you safe right now? Are you currently having thoughts of hurting yourself?

Please know that the Zimbabwe National Crisis Line is available 24/7 at 0800 123 456. They have trained counselors who can provide immediate support. You can also visit a Friendship Bench counselor at your local clinic.

What you're feeling is real and painful, but there is help available. You don't have to face this alone. Can you tell me what's been happening that's brought you to this point?"""

        elif 'harm myself' in message_lower or 'hurt myself' in message_lower:
            return """Thank you for telling me this. Your safety is the most important thing right now.

Are you in immediate danger? Do you have a plan to hurt yourself?

I strongly encourage you to reach out to the Zimbabwe National Crisis Line at 0800 123 456 right now, or visit a Friendship Bench counselor at your local clinic. Is there a trusted family member or friend nearby who can be with you?

What you're feeling is valid, and there are people who want to help. I'm here to listen. What's been going on?"""

        elif 'can\'t take' in message_lower or 'cannot take' in message_lower or 'unbearable' in message_lower:
            return """I can hear that you're in a lot of pain right now. I'm so sorry you're going through this.

Are you feeling safe? Is there someone nearby who can support you right now?

When things feel unbearable, it's important to reach out for help. The Zimbabwe National Crisis Line (0800 123 456) is available 24/7, and Friendship Bench counselors at local clinics can provide support.

You don't have to carry this alone. Can you share more about what's happening that's making things feel so overwhelming?"""

        else:
            # General crisis response
            return """I'm really concerned about what you're sharing, and I want to make sure you get the support you need.

Are you safe right now? How are you feeling at this moment?

If you're in crisis or thinking about hurting yourself, please reach out to the Zimbabwe National Crisis Line at 0800 123 456 (available 24/7), or visit a Friendship Bench counselor at your local clinic.

I'm here to listen and support you. Can you tell me more about what's going on?"""

    async def _generate_with_timeout(self, prompt: str, pipeline=None, **kwargs) -> List[Dict[str, Any]]:
        """Generate response with timeout handling using specified pipeline."""

        try:
            # Use provided pipeline or default to casual
            active_pipeline = pipeline if pipeline is not None else model_service.get_pipeline("casual")

            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None,
                lambda: active_pipeline(
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