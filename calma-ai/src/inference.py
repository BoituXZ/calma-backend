from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except ImportError:
    from peft.peft_model import PeftModel
import uvicorn
from datetime import datetime
from typing import Optional, List, Dict

app = FastAPI(title="Calma AI Service", description="Psychological Health Chatbot API")

class CulturalProfile(BaseModel):
    ageGroup: str = "ADULT"
    location: str = "URBAN"
    educationLevel: str = "SECONDARY"
    ethnicBackground: Optional[str] = "Shona"
    religiousBackground: Optional[str] = None
    familyStructure: str = "NUCLEAR"
    respectLevel: str = "MODERATE"
    economicStatus: str = "MIDDLE"

class ConversationMemory(BaseModel):
    previousTopics: List[str] = []
    relationshipStrength: float = 0.5
    trustLevel: float = 0.5
    conversationHistory: List[Dict] = []

class ChatRequest(BaseModel):
    message: str
    userId: str
    sessionId: Optional[str] = None
    culturalProfile: Optional[CulturalProfile] = None
    conversationMemory: Optional[ConversationMemory] = None
    max_length: int = 512

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    confidence: float
    emotionalTone: Optional[str] = None
    detectedTopics: List[str] = []
    culturalContext: Optional[Dict] = None
    memoryReferences: List[str] = []
    followUpNeeded: bool = False

class CalmaInferenceService:
    """
    Enhanced FastAPI service for Calma chatbot inference with cultural awareness and memory.
    Integrates with the Prisma database schema.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model()
        
        # Topic classification patterns
        self.topic_patterns = {
            "family_conflict": ["family", "parent", "mother", "father", "sibling", "relative"],
            "work_stress": ["work", "job", "boss", "colleague", "office", "employment"],
            "academic_pressure": ["school", "exam", "study", "university", "grade", "student"],
            "relationship_issues": ["boyfriend", "girlfriend", "partner", "relationship", "dating"],
            "financial_worry": ["money", "financial", "afford", "expensive", "budget", "income"],
            "health_anxiety": ["health", "sick", "illness", "medical", "doctor", "hospital"],
            "career_uncertainty": ["career", "future", "profession", "calling", "ambition"],
            "social_isolation": ["lonely", "alone", "friends", "social", "isolated"],
            "cultural_identity": ["culture", "tradition", "identity", "belong", "heritage"]
        }
        
        # Emotional tone patterns
        self.emotion_patterns = {
            "anxious": ["worried", "nervous", "scared", "panic", "stress"],
            "depressed": ["sad", "hopeless", "empty", "worthless", "down"],
            "frustrated": ["angry", "annoyed", "fed up", "irritated"],
            "hopeful": ["optimistic", "positive", "confident", "excited"],
            "confused": ["lost", "unsure", "don't know", "confused", "unclear"]
        }
    
    def load_model(self):
        """Load the fine-tuned model for inference."""
        print("Loading model for inference...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load fine-tuned weights
        try:
            peft_model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = peft_model.merge_and_unload()
            self.model.eval()
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            print("Using base model instead...")
            self.model = base_model
            self.model.eval()
        
        print("Model loaded successfully!")
    
    def analyze_emotional_tone(self, message: str) -> str:
        """Analyze emotional tone of user message."""
        message_lower = message.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_patterns.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        return "neutral"
    
    def detect_topics(self, message: str) -> List[str]:
        """Detect topics discussed in the message."""
        message_lower = message.lower()
        detected_topics = []
        
        for topic, keywords in self.topic_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics or ["general_wellbeing"]
    
    def build_cultural_context(self, profile: Optional[CulturalProfile]) -> Dict:
        """Build cultural context for response generation."""
        if not profile:
            return {
                "ageGroup": "ADULT",
                "location": "URBAN",
                "culturalAdaptations": ["ubuntu_philosophy", "community_focus"]
            }
        
        cultural_adaptations = []
        
        # Age-based adaptations
        if profile.ageGroup == "YOUTH":
            cultural_adaptations.extend(["peer_pressure_awareness", "educational_focus"])
        elif profile.ageGroup == "ELDER":
            cultural_adaptations.extend(["wisdom_seeking", "traditional_values"])
        
        # Location-based adaptations
        if profile.location == "RURAL":
            cultural_adaptations.extend(["traditional_healing", "community_elders"])
        elif profile.location == "URBAN":
            cultural_adaptations.extend(["modern_challenges", "work_life_balance"])
        
        # Respect level adaptations
        if profile.respectLevel == "HIGH":
            cultural_adaptations.append("formal_communication")
        elif profile.respectLevel == "RELAXED":
            cultural_adaptations.append("casual_approach")
        
        return {
            "ageGroup": profile.ageGroup,
            "location": profile.location,
            "familyStructure": profile.familyStructure,
            "culturalAdaptations": cultural_adaptations
        }
    
    def build_memory_context(self, memory: Optional[ConversationMemory]) -> str:
        """Build conversation memory context."""
        if not memory or not memory.conversationHistory:
            return ""
        
        memory_context = ""
        
        # Add previous topics
        if memory.previousTopics:
            memory_context += f"Previous topics discussed: {', '.join(memory.previousTopics)}. "
        
        # Add relationship context
        if memory.relationshipStrength > 0.7:
            memory_context += "You have built a strong rapport with this user. "
        elif memory.trustLevel > 0.7:
            memory_context += "The user trusts you and values your advice. "
        
        # Add recent conversation context
        if memory.conversationHistory:
            recent_context = memory.conversationHistory[-2:]  # Last 2 exchanges
            for exchange in recent_context:
                if exchange.get('type') == 'user':
                    memory_context += f"User previously mentioned: {exchange.get('message', '')[:100]}... "
        
        return memory_context
    
    def generate_response(self, request: ChatRequest) -> ChatResponse:
        """Generate culturally-aware response with memory integration."""
        
        # Analyze the message
        emotional_tone = self.analyze_emotional_tone(request.message)
        detected_topics = self.detect_topics(request.message)
        cultural_context = self.build_cultural_context(request.culturalProfile)
        memory_context = self.build_memory_context(request.conversationMemory)
        
        # Build enhanced system prompt
        system_prompt = self._build_system_prompt(cultural_context, memory_context)
        
        # Build user message with context
        user_message = self._build_user_message(request, cultural_context, memory_context)
        
        # Create chat format
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Generate response
        response_text = self._generate_text(chat, request.max_length)
        
        # Analyze response for follow-up needs
        follow_up_needed = self._needs_followup(request.message, response_text, emotional_tone)
        
        # Build memory references
        memory_references = self._build_memory_references(detected_topics, request.conversationMemory)
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat(),
            confidence=self._calculate_confidence(response_text, detected_topics),
            emotionalTone=emotional_tone,
            detectedTopics=detected_topics,
            culturalContext=cultural_context,
            memoryReferences=memory_references,
            followUpNeeded=follow_up_needed
        )
    
    def _build_system_prompt(self, cultural_context: Dict, memory_context: str) -> str:
        """Build enhanced system prompt with cultural and memory context."""
        base_prompt = """You are Calma, a culturally-aware psychological health assistant designed for African users, specifically Zimbabwean contexts. You understand and respect:

- Ubuntu philosophy: "I am because we are" - emphasizing interconnectedness
- Traditional African family structures and deep respect for elders
- Zimbabwean cultural values, including Shona and Ndebele traditions
- The importance of family, community, and spiritual well-being
- Economic realities and challenges faced by Zimbabweans
- The role of traditional healing alongside modern approaches"""
        
        # Add cultural adaptations
        cultural_adaptations = cultural_context.get("culturalAdaptations", [])
        if "formal_communication" in cultural_adaptations:
            base_prompt += "\n\nUse formal, respectful language that shows proper deference."
        if "community_elders" in cultural_adaptations:
            base_prompt += "\n\nIncorporate references to seeking wisdom from elders and community leaders."
        if "traditional_healing" in cultural_adaptations:
            base_prompt += "\n\nAcknowledge traditional healing practices alongside modern mental health approaches."
        
        # Add memory context
        if memory_context:
            base_prompt += f"\n\nConversation context: {memory_context}"
        
        base_prompt += "\n\nProvide empathetic, practical advice that incorporates these cultural values while being supportive and non-judgmental."
        
        return base_prompt
    
    def _build_user_message(self, request: ChatRequest, cultural_context: Dict, memory_context: str) -> str:
        """Build enhanced user message with full context."""
        profile = request.culturalProfile
        
        context_parts = []
        
        if profile:
            context_parts.append(f"User Profile: {profile.ageGroup} from {profile.location} background")
            context_parts.append(f"Family Structure: {profile.familyStructure}")
            if profile.ethnicBackground:
                context_parts.append(f"Cultural Background: {profile.ethnicBackground}")
        
        if memory_context:
            context_parts.append(f"Previous Context: {memory_context}")
        
        context_string = " | ".join(context_parts)
        
        return f"{context_string}\n\nCurrent Issue: {request.message}"
    
    def _generate_text(self, chat: List[Dict], max_length: int) -> str:
        """Generate text using the fine-tuned model."""
        # Apply chat template
        if self.tokenizer is not None:
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = str(chat)
        
        # Tokenize
        if self.tokenizer is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        else:
            raise ValueError("Tokenizer not initialized")
        
        # Generate
        with torch.no_grad():
            if self.model is not None and self.tokenizer is not None:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                raise ValueError("Model or tokenizer not initialized")
        
        # Decode response
        if self.tokenizer is not None:
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise ValueError("Tokenizer not initialized")
        
        # Extract only the assistant's response
        assistant_start = full_response.find("assistant\n")
        if assistant_start != -1:
            response = full_response[assistant_start + len("assistant\n"):].strip()
        else:
            # Fallback: extract everything after the prompt
            if self.tokenizer is not None:
                prompt_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
                response = full_response[prompt_length:].strip()
            else:
                response = full_response
        
        return response
    
    def _needs_followup(self, user_message: str, response: str, emotional_tone: str) -> bool:
        """Determine if conversation needs professional follow-up."""
        crisis_indicators = [
            "suicide", "kill myself", "end it all", "no point living",
            "hurt myself", "want to die", "can't go on", "worthless"
        ]
        
        high_risk_emotions = ["depressed", "anxious"]
        
        user_lower = user_message.lower()
        has_crisis_language = any(indicator in user_lower for indicator in crisis_indicators)
        has_persistent_negative_emotion = emotional_tone in high_risk_emotions
        
        return has_crisis_language or has_persistent_negative_emotion
    
    def _build_memory_references(self, topics: List[str], memory: Optional[ConversationMemory]) -> List[str]:
        """Build references to previous conversations."""
        references = []
        
        if memory and memory.previousTopics:
            common_topics = set(topics) & set(memory.previousTopics)
            for topic in common_topics:
                references.append(f"previous_discussion_{topic}")
        
        return references
    
    def _calculate_confidence(self, response: str, topics: List[str]) -> float:
        """Calculate confidence score for the response."""
        base_confidence = 0.7
        
        # Increase confidence if response is substantial
        if len(response) > 100:
            base_confidence += 0.1
        
        # Increase confidence if specific topics are addressed
        if len(topics) > 1:
            base_confidence += 0.1
        
        # Cap at 0.95
        return min(base_confidence, 0.95)

# Initialize the service
service = CalmaInferenceService("models/fine_tuned")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Enhanced chat endpoint with cultural awareness and memory."""
    try:
        response = service.generate_response(request)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-message")
async def analyze_message(request: ChatRequest):
    """Analyze message for emotional tone and topics without generating response."""
    try:
        emotional_tone = service.analyze_emotional_tone(request.message)
        detected_topics = service.detect_topics(request.message)
        
        return {
            "emotionalTone": emotional_tone,
            "detectedTopics": detected_topics,
            "followUpNeeded": service._needs_followup(request.message, "", emotional_tone)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "model_loaded": service.model is not None,
        "device": str(service.device)
    }

@app.get("/cultural-adaptations")
async def get_cultural_adaptations():
    """Get available cultural adaptation options."""
    return {
        "ageGroups": ["YOUTH", "ADULT", "ELDER"],
        "locations": ["URBAN", "RURAL", "PERI_URBAN"],
        "educationLevels": ["PRIMARY", "SECONDARY", "TERTIARY", "POSTGRADUATE"],
        "familyTypes": ["NUCLEAR", "EXTENDED", "SINGLE_PARENT", "GUARDIAN"],
        "respectLevels": ["HIGH", "MODERATE", "RELAXED"],
        "economicLevels": ["LOW", "MIDDLE", "HIGH"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)