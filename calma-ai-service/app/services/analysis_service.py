"""Analysis service for mood detection and resource recommendations."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MoodAnalysis:
    """Result of mood analysis."""
    mood_detected: str
    confidence: float
    intensity: int  # 1-10 scale
    emotional_indicators: List[str]


@dataclass
class ResourceRecommendation:
    """Resource recommendation based on analysis."""
    resource_type: str
    reason: str
    priority: int  # 1-5 scale


class AnalysisService:
    """Analyzes messages for mood, cultural elements, and resource recommendations."""

    def __init__(self):
        self.mood_patterns = self._initialize_mood_patterns()
        self.cultural_patterns = self._initialize_cultural_patterns()
        self.resource_mapping = self._initialize_resource_mapping()

    def _initialize_mood_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mood detection patterns with intensity levels."""
        return {
            "negative": {
                "patterns": [
                    # Anxiety and stress
                    r"\b(anxious|worried|stressed|overwhelmed|panic|nervous|tense)\b",
                    r"\b(can't cope|too much|breaking down|falling apart)\b",
                    r"\b(sleepless|insomnia|restless|tired|exhausted)\b",

                    # Depression indicators
                    r"\b(sad|depressed|hopeless|empty|worthless|useless)\b",
                    r"\b(lonely|isolated|alone|abandoned|rejected)\b",
                    r"\b(giving up|no point|why bother|what's the use)\b",

                    # Anger and frustration
                    r"\b(angry|frustrated|furious|irritated|annoyed)\b",
                    r"\b(hate|can't stand|fed up|sick of)\b",

                    # Fear and uncertainty
                    r"\b(scared|afraid|frightened|terrified|fearful)\b",
                    r"\b(uncertain|confused|lost|don't know what to do)\b",
                ],
                "intensity_multipliers": {
                    r"\b(very|extremely|really|so|too)\b": 1.5,
                    r"\b(completely|totally|absolutely)\b": 2.0,
                    r"\b(a bit|somewhat|slightly)\b": 0.7,
                }
            },
            "neutral": {
                "patterns": [
                    r"\b(okay|fine|alright|normal|average)\b",
                    r"\b(thinking about|considering|wondering)\b",
                    r"\b(question|ask|help|advice|guidance)\b",
                ],
                "intensity_multipliers": {}
            },
            "positive": {
                "patterns": [
                    r"\b(happy|joyful|excited|pleased|glad|grateful)\b",
                    r"\b(better|improving|getting well|recovering)\b",
                    r"\b(hopeful|optimistic|confident|strong|capable)\b",
                    r"\b(love|appreciate|thankful|blessed)\b",
                    r"\b(accomplished|proud|successful|achieved)\b",
                ],
                "intensity_multipliers": {
                    r"\b(very|extremely|really|so)\b": 1.3,
                    r"\b(quite|pretty|fairly)\b": 1.1,
                }
            }
        }

    def _initialize_cultural_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting cultural elements."""
        return {
            "family_support": [
                r"\b(family|relatives|parents|siblings|aunts?|uncles?)\b",
                r"\b(mother|father|mama|baba|gogo|sekuru)\b",
                r"\b(extended family|clan|lineage)\b",
            ],
            "community_help": [
                r"\b(community|neighbors?|village|neighbourhood)\b",
                r"\b(church|congregation|fellowship|prayer)\b",
                r"\b(traditional|custom|culture|heritage)\b",
            ],
            "traditional_healing": [
                r"\b(traditional healer|n'anga|sangoma)\b",
                r"\b(herbs|traditional medicine|ancestral)\b",
                r"\b(spirits|ancestors|guidance from elders)\b",
            ],
            "economic_stress": [
                r"\b(money|financial|economic|poverty|poor)\b",
                r"\b(job|work|employment|unemployment|income)\b",
                r"\b(school fees|university|education costs)\b",
                r"\b(food|hunger|basic needs|survival)\b",
            ],
            "ubuntu_values": [
                r"\b(ubuntu|togetherness|interconnected|community spirit)\b",
                r"\b(helping others|supporting each other|collective)\b",
                r"\b(shared|common|united|solidarity)\b",
            ],
            "respect_hierarchy": [
                r"\b(elders|elderly|older people|seniors)\b",
                r"\b(respect|honor|reverence|dignity)\b",
                r"\b(authority|leadership|guidance)\b",
            ]
        }

    def _initialize_resource_mapping(self) -> Dict[str, List[str]]:
        """Initialize mapping of detected issues to resource recommendations."""
        return {
            "stress_management": [
                "anxiety", "stress", "overwhelmed", "panic", "nervous",
                "worried", "tense", "pressure"
            ],
            "depression_support": [
                "sad", "depressed", "hopeless", "empty", "worthless",
                "lonely", "isolated", "giving up"
            ],
            "family_counseling": [
                "family problems", "family conflict", "parents", "siblings",
                "family pressure", "family expectations"
            ],
            "financial_guidance": [
                "money", "financial", "economic", "poverty", "job",
                "unemployment", "school fees", "basic needs"
            ],
            "study_techniques": [
                "studies", "education", "school", "university", "exams",
                "academic", "learning", "concentration"
            ],
            "anger_management": [
                "angry", "frustrated", "furious", "irritated", "hate",
                "rage", "violent thoughts"
            ],
            "social_skills": [
                "social", "friends", "relationships", "communication",
                "shy", "introverted", "social anxiety"
            ],
            "spiritual_guidance": [
                "spiritual", "faith", "religion", "prayer", "church",
                "traditional", "ancestors", "meaning of life"
            ],
            "trauma_support": [
                "trauma", "abuse", "violence", "assault", "accident",
                "loss", "grief", "death", "flashbacks"
            ],
            "self_esteem": [
                "self-worth", "confidence", "self-esteem", "worthless",
                "useless", "not good enough", "failure"
            ]
        }

    def analyze_mood(self, message: str) -> MoodAnalysis:
        """Analyze the emotional tone and mood of a message."""
        message_lower = message.lower()
        mood_scores = {"negative": 0, "neutral": 0, "positive": 0}
        detected_indicators = []

        # Score each mood category
        for mood, patterns in self.mood_patterns.items():
            base_score = 0

            # Check main patterns
            for pattern in patterns["patterns"]:
                matches = re.findall(pattern, message_lower, re.IGNORECASE)
                if matches:
                    base_score += len(matches)
                    detected_indicators.extend(matches)

            # Apply intensity multipliers
            for multiplier_pattern, multiplier in patterns["intensity_multipliers"].items():
                if re.search(multiplier_pattern, message_lower, re.IGNORECASE):
                    base_score *= multiplier

            mood_scores[mood] = base_score

        # Determine dominant mood
        dominant_mood = max(mood_scores, key=mood_scores.get)
        total_score = sum(mood_scores.values())

        # Calculate confidence and intensity
        confidence = mood_scores[dominant_mood] / max(total_score, 1)
        confidence = min(confidence, 1.0)  # Cap at 1.0

        # If all scores are low, default to neutral
        if total_score < 1:
            dominant_mood = "neutral"
            confidence = 0.5

        # Calculate intensity (1-10 scale)
        intensity = min(int(mood_scores[dominant_mood] * 2) + 3, 10)
        if dominant_mood == "neutral":
            intensity = 5

        return MoodAnalysis(
            mood_detected=dominant_mood,
            confidence=round(confidence, 2),
            intensity=intensity,
            emotional_indicators=list(set(detected_indicators))
        )

    def detect_cultural_elements(self, message: str) -> List[str]:
        """Detect cultural elements and contexts in the message."""
        message_lower = message.lower()
        detected_elements = []

        for element, patterns in self.cultural_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    detected_elements.append(element)
                    break  # Avoid duplicate detection for same element

        return detected_elements

    def recommend_resources(self, message: str, mood_analysis: MoodAnalysis) -> List[str]:
        """Recommend appropriate resources based on message content and mood."""
        message_lower = message.lower()
        recommended_resources = set()

        # Mood-based recommendations
        if mood_analysis.mood_detected == "negative":
            if mood_analysis.intensity >= 8:
                recommended_resources.add("crisis_support")

            if any(indicator in ["sad", "depressed", "hopeless", "empty"]
                   for indicator in mood_analysis.emotional_indicators):
                recommended_resources.add("depression_support")

            if any(indicator in ["anxious", "worried", "stressed", "overwhelmed"]
                   for indicator in mood_analysis.emotional_indicators):
                recommended_resources.add("stress_management")

        # Content-based recommendations
        for resource_type, keywords in self.resource_mapping.items():
            for keyword in keywords:
                if keyword in message_lower:
                    recommended_resources.add(resource_type)

        # Cultural context recommendations
        cultural_elements = self.detect_cultural_elements(message)
        if "family_support" in cultural_elements:
            recommended_resources.add("family_counseling")
        if "economic_stress" in cultural_elements:
            recommended_resources.add("financial_guidance")
        if "traditional_healing" in cultural_elements:
            recommended_resources.add("spiritual_guidance")

        return list(recommended_resources)

    def analyze_message(self, message: str, ai_response: str) -> Dict[str, Any]:
        """Comprehensive analysis of user message and AI response."""

        # Analyze user mood
        mood_analysis = self.analyze_mood(message)

        # Detect cultural elements
        cultural_elements = self.detect_cultural_elements(message)

        # Recommend resources
        resource_recommendations = self.recommend_resources(message, mood_analysis)

        # Analyze response quality (basic metrics)
        response_metrics = self._analyze_response_quality(ai_response)

        return {
            "mood_detected": mood_analysis.mood_detected,
            "confidence": mood_analysis.confidence,
            "emotional_intensity": mood_analysis.intensity,
            "emotional_indicators": mood_analysis.emotional_indicators,
            "cultural_elements_detected": cultural_elements,
            "suggested_resources": resource_recommendations,
            "response_metrics": response_metrics
        }

    def _analyze_response_quality(self, response: str) -> Dict[str, Any]:
        """Analyze the quality and characteristics of the AI response."""

        word_count = len(response.split())
        sentence_count = len(re.findall(r'[.!?]+', response))

        # Check for cultural sensitivity indicators
        cultural_terms = [
            "community", "family", "ubuntu", "traditional", "elders",
            "respect", "culture", "together", "support"
        ]

        cultural_awareness_score = sum(
            1 for term in cultural_terms
            if term.lower() in response.lower()
        ) / len(cultural_terms)

        # Check for empathy indicators
        empathy_terms = [
            "understand", "feel", "sorry", "support", "here for you",
            "listening", "care", "important", "valid", "recognize"
        ]

        empathy_score = sum(
            1 for term in empathy_terms
            if term.lower() in response.lower()
        ) / len(empathy_terms)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "cultural_awareness_score": round(cultural_awareness_score, 2),
            "empathy_score": round(empathy_score, 2),
            "response_length_category": self._categorize_response_length(word_count)
        }

    def _categorize_response_length(self, word_count: int) -> str:
        """Categorize response length."""
        if word_count < 20:
            return "very_short"
        elif word_count < 50:
            return "short"
        elif word_count < 100:
            return "medium"
        elif word_count < 200:
            return "long"
        else:
            return "very_long"


# Global analysis service instance
analysis_service = AnalysisService()