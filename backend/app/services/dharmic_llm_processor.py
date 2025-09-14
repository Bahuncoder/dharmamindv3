"""
🕉️ Dharmic LLM Processor
========================

Processes LLM responses through a dharmic lens to ensure spiritual alignment.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

class DharmicProcessingMode(str, Enum):
    """Modes for dharmic processing"""
    STRICT = "strict"        # Strict dharmic alignment
    BALANCED = "balanced"    # Balance modern and traditional
    GENTLE = "gentle"        # Gentle spiritual guidance
    ACADEMIC = "academic"    # Academic/scholarly approach

class DharmicResponse:
    """Response after dharmic processing"""
    
    def __init__(
        self,
        original_response: str,
        processed_response: str,
        dharmic_alignment_score: float,
        spiritual_insights: List[str],
        sanskrit_references: Optional[List[str]] = None,
        rishi_guidance: Optional[str] = None,
        processing_mode: DharmicProcessingMode = DharmicProcessingMode.BALANCED
    ):
        self.original_response = original_response
        self.processed_response = processed_response
        self.dharmic_alignment_score = dharmic_alignment_score
        self.spiritual_insights = spiritual_insights
        self.sanskrit_references = sanskrit_references or []
        self.rishi_guidance = rishi_guidance
        self.processing_mode = processing_mode
        self.timestamp = datetime.now()

class DharmicLLMProcessor:
    """🕉️ Dharmic LLM Response Processor"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Dharmic principles and values
        self.dharmic_principles = [
            "ahimsa",           # Non-violence
            "satya",            # Truthfulness  
            "asteya",           # Non-stealing
            "brahmacharya",     # Spiritual discipline
            "aparigraha",       # Non-possessiveness
            "dharma",           # Righteous duty
            "karma",            # Action and consequence
            "moksha",           # Liberation
            "bhakti",           # Devotion
            "seva",             # Selfless service
        ]
        
        # Sanskrit wisdom phrases
        self.sanskrit_wisdom = {
            "truth": ["सत्यमेव जयते (Satyameva Jayate) - Truth alone triumphs"],
            "peace": ["शान्ति शान्ति शान्ति (Shanti Shanti Shanti) - Peace, peace, peace"],
            "wisdom": ["विद्या ददाति विनयं (Vidya Dadati Vinayam) - Knowledge gives humility"],
            "dharma": ["धर्मो रक्षति रक्षितः (Dharmo Rakshati Rakshitah) - Dharma protects those who protect it"],
            "love": ["वसुधैव कुटुम्बकम् (Vasudhaiva Kutumbakam) - The world is one family"],
        }
        
        self.logger.info("🕉️ Dharmic LLM Processor initialized")
    
    async def process_response(
        self,
        response: str,
        context: Dict[str, Any],
        mode: DharmicProcessingMode = DharmicProcessingMode.BALANCED
    ) -> DharmicResponse:
        """Process LLM response through dharmic lens"""
        
        try:
            # Analyze dharmic alignment
            alignment_score = self._calculate_dharmic_alignment(response)
            
            # Extract spiritual insights
            spiritual_insights = self._extract_spiritual_insights(response, context)
            
            # Add Sanskrit references if appropriate
            sanskrit_refs = self._add_sanskrit_references(response, context)
            
            # Get Rishi guidance if needed
            rishi_guidance = self._get_rishi_guidance(response, context, mode)
            
            # Process and enhance the response
            processed_response = self._enhance_response(
                response, spiritual_insights, sanskrit_refs, rishi_guidance, mode
            )
            
            return DharmicResponse(
                original_response=response,
                processed_response=processed_response,
                dharmic_alignment_score=alignment_score,
                spiritual_insights=spiritual_insights,
                sanskrit_references=sanskrit_refs,
                rishi_guidance=rishi_guidance,
                processing_mode=mode
            )
            
        except Exception as e:
            self.logger.error(f"Error processing dharmic response: {e}")
            # Return original response on error
            return DharmicResponse(
                original_response=response,
                processed_response=response,
                dharmic_alignment_score=0.5,
                spiritual_insights=["Error in processing - using original response"],
                processing_mode=mode
            )
    
    def _calculate_dharmic_alignment(self, response: str) -> float:
        """Calculate how well response aligns with dharmic principles"""
        try:
            response_lower = response.lower()
            
            # Check for dharmic principles
            principle_matches = 0
            total_principles = len(self.dharmic_principles)
            
            for principle in self.dharmic_principles:
                if principle in response_lower:
                    principle_matches += 1
            
            # Check for positive spiritual concepts
            positive_concepts = [
                "compassion", "wisdom", "peace", "love", "truth", "mindfulness",
                "meditation", "spiritual", "dharma", "karma", "enlightenment",
                "consciousness", "awareness", "harmony", "balance", "unity"
            ]
            
            concept_matches = 0
            for concept in positive_concepts:
                if concept in response_lower:
                    concept_matches += 1
            
            # Check for negative concepts that reduce alignment
            negative_concepts = [
                "violence", "hatred", "greed", "anger", "ignorance",
                "attachment", "ego", "materialism", "suffering"
            ]
            
            negative_matches = 0
            for concept in negative_concepts:
                if concept in response_lower:
                    negative_matches += 1
            
            # Calculate score (0.0 to 1.0)
            base_score = 0.5  # Neutral starting point
            principle_boost = (principle_matches / total_principles) * 0.3
            concept_boost = min(concept_matches * 0.05, 0.3)
            negative_penalty = min(negative_matches * 0.1, 0.2)
            
            final_score = base_score + principle_boost + concept_boost - negative_penalty
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating dharmic alignment: {e}")
            return 0.5
    
    def _extract_spiritual_insights(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Extract spiritual insights from the response"""
        insights = []
        
        try:
            response_lower = response.lower()
            
            # Map keywords to insights
            insight_map = {
                "meditation": "Consider deepening your meditation practice for greater awareness",
                "mindfulness": "Mindful awareness is the foundation of spiritual growth",
                "compassion": "Compassion toward all beings is the essence of dharma",
                "wisdom": "True wisdom comes from understanding the interconnectedness of all",
                "peace": "Inner peace is reflected in outer harmony",
                "dharma": "Living according to dharma brings lasting fulfillment",
                "karma": "Understanding karma helps us make conscious choices",
                "consciousness": "Expanding consciousness leads to spiritual evolution",
                "love": "Universal love transcends all boundaries and differences",
                "truth": "Seeking truth is the path to liberation"
            }
            
            for keyword, insight in insight_map.items():
                if keyword in response_lower and insight not in insights:
                    insights.append(insight)
            
            # Add general spiritual insight if none found
            if not insights:
                insights.append("Every moment offers an opportunity for spiritual growth")
            
            return insights[:3]  # Limit to 3 insights
            
        except Exception as e:
            self.logger.error(f"Error extracting spiritual insights: {e}")
            return ["Spiritual wisdom emerges through mindful reflection"]
    
    def _add_sanskrit_references(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Add relevant Sanskrit references"""
        try:
            response_lower = response.lower()
            references = []
            
            # Map concepts to Sanskrit
            concept_to_sanskrit = {
                "truth": "सत्यमेव जयते (Satyameva Jayate) - Truth alone triumphs",
                "peace": "शान्ति शान्ति शान्ति (Shanti Shanti Shanti) - Peace, peace, peace", 
                "wisdom": "विद्या ददाति विनयं (Vidya Dadati Vinayam) - Knowledge gives humility",
                "dharma": "धर्मो रक्षति रक्षितः (Dharmo Rakshati Rakshitah) - Dharma protects those who protect it",
                "love": "वसुधैव कुटुम्बकम् (Vasudhaiva Kutumbakam) - The world is one family",
                "consciousness": "अहं ब्रह्मास्मि (Aham Brahmasmi) - I am Brahman",
                "unity": "सर्वं खल्विदं ब्रह्म (Sarvam Khalvidam Brahma) - All this is indeed Brahman"
            }
            
            for concept, sanskrit in concept_to_sanskrit.items():
                if concept in response_lower and sanskrit not in references:
                    references.append(sanskrit)
            
            return references[:2]  # Limit to 2 references
            
        except Exception as e:
            self.logger.error(f"Error adding Sanskrit references: {e}")
            return []
    
    def _get_rishi_guidance(
        self, 
        response: str, 
        context: Dict[str, Any], 
        mode: DharmicProcessingMode
    ) -> Optional[str]:
        """Get appropriate Rishi guidance"""
        try:
            if mode == DharmicProcessingMode.ACADEMIC:
                return None  # No Rishi guidance in academic mode
            
            # Simple guidance based on context
            user_emotion = context.get("emotional_state", "neutral")
            
            guidance_map = {
                "sadness": "As Sage Vasishtha teaches, suffering is the gateway to compassion - honor your feelings while remembering your eternal nature.",
                "anger": "Channel this energy like Sage Jamadagni - with purpose and controlled power, transforming anger into righteous action.",
                "fear": "Trust in the cosmic order as Sage Atri reminds us - you are protected by the divine consciousness that flows through all.",
                "joy": "Celebrate this blessing while remaining detached, as Sage Bharadwaja teaches - joy shared is joy multiplied.",
                "confusion": "In uncertainty, return to your breath and inner knowing, as Sage Gautama guides - clarity comes through stillness.",
                "neutral": "Stay present and aware, as all the Rishis teach - each moment contains infinite wisdom."
            }
            
            return guidance_map.get(user_emotion, guidance_map["neutral"])
            
        except Exception as e:
            self.logger.error(f"Error getting Rishi guidance: {e}")
            return None
    
    def _enhance_response(
        self,
        response: str,
        insights: List[str],
        sanskrit_refs: List[str],
        rishi_guidance: Optional[str],
        mode: DharmicProcessingMode
    ) -> str:
        """Enhance response with dharmic elements"""
        
        try:
            enhanced = response
            
            # Add spiritual insights if mode allows
            if insights and mode != DharmicProcessingMode.ACADEMIC:
                enhanced += "\n\n🌟 Spiritual Insight: " + insights[0]
            
            # Add Sanskrit reference if available
            if sanskrit_refs and mode in [DharmicProcessingMode.BALANCED, DharmicProcessingMode.STRICT]:
                enhanced += "\n\n📿 " + sanskrit_refs[0]
            
            # Add Rishi guidance if available
            if rishi_guidance and mode in [DharmicProcessingMode.BALANCED, DharmicProcessingMode.STRICT]:
                enhanced += "\n\n🧘 " + rishi_guidance
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing response: {e}")
            return response

# Global processor instance
_dharmic_processor: Optional[DharmicLLMProcessor] = None

def get_dharmic_llm_processor() -> DharmicLLMProcessor:
    """Get global dharmic LLM processor instance"""
    global _dharmic_processor
    if _dharmic_processor is None:
        _dharmic_processor = DharmicLLMProcessor()
    return _dharmic_processor

def create_dharmic_llm_processor() -> DharmicLLMProcessor:
    """Create new dharmic LLM processor instance"""
    return DharmicLLMProcessor()

# Export commonly used classes and functions
__all__ = [
    'DharmicLLMProcessor',
    'DharmicResponse',
    'DharmicProcessingMode',
    'get_dharmic_llm_processor',
    'create_dharmic_llm_processor'
]
