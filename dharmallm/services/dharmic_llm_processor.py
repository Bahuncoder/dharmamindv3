"""
Dharmic LLM Processor for DharmaMind platform

Processes LLM responses through dharmic principles to ensure spiritual alignment,
cultural sensitivity, and compassionate guidance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DharmicProcessingMode(str, Enum):
    """Processing modes for dharmic alignment"""
    GENTLE = "gentle"
    STANDARD = "standard"
    STRICT = "strict"
    COMPASSIONATE = "compassionate"
    WISDOM_FOCUSED = "wisdom_focused"

class DharmicPrinciple(str, Enum):
    """Core dharmic principles"""
    AHIMSA = "ahimsa"  # Non-violence
    SATYA = "satya"    # Truthfulness
    ASTEYA = "asteya"  # Non-stealing
    BRAHMACHARYA = "brahmacharya"  # Celibacy/moderation
    APARIGRAHA = "aparigraha"  # Non-possessiveness
    COMPASSION = "compassion"
    WISDOM = "wisdom"
    MINDFULNESS = "mindfulness"

class DharmicResponse(BaseModel):
    """Response processed through dharmic principles"""
    original_response: str = Field(..., description="Original LLM response")
    processed_response: str = Field(..., description="Dharmic-aligned response")
    dharmic_score: float = Field(..., ge=0.0, le=1.0, description="Dharmic alignment score")
    
    # Principle analysis
    principles_applied: List[DharmicPrinciple] = Field(default_factory=list, description="Applied principles")
    principle_scores: Dict[str, float] = Field(default_factory=dict, description="Individual principle scores")
    
    # Quality metrics
    compassion_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Compassion level")
    wisdom_depth: float = Field(default=0.0, ge=0.0, le=1.0, description="Wisdom depth")
    cultural_sensitivity: float = Field(default=0.0, ge=0.0, le=1.0, description="Cultural sensitivity")
    
    # Processing metadata
    processing_mode: DharmicProcessingMode = Field(..., description="Processing mode used")
    modifications_made: List[str] = Field(default_factory=list, description="Modifications applied")
    spiritual_insights: List[str] = Field(default_factory=list, description="Spiritual insights added")
    
    # Timestamps
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    
    class Config:
        """Pydantic configuration"""
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class DharmicLLMProcessor:
    """Processor for aligning LLM responses with dharmic principles"""
    
    def __init__(self):
        self.principle_weights = {
            DharmicPrinciple.AHIMSA: 1.0,
            DharmicPrinciple.SATYA: 0.9,
            DharmicPrinciple.COMPASSION: 1.0,
            DharmicPrinciple.WISDOM: 0.8,
            DharmicPrinciple.MINDFULNESS: 0.7,
        }
        
        self.harmful_patterns = [
            "violence", "hate", "discrimination", "harm", "hurt",
            "attack", "destroy", "kill", "evil", "revenge"
        ]
        
        self.dharmic_replacements = {
            "fight": "resolve peacefully",
            "defeat": "overcome with wisdom",
            "destroy": "transform",
            "hate": "find understanding for",
            "revenge": "forgiveness and healing",
            "anger": "inner peace through understanding"
        }
        
    async def initialize(self) -> bool:
        """Initialize the dharmic processor"""
        try:
            logger.info("Dharmic LLM Processor initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Dharmic LLM Processor: {e}")
            return False
    
    async def process_response(
        self,
        response: str,
        mode: DharmicProcessingMode = DharmicProcessingMode.STANDARD,
        context: Optional[Dict[str, Any]] = None
    ) -> DharmicResponse:
        """Process LLM response through dharmic principles"""
        try:
            original_response = response
            processed_response = response
            modifications = []
            spiritual_insights = []
            principles_applied = []
            principle_scores = {}
            
            # Apply dharmic filtering and enhancement
            processed_response, mods = await self._apply_ahimsa_filter(processed_response)
            modifications.extend(mods)
            if mods:
                principles_applied.append(DharmicPrinciple.AHIMSA)
                principle_scores[DharmicPrinciple.AHIMSA.value] = 0.9
            
            processed_response, insights = await self._add_spiritual_wisdom(processed_response, mode)
            spiritual_insights.extend(insights)
            
            processed_response, compassion_score = await self._enhance_compassion(processed_response)
            if compassion_score > 0.5:
                principles_applied.append(DharmicPrinciple.COMPASSION)
                principle_scores[DharmicPrinciple.COMPASSION.value] = compassion_score
            
            # Calculate dharmic score
            dharmic_score = await self._calculate_dharmic_score(processed_response, principles_applied)
            
            # Cultural sensitivity check
            cultural_sensitivity = await self._check_cultural_sensitivity(processed_response)
            
            # Wisdom depth analysis
            wisdom_depth = await self._analyze_wisdom_depth(processed_response)
            
            return DharmicResponse(
                original_response=original_response,
                processed_response=processed_response,
                dharmic_score=dharmic_score,
                principles_applied=principles_applied,
                principle_scores=principle_scores,
                compassion_level=compassion_score,
                wisdom_depth=wisdom_depth,
                cultural_sensitivity=cultural_sensitivity,
                processing_mode=mode,
                modifications_made=modifications,
                spiritual_insights=spiritual_insights
            )
            
        except Exception as e:
            logger.error(f"Error processing dharmic response: {e}")
            # Return original response with minimal processing
            return DharmicResponse(
                original_response=response,
                processed_response=response,
                dharmic_score=0.5,
                processing_mode=mode,
                modifications_made=["Error in processing - returned original"]
            )
    
    async def _apply_ahimsa_filter(self, response: str) -> tuple[str, List[str]]:
        """Apply non-violence (ahimsa) filtering"""
        modifications = []
        processed = response
        
        for pattern in self.harmful_patterns:
            if pattern.lower() in processed.lower():
                if pattern in self.dharmic_replacements:
                    processed = processed.replace(pattern, self.dharmic_replacements[pattern])
                    modifications.append(f"Replaced '{pattern}' with dharmic alternative")
        
        return processed, modifications
    
    async def _add_spiritual_wisdom(
        self, 
        response: str, 
        mode: DharmicProcessingMode
    ) -> tuple[str, List[str]]:
        """Add spiritual wisdom and insights"""
        insights = []
        
        if mode == DharmicProcessingMode.WISDOM_FOCUSED:
            # Add wisdom-focused insights
            wisdom_additions = [
                "Remember that all experiences are opportunities for spiritual growth.",
                "In dharmic tradition, we seek to respond rather than react.",
                "May this guidance serve your highest good and the welfare of all beings."
            ]
            
            # Randomly add one wisdom insight
            import random
            if random.random() > 0.7:  # 30% chance
                insight = random.choice(wisdom_additions)
                response += f"\n\n🕉️ {insight}"
                insights.append("Added spiritual wisdom insight")
        
        return response, insights
    
    async def _enhance_compassion(self, response: str) -> tuple[str, float]:
        """Enhance compassionate language"""
        compassion_words = ["understanding", "kindness", "gentle", "loving", "peaceful", "harmonious"]
        compassion_score = 0.0
        
        # Check for existing compassionate language
        for word in compassion_words:
            if word.lower() in response.lower():
                compassion_score += 0.1
        
        # Add compassionate framing if needed
        if compassion_score < 0.3:
            response = f"With loving-kindness and understanding, {response.lower()}"
            compassion_score += 0.4
        
        return response, min(compassion_score, 1.0)
    
    async def _calculate_dharmic_score(
        self, 
        response: str, 
        principles_applied: List[DharmicPrinciple]
    ) -> float:
        """Calculate overall dharmic alignment score"""
        base_score = 0.5
        
        # Bonus for applied principles
        principle_bonus = len(principles_applied) * 0.1
        
        # Check for dharmic keywords
        dharmic_keywords = [
            "dharma", "compassion", "wisdom", "mindfulness", "peace", 
            "harmony", "understanding", "loving-kindness", "enlightenment"
        ]
        
        keyword_bonus = 0.0
        for keyword in dharmic_keywords:
            if keyword.lower() in response.lower():
                keyword_bonus += 0.05
        
        # Check for harmful content (reduces score)
        harmful_penalty = 0.0
        for pattern in self.harmful_patterns:
            if pattern.lower() in response.lower():
                harmful_penalty += 0.1
        
        final_score = base_score + principle_bonus + keyword_bonus - harmful_penalty
        return max(0.0, min(1.0, final_score))
    
    async def _check_cultural_sensitivity(self, response: str) -> float:
        """Check cultural sensitivity of response"""
        # Simple heuristic - in production would use more sophisticated analysis
        sensitive_terms = ["tradition", "culture", "respect", "honor", "sacred"]
        score = 0.6  # Base score
        
        for term in sensitive_terms:
            if term.lower() in response.lower():
                score += 0.08
        
        return min(1.0, score)
    
    async def _analyze_wisdom_depth(self, response: str) -> float:
        """Analyze depth of wisdom in response"""
        wisdom_indicators = [
            "ancient", "tradition", "teaching", "wisdom", "insight", 
            "understanding", "realization", "enlightenment", "truth"
        ]
        
        depth_score = 0.3  # Base score
        
        for indicator in wisdom_indicators:
            if indicator.lower() in response.lower():
                depth_score += 0.07
        
        # Bonus for spiritual concepts
        if any(word in response.lower() for word in ["dharma", "karma", "moksha", "nirvana"]):
            depth_score += 0.2
        
        return min(1.0, depth_score)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check processor health"""
        return {
            "status": "healthy",
            "processor": "dharmic_llm",
            "principles_supported": len(self.principle_weights),
            "filtering_patterns": len(self.harmful_patterns)
        }

# Global processor instance
_dharmic_processor: Optional[DharmicLLMProcessor] = None

async def get_dharmic_llm_processor() -> DharmicLLMProcessor:
    """Get the global dharmic LLM processor instance"""
    global _dharmic_processor
    
    if _dharmic_processor is None:
        _dharmic_processor = DharmicLLMProcessor()
        await _dharmic_processor.initialize()
    
    return _dharmic_processor

def get_dharmic_processor_sync() -> DharmicLLMProcessor:
    """Get dharmic processor instance synchronously (may not be initialized)"""
    global _dharmic_processor
    
    if _dharmic_processor is None:
        _dharmic_processor = DharmicLLMProcessor()
    
    return _dharmic_processor