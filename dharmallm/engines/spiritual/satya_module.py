"""
🔵 Satya Module - Truth and Honesty
Complete Satya (Truth) system based on dharmic principles
Provides guidance on practicing truthfulness, integrity, and authentic living
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TruthLevel(Enum):
    """Levels of truth practice"""
    FACTUAL = "factual_accuracy"     # Basic factual honesty
    EMOTIONAL = "emotional_honesty"   # Honest about feelings
    SPIRITUAL = "spiritual_truth"     # Alignment with dharma
    ABSOLUTE = "ultimate_truth"       # Recognition of ultimate reality

class TruthContext(Enum):
    """Contexts where truth applies"""
    PERSONAL = "self_honesty"         # Honesty with oneself
    INTERPERSONAL = "relationship"    # Truth in relationships
    PROFESSIONAL = "work_ethics"      # Honesty in work/business
    SOCIAL = "community_truth"        # Truth in social situations
    SPIRITUAL = "dharmic_truth"       # Truth in spiritual practice

class UntruthType(Enum):
    """Types of untruth according to dharmic understanding"""
    FALSEHOOD = "deliberate_lie"      # Intentional false statement
    DECEPTION = "misleading"          # Technically true but misleading
    OMISSION = "withholding_truth"    # Not revealing important truth
    EXAGGERATION = "distortion"       # Amplifying or minimizing truth
    SELF_DECEPTION = "denial"         # Lying to oneself
    HARMFUL_TRUTH = "truth_as_weapon" # Using truth to cause harm

@dataclass
class TruthInsight:
    """An insight about practicing truthfulness"""
    situation: str
    truth_level: TruthLevel
    ethical_consideration: str
    recommended_approach: str
    potential_consequences: List[str]
    dharmic_principle: str
    supporting_wisdom: List[str]
    timestamp: datetime

@dataclass
class TruthDilemma:
    """Framework for handling truth dilemmas"""
    dilemma_type: str
    conflicting_values: List[str]
    stakeholders: List[str]
    dharmic_guidelines: List[str]
    recommended_approach: str
    communication_strategy: str
    long_term_considerations: List[str]

class SatyaInsight(BaseModel):
    """Insight from satya module"""
    truth_level: str = Field(description="Level of truth practice needed")
    ethical_consideration: str = Field(description="Ethical consideration")
    recommended_approach: str = Field(description="Recommended approach")
    dharmic_principle: str = Field(description="Relevant dharmic principle")

class SatyaResponse(BaseModel):
    """Response from Satya module"""
    truth_context: str = Field(description="Context where truth applies")
    ethical_challenges: List[str] = Field(description="Identified ethical challenges")
    truth_guidance: str = Field(description="Specific guidance for truthful action")
    communication_strategy: Dict[str, Any] = Field(description="How to communicate truthfully")
    dharmic_principles: List[str] = Field(description="Relevant dharmic principles")
    practical_steps: List[str] = Field(description="Practical steps for truth practice")
    potential_outcomes: Dict[str, List[str]] = Field(description="Potential outcomes of truth-telling")
    daily_practice: Dict[str, Any] = Field(description="Daily truth practice suggestions")

class SatyaModule:
    """
    🔵 Satya Module - The Path of Truth and Integrity
    
    Provides guidance on practicing truthfulness in all aspects of life,
    maintaining integrity, and navigating complex situations where truth
    and compassion must be balanced according to dharmic principles.
    """
    
    def __init__(self):
        self.name = "Satya"
        self.color = "🔵"
        self.element = "Truth/Integrity"
        self.mantra = "SATYAM"  # Truth
        self.deity = "Saraswati"  # Goddess of wisdom and truth
        self.principles = ["Truthfulness", "Integrity", "Authenticity", "Honest Communication"]
        self.truth_frameworks = self._initialize_truth_frameworks()
        self.ethical_guidelines = self._initialize_ethical_guidelines()
        self.communication_strategies = self._initialize_communication_strategies()
        
    def _initialize_truth_frameworks(self) -> Dict[TruthContext, Dict[str, Any]]:
        """Initialize truth practice frameworks"""
        return {
            TruthContext.PERSONAL: {
                "key_practices": [
                    "Regular self-reflection and honest self-assessment",
                    "Acknowledging personal limitations and mistakes",
                    "Being authentic to your values and beliefs",
                    "Facing uncomfortable truths about yourself"
                ],
                "common_challenges": [
                    "Self-deception and denial",
                    "Fear of confronting difficult truths",
                    "Ego protection mechanisms",
                    "Rationalization of harmful behaviors"
                ],
                "dharmic_guidance": "Svadharma - Be true to your authentic nature and purpose"
            },
            TruthContext.INTERPERSONAL: {
                "key_practices": [
                    "Honest communication in relationships",
                    "Expressing feelings authentically but kindly",
                    "Keeping promises and commitments",
                    "Admitting when you don't know something"
                ],
                "common_challenges": [
                    "Fear of hurting others' feelings",
                    "Conflict avoidance",
                    "People-pleasing tendencies",
                    "White lies to avoid discomfort"
                ],
                "dharmic_guidance": "Satyam vada, priyam vada - Speak truth, speak kindly"
            },
            TruthContext.PROFESSIONAL: {
                "key_practices": [
                    "Honest representation of skills and experience",
                    "Transparent business practices",
                    "Acknowledging errors and taking responsibility",
                    "Fair dealing with clients and colleagues"
                ],
                "common_challenges": [
                    "Pressure to exaggerate achievements",
                    "Competitive environments encouraging deception",
                    "Conflicts between profit and honesty",
                    "Fear of professional consequences"
                ],
                "dharmic_guidance": "Dharmic wealth - Success achieved through righteous means"
            },
            TruthContext.SOCIAL: {
                "key_practices": [
                    "Standing up for truth in group settings",
                    "Not spreading gossip or unverified information",
                    "Being genuine rather than putting on facades",
                    "Correcting misinformation when encountered"
                ],
                "common_challenges": [
                    "Social pressure to conform",
                    "Fear of being ostracized",
                    "Gossip and social dynamics",
                    "Balancing honesty with social harmony"
                ],
                "dharmic_guidance": "Dharmic duty to uphold truth in society"
            },
            TruthContext.SPIRITUAL: {
                "key_practices": [
                    "Honest spiritual seeking without pretense",
                    "Authentic practice rather than spiritual materialism",
                    "Acknowledging spiritual experiences honestly",
                    "Being truthful about your spiritual development"
                ],
                "common_challenges": [
                    "Spiritual ego and pretense",
                    "Exaggerating spiritual experiences",
                    "Using spirituality to avoid responsibility",
                    "Comparing spiritual progress with others"
                ],
                "dharmic_guidance": "Sat-chit-ananda - Truth, consciousness, bliss as the ultimate reality"
            }
        }
    
    def _initialize_ethical_guidelines(self) -> List[TruthDilemma]:
        """Initialize ethical guidelines for truth dilemmas"""
        return [
            TruthDilemma(
                dilemma_type="Truth vs. Kindness",
                conflicting_values=["Honesty", "Compassion", "Non-harm"],
                stakeholders=["Speaker", "Listener", "Community"],
                dharmic_guidelines=[
                    "Truth should be spoken with compassion",
                    "Consider the motivation behind speaking truth",
                    "Timing and manner matter as much as content",
                    "Sometimes silence is more dharmic than harmful truth"
                ],
                recommended_approach="Find compassionate ways to convey necessary truths",
                communication_strategy="Use loving speech while maintaining honesty",
                long_term_considerations=["Relationship preservation", "Personal integrity", "Greater good"]
            ),
            TruthDilemma(
                dilemma_type="Personal Truth vs. Social Expectations",
                conflicting_values=["Authenticity", "Social harmony", "Family expectations"],
                stakeholders=["Individual", "Family", "Community"],
                dharmic_guidelines=[
                    "Honor your svadharma (personal duty/nature)",
                    "Consider impact on family and community",
                    "Seek ways to be authentic without causing unnecessary harm",
                    "Sometimes gradual truth-telling is more skillful"
                ],
                recommended_approach="Balance personal authenticity with compassionate consideration of others",
                communication_strategy="Honest dialogue with patience and understanding",
                long_term_considerations=["Personal happiness", "Family relationships", "Societal progress"]
            ),
            TruthDilemma(
                dilemma_type="Professional Ethics vs. Loyalty",
                conflicting_values=["Professional integrity", "Loyalty to employer", "Financial security"],
                stakeholders=["Professional", "Employer", "Clients", "Society"],
                dharmic_guidelines=[
                    "Dharmic principles supersede personal gain",
                    "Consider long-term consequences of dishonesty",
                    "Seek solutions that uphold both integrity and relationships",
                    "Sometimes sacrificing immediate gain serves higher purpose"
                ],
                recommended_approach="Find ethical solutions that honor professional standards",
                communication_strategy="Clear, respectful communication about ethical boundaries",
                long_term_considerations=["Professional reputation", "Karmic consequences", "Societal trust"]
            )
        ]
    
    def _initialize_communication_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize truth communication strategies"""
        return {
            "compassionate_honesty": {
                "approach": "Speak truth with love and understanding",
                "techniques": [
                    "Use 'I' statements to express personal truth",
                    "Acknowledge the other person's perspective",
                    "Choose timing when the person can best receive truth",
                    "Offer support along with difficult truths"
                ],
                "example_phrases": [
                    "I need to share something that's important to me...",
                    "I care about you, which is why I want to be honest...",
                    "This is difficult to say, but I believe you deserve to know..."
                ]
            },
            "constructive_feedback": {
                "approach": "Truth that helps rather than hurts",
                "techniques": [
                    "Focus on specific behaviors, not character judgments",
                    "Offer concrete suggestions for improvement",
                    "Balance criticism with recognition of strengths",
                    "Create safe space for dialogue"
                ],
                "example_phrases": [
                    "I noticed that... and I'm wondering if...",
                    "What I observed was... How do you see it?",
                    "Could we explore together how to..."
                ]
            },
            "difficult_truths": {
                "approach": "Necessary truths delivered with wisdom",
                "techniques": [
                    "Prepare the person for difficult news",
                    "Provide context and reasoning",
                    "Offer emotional support and next steps",
                    "Allow time for processing and questions"
                ],
                "example_phrases": [
                    "I have something difficult to share...",
                    "This may be hard to hear, but...",
                    "I'm here to support you through this..."
                ]
            }
        }
    
    def identify_truth_context(self, situation: str) -> Optional[TruthContext]:
        """Identify the primary truth context"""
        situation_lower = situation.lower()
        
        if any(word in situation_lower for word in ["relationship", "family", "friend", "partner"]):
            return TruthContext.INTERPERSONAL
        elif any(word in situation_lower for word in ["work", "job", "business", "professional", "career"]):
            return TruthContext.PROFESSIONAL
        elif any(word in situation_lower for word in ["myself", "self", "personal", "inner"]):
            return TruthContext.PERSONAL
        elif any(word in situation_lower for word in ["spiritual", "dharma", "meditation", "practice"]):
            return TruthContext.SPIRITUAL
        elif any(word in situation_lower for word in ["social", "community", "group", "society"]):
            return TruthContext.SOCIAL
        else:
            return TruthContext.PERSONAL  # Default
    
    def identify_ethical_challenges(self, situation: str, context: Dict[str, Any]) -> List[str]:
        """Identify ethical challenges in the situation"""
        challenges = []
        situation_lower = situation.lower()
        
        if any(word in situation_lower for word in ["hurt", "feelings", "sensitive"]):
            challenges.append("Balancing truth with kindness")
        
        if any(word in situation_lower for word in ["conflict", "disagree", "different"]):
            challenges.append("Managing conflicting perspectives")
        
        if any(word in situation_lower for word in ["secret", "private", "confidential"]):
            challenges.append("Respecting confidentiality")
        
        if any(word in situation_lower for word in ["mistake", "error", "wrong"]):
            challenges.append("Taking responsibility")
        
        if context.get("power_imbalance", False):
            challenges.append("Speaking truth to authority")
        
        if context.get("family_involved", False):
            challenges.append("Family dynamics and expectations")
        
        return challenges if challenges else ["Maintaining integrity while being compassionate"]
    
    def generate_truth_guidance(self, situation: str, truth_context: TruthContext, 
                              challenges: List[str], context: Dict[str, Any]) -> str:
        """Generate specific truth guidance for the situation"""
        base_guidance = "🔵 The path of satya (truth) guides us to align our thoughts, words, and actions with dharmic principles. "
        
        framework = self.truth_frameworks.get(truth_context, {})
        specific_guidance = []
        
        # Add context-specific guidance
        dharmic_guidance = framework.get("dharmic_guidance", "")
        if dharmic_guidance:
            specific_guidance.append(f"Remember: {dharmic_guidance}.")
        
        # Address specific challenges
        if "Balancing truth with kindness" in challenges:
            specific_guidance.append("Consider both the truth and the manner of its delivery. Ask yourself: 'Is this truth necessary? Will it help? How can I speak it with love?'")
        
        if "Taking responsibility" in challenges:
            specific_guidance.append("Taking responsibility for our actions is a dharmic duty. Acknowledge mistakes honestly and focus on making amends and learning.")
        
        if "Family dynamics and expectations" in challenges:
            specific_guidance.append("Honor your family while remaining true to your authentic self. Seek dialogue and understanding rather than confrontation.")
        
        return base_guidance + " ".join(specific_guidance)
    
    def recommend_communication_strategy(self, situation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend communication strategy based on situation"""
        # Determine most appropriate strategy
        if any(word in situation.lower() for word in ["difficult", "bad news", "disappointing"]):
            return self.communication_strategies["difficult_truths"]
        elif any(word in situation.lower() for word in ["feedback", "improvement", "performance"]):
            return self.communication_strategies["constructive_feedback"]
        else:
            return self.communication_strategies["compassionate_honesty"]
    
    def get_relevant_principles(self, truth_context: TruthContext) -> List[str]:
        """Get relevant dharmic principles for truth practice"""
        universal_principles = [
            "Satyam eva jayate - Truth alone triumphs",
            "Satyam vada, priyam vada - Speak truth, speak kindly",
            "Dharma eva hato hanti - Unrighteousness destroys the unrighteous"
        ]
        
        if truth_context == TruthContext.PERSONAL:
            universal_principles.append("Atmanam viddhi - Know thyself")
        elif truth_context == TruthContext.INTERPERSONAL:
            universal_principles.append("Matru devo bhava - Honor truth in relationships")
        elif truth_context == TruthContext.PROFESSIONAL:
            universal_principles.append("Karmasu kaushalam - Excellence in action through integrity")
        elif truth_context == TruthContext.SPIRITUAL:
            universal_principles.append("Sat-chit-ananda - Truth, consciousness, bliss")
        
        return universal_principles
    
    def generate_practical_steps(self, situation: str, truth_context: TruthContext) -> List[str]:
        """Generate practical steps for truth practice"""
        universal_steps = [
            "Pause and reflect on your intention before speaking",
            "Consider the impact of your words on all involved",
            "Choose language that is both honest and compassionate",
            "Be prepared to take responsibility for the consequences"
        ]
        
        framework = self.truth_frameworks.get(truth_context, {})
        practices = framework.get("key_practices", [])
        if practices:
            universal_steps.extend(practices[:2])  # Add first two context-specific practices
        
        return universal_steps
    
    def analyze_potential_outcomes(self, situation: str, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze potential outcomes of truth-telling"""
        return {
            "positive_outcomes": [
                "Increased trust and credibility",
                "Relief from burden of deception",
                "Stronger, more authentic relationships",
                "Personal integrity and self-respect",
                "Setting positive example for others"
            ],
            "potential_challenges": [
                "Initial discomfort or conflict",
                "Need to repair any damage caused",
                "Possible resistance from others",
                "Temporary relationship strain",
                "Requiring courage to maintain truth"
            ],
            "long_term_benefits": [
                "Deeper, more meaningful relationships",
                "Reduced stress from maintaining deceptions",
                "Clear conscience and inner peace",
                "Reputation for integrity and trustworthiness",
                "Positive karma from dharmic action"
            ]
        }
    
    def get_daily_truth_practice(self, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provide daily truth practice suggestions"""
        context = user_context or {}
        
        return {
            "morning_intention": [
                "Set intention to speak and act truthfully throughout the day",
                "Reflect on any areas where you've been avoiding truth",
                "Ask for courage to face difficult truths with compassion"
            ],
            "throughout_day": [
                "Before speaking, pause and consider: 'Is this true? Is it necessary? Is it kind?'",
                "Notice when you're tempted to exaggerate, omit, or deceive",
                "Practice authentic expression of thoughts and feelings",
                "Take responsibility promptly when you make mistakes"
            ],
            "evening_reflection": [
                "Review the day for moments when you chose truth or avoidance",
                "Acknowledge any deceptions and commit to correction if needed",
                "Appreciate moments when truth strengthened relationships",
                "Set intention for greater authenticity tomorrow"
            ],
            "weekly_practice": "Choose one relationship to practice deeper honesty",
            "monthly_goal": "Address one area of self-deception or avoidance",
            "reminder": "Truth is not just about words - it's about living authentically and with integrity"
        }
    
    async def process_satya_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> SatyaResponse:
        """Process satya-related query and provide guidance"""
        try:
            context = user_context or {}
            
            # Analyze the truth context
            truth_context = self.identify_truth_context(query)
            if not truth_context:
                truth_context = TruthContext.PERSONAL  # Default fallback
            
            # Detect potential ethical dilemmas
            ethical_challenges = self.identify_ethical_challenges(query, context)
            
            # Generate truth guidance
            truth_guidance = self.generate_truth_guidance(query, truth_context, ethical_challenges, context)
            
            # Provide communication strategy
            communication_strategy = self.recommend_communication_strategy(query, context)
            
            # Get dharmic principles
            dharmic_principles = self.get_relevant_principles(truth_context)
            
            # Generate practical steps
            practical_steps = self.generate_practical_steps(query, truth_context)
            
            # Analyze potential outcomes
            potential_outcomes = self.analyze_potential_outcomes(query, context)
            
            # Get daily practice
            daily_practice = self.get_daily_truth_practice(context)
            
            return SatyaResponse(
                truth_context=truth_context.value,
                ethical_challenges=ethical_challenges,
                truth_guidance=truth_guidance,
                communication_strategy=communication_strategy,
                dharmic_principles=dharmic_principles,
                practical_steps=practical_steps,
                potential_outcomes=potential_outcomes,
                daily_practice=daily_practice
            )
            
        except Exception as e:
            logger.error(f"Error processing satya query: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> SatyaResponse:
        """Create fallback response when processing fails"""
        return SatyaResponse(
            truth_context="personal",
            ethical_challenges=["Maintaining integrity while being compassionate"],
            truth_guidance="🔵 The path of satya guides us to align thoughts, words, and actions with dharmic principles. Speak truth with compassion and wisdom.",
            communication_strategy=self.communication_strategies["compassionate_honesty"],
            dharmic_principles=[
                "Satyam eva jayate - Truth alone triumphs",
                "Satyam vada, priyam vada - Speak truth, speak kindly"
            ],
            practical_steps=[
                "Pause and reflect on your intention before speaking",
                "Consider the impact of your words on all involved",
                "Choose language that is both honest and compassionate",
                "Be prepared to take responsibility for the consequences"
            ],
            potential_outcomes={
                "positive_outcomes": ["Increased trust and integrity", "Relief from deception", "Authentic relationships"],
                "potential_challenges": ["Initial discomfort", "Need for courage"],
                "long_term_benefits": ["Clear conscience", "Trustworthy reputation", "Inner peace"]
            },
            daily_practice=self.get_daily_truth_practice()
        )
    
    def get_satya_insight(self, situation: str) -> Optional[SatyaInsight]:
        """Get specific insight about truth practice in a situation"""
        truth_context = self.identify_truth_context(situation)
        if not truth_context:
            truth_context = TruthContext.PERSONAL  # Default fallback
            
        framework = self.truth_frameworks.get(truth_context, {})
        
        return SatyaInsight(
            truth_level=TruthLevel.EMOTIONAL.value,  # Default appropriate level
            ethical_consideration="Balance honesty with compassion",
            recommended_approach="Speak truth with love and understanding",
            dharmic_principle=framework.get("dharmic_guidance", "Satyam vada, priyam vada - Speak truth, speak kindly")
        )

# Global instance
_satya_module = None

def get_satya_module() -> SatyaModule:
    """Get global Satya module instance"""
    global _satya_module
    if _satya_module is None:
        _satya_module = SatyaModule()
    return _satya_module

# Factory function for easy access
def create_satya_guidance(query: str, user_context: Optional[Dict[str, Any]] = None) -> SatyaResponse:
    """Factory function to create satya guidance"""
    import asyncio
    module = get_satya_module()
    return asyncio.run(module.process_satya_query(query, user_context))
