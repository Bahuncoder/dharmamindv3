"""
DharmaMind Conversational AI - ChatGPT Style with Dharmic Wisdom

This creates a conversational AI that responds like ChatGPT but is powered by:
- Hindu dharmic knowledge and wisdom
- Chakra module guidance for holistic responses
- Personal growth and spiritual guidance
- Ethical living principles
- Universal language for all humanity

The AI acts as a personal deep guide powered by dharma and Hinduism knowledge.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import time
from datetime import datetime

from ...services.llm_router import get_llm_router
from ...chakra_modules import (
    get_consciousness_core, get_knowledge_base, get_emotional_intelligence,
    get_dharma_engine, get_ai_core
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dharmic", tags=["Dharmic Chat"])

class DharmicChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "seeker"
    include_personal_growth: Optional[bool] = True
    include_spiritual_guidance: Optional[bool] = True
    include_ethical_guidance: Optional[bool] = True
    response_style: Optional[str] = "conversational"  # conversational, wise, practical

class DharmicChatResponse(BaseModel):
    response: str
    conversation_id: str
    dharmic_insights: List[str]
    growth_suggestions: List[str]
    spiritual_context: str
    ethical_guidance: str
    conversation_style: str
    processing_info: Dict[str, Any]

class DharmicConversationalAI:
    """
    Conversational AI that responds like ChatGPT but with dharmic wisdom
    """
    
    def __init__(self):
        self.conversation_contexts = {}
        
    async def generate_dharmic_response(self, request: DharmicChatRequest) -> DharmicChatResponse:
        """Generate a ChatGPT-style response with dharmic wisdom"""
        start_time = time.time()
        
        try:
            # Step 1: Get Chakra module insights
            consciousness_core = get_consciousness_core()
            knowledge_base = get_knowledge_base()
            emotional_intelligence = get_emotional_intelligence()
            dharma_engine = get_dharma_engine()
            ai_core = get_ai_core()
            
            # Step 2: Analyze user message for spiritual and ethical context
            user_context = await self._analyze_user_context(request.message)
            
            # Step 3: Get dharmic knowledge relevant to the query
            dharmic_knowledge = await self._get_relevant_dharmic_knowledge(request.message, knowledge_base)
            
            # Step 4: Create conversational prompt with dharmic guidance
            conversational_prompt = await self._create_dharmic_prompt(
                user_message=request.message,
                user_context=user_context,
                dharmic_knowledge=dharmic_knowledge,
                response_style=request.response_style
            )
            
            # Step 5: Generate response using local LLM with dharmic context
            from ...services.local_llm import get_local_llm_service
            local_llm = await get_local_llm_service()
            
            response_data = await local_llm.generate_response(
                message=conversational_prompt,
                model_name="distilgpt2",
                max_length=512,
                temperature=0.7
            )
            
            # Step 6: Enhance response with Chakra module wisdom
            enhanced_response = await self._enhance_with_chakra_wisdom(
                response_data.get("content", ""),
                request,
                user_context
            )
            
            # Step 7: Generate practical guidance
            growth_suggestions = await self._generate_growth_suggestions(request.message, user_context)
            ethical_guidance = await self._generate_ethical_guidance(request.message, dharma_engine)
            spiritual_context = await self._generate_spiritual_context(request.message, knowledge_base)
            
            processing_time = time.time() - start_time
            
            return DharmicChatResponse(
                response=enhanced_response,
                conversation_id=request.conversation_id or f"dharmic-{int(time.time())}",
                dharmic_insights=await self._extract_dharmic_insights(dharmic_knowledge),
                growth_suggestions=growth_suggestions,
                spiritual_context=spiritual_context,
                ethical_guidance=ethical_guidance,
                conversation_style=request.response_style,
                processing_info={
                    "processing_time": processing_time,
                    "chakra_modules_used": ["consciousness", "knowledge", "emotional", "dharma", "ai"],
                    "model_used": response_data.get("model_name", "local-dharmic-ai"),
                    "dharmic_enhancement": True
                }
            )
            
        except Exception as e:
            logger.error(f"Dharmic chat error: {e}")
            raise HTTPException(status_code=500, detail=f"Dharmic AI error: {str(e)}")
    
    async def _analyze_user_context(self, message: str) -> Dict[str, Any]:
        """Analyze user message for emotional and spiritual context"""
        context = {
            "emotional_state": "neutral",
            "spiritual_seeking": False,
            "personal_growth": False,
            "ethical_concern": False,
            "life_challenge": False,
            "wisdom_seeking": False
        }
        
        message_lower = message.lower()
        
        # Detect emotional state
        if any(word in message_lower for word in ["sad", "depressed", "anxious", "worried", "fear"]):
            context["emotional_state"] = "struggling"
        elif any(word in message_lower for word in ["happy", "grateful", "blessed", "joy"]):
            context["emotional_state"] = "positive"
        elif any(word in message_lower for word in ["confused", "lost", "uncertain"]):
            context["emotional_state"] = "seeking_clarity"
            
        # Detect spiritual seeking
        context["spiritual_seeking"] = any(word in message_lower for word in [
            "spiritual", "meditation", "consciousness", "enlightenment", "awakening",
            "dharma", "karma", "moksha", "atman", "brahman", "yoga", "mantra"
        ])
        
        # Detect personal growth interest
        context["personal_growth"] = any(word in message_lower for word in [
            "grow", "develop", "improve", "better", "change", "transform", "evolve"
        ])
        
        # Detect ethical concerns
        context["ethical_concern"] = any(word in message_lower for word in [
            "right", "wrong", "ethical", "moral", "should", "virtue", "duty", "responsibility"
        ])
        
        # Detect life challenges
        context["life_challenge"] = any(word in message_lower for word in [
            "problem", "challenge", "difficulty", "struggle", "help", "advice", "guidance"
        ])
        
        # Detect wisdom seeking
        context["wisdom_seeking"] = any(word in message_lower for word in [
            "wisdom", "truth", "meaning", "purpose", "understanding", "insight", "knowledge"
        ])
        
        return context
    
    async def _get_relevant_dharmic_knowledge(self, message: str, knowledge_base) -> Dict[str, str]:
        """Get relevant Hindu dharmic knowledge for the query"""
        
        # Use knowledge base to get relevant dharmic concepts
        try:
            if hasattr(knowledge_base, 'search_knowledge'):
                relevant_knowledge = await knowledge_base.search_knowledge(message)
            else:
                # Fallback to predefined dharmic wisdom
                relevant_knowledge = self._get_default_dharmic_wisdom(message)
            
            return relevant_knowledge
        except Exception as e:
            logger.warning(f"Could not get dharmic knowledge: {e}")
            return self._get_default_dharmic_wisdom(message)
    
    def _get_default_dharmic_wisdom(self, message: str) -> Dict[str, str]:
        """Fallback dharmic wisdom based on message content"""
        message_lower = message.lower()
        
        dharmic_wisdom = {}
        
        if any(word in message_lower for word in ["purpose", "meaning", "life"]):
            dharmic_wisdom["life_purpose"] = "According to Hindu dharma, life's purpose (purushartha) includes dharma (righteous living), artha (prosperity), kama (pleasure), and moksha (liberation)."
        
        if any(word in message_lower for word in ["suffering", "pain", "difficult"]):
            dharmic_wisdom["dealing_with_suffering"] = "The Bhagavad Gita teaches that suffering comes from attachment. Practice detachment (vairagya) while performing your duty (dharma)."
        
        if any(word in message_lower for word in ["decision", "choice", "confused"]):
            dharmic_wisdom["decision_making"] = "Follow your dharma - your righteous duty. Consider what is right for your stage of life (ashrama) and your nature (svabhava)."
        
        if any(word in message_lower for word in ["relationship", "family", "love"]):
            dharmic_wisdom["relationships"] = "Treat all beings with love and compassion. Remember that all souls are connected through the universal consciousness (Brahman)."
        
        if any(word in message_lower for word in ["success", "failure", "achievement"]):
            dharmic_wisdom["success_and_failure"] = "Focus on your actions (karma) not the results. Do your duty without attachment to outcomes - this is karma yoga."
        
        return dharmic_wisdom
    
    async def _create_dharmic_prompt(self, user_message: str, user_context: Dict, dharmic_knowledge: Dict, response_style: str) -> str:
        """Create a conversational prompt with dharmic guidance"""
        
        # Simplified prompt for local LLM
        if user_context.get("spiritual_seeking") or any(word in user_message.lower() for word in ["purpose", "meaning", "lost"]):
            prompt = f"Question: {user_message}\n\nAnswer: According to Hindu wisdom and dharma, "
        else:
            prompt = f"Question: {user_message}\n\nAnswer: "
        
        return prompt
    
    def _format_dharmic_knowledge(self, dharmic_knowledge: Dict[str, str]) -> str:
        """Format dharmic knowledge for the prompt"""
        if not dharmic_knowledge:
            return "- Universal principles of compassion, truth, and righteous living"
        
        formatted = []
        for key, value in dharmic_knowledge.items():
            formatted.append(f"- {value}")
        
        return "\n".join(formatted)
    
    async def _enhance_with_chakra_wisdom(self, response: str, request: DharmicChatRequest, user_context: Dict) -> str:
        """Enhance the response with Chakra module wisdom"""
        
        # If response is empty or too short, create a dharmic response
        if not response or len(response.strip()) < 10:
            response = await self._generate_dharmic_response_direct(request.message, user_context)
        
        enhanced_response = response
        
        # Add dharmic wisdom based on context
        if user_context.get("spiritual_seeking"):
            enhanced_response += f"\n\n🕉️ From a spiritual perspective: Your seeking itself is a sacred journey. In Hindu philosophy, the very desire for spiritual growth shows the awakening of the soul (atman) towards its true nature."
        
        if user_context.get("life_challenge"):
            enhanced_response += f"\n\n� The Bhagavad Gita teaches us that challenges are opportunities for growth. When we face difficulties with dharmic principles - truth, compassion, and detachment - we transform suffering into wisdom."
        
        if user_context.get("wisdom_seeking"):
            enhanced_response += f"\n\n📚 Ancient wisdom reminds us: 'Tat tvam asi' - That thou art. You are not separate from the universal consciousness. The answers you seek are already within you."
        
        # Add practical dharmic guidance
        if request.include_personal_growth and user_context.get("personal_growth"):
            enhanced_response += f"\n\n🌱 For personal growth: Practice daily self-reflection (svadhyaya), act according to your dharma, and remember that every action is an opportunity to express your highest nature."
        
        # Add ethical guidance if relevant
        if request.include_ethical_guidance and user_context.get("ethical_concern"):
            enhanced_response += f"\n\n⚖️ Dharmic guidance: When in doubt about right action, ask yourself: 'What would bring the greatest good to all beings?' Let compassion and truth guide your decisions."
        
        return enhanced_response
    
    async def _generate_dharmic_response_direct(self, message: str, user_context: Dict) -> str:
        """Generate a direct dharmic response when LLM fails"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["purpose", "meaning", "lost"]):
            return "Your life's purpose, according to Hindu dharma, is multifaceted: to live righteously (dharma), to achieve prosperity through ethical means (artha), to experience joy and fulfillment (kama), and ultimately to realize your true spiritual nature (moksha). You are not lost - you are on a journey of self-discovery."
        
        elif any(word in message_lower for word in ["stress", "anxiety", "worry", "overwhelmed"]):
            return "Stress and anxiety are signals that the mind is attached to outcomes beyond our control. Hindu wisdom teaches us pranayama (breathing practices), meditation, and the principle of karma yoga - performing our duties without attachment to results. Practice deep breathing, remember the impermanent nature of all challenges, and focus on what you can control: your actions and responses."
        
        elif any(word in message_lower for word in ["suffering", "pain", "difficult", "hard"]):
            return "Suffering, while difficult, serves a sacred purpose in Hindu philosophy. It teaches us detachment, compassion, and the impermanent nature of all experiences. The Bhagavad Gita reminds us to perform our duty without attachment to results, finding peace in action itself. This too shall pass, and you will emerge stronger and wiser."
        
        elif any(word in message_lower for word in ["relationship", "love", "family", "marriage"]):
            return "Relationships are sacred opportunities to practice dharma. Hindu teachings emphasize that love is the recognition of the divine in all beings. Treat others with compassion, speak truthfully but kindly, respect their journey, and remember that all souls are interconnected through universal consciousness. Practice patience, forgiveness, and unconditional love."
        
        elif any(word in message_lower for word in ["success", "failure", "career", "work", "job"]):
            return "True success, according to dharmic principles, is not just material achievement but the cultivation of wisdom, compassion, and inner peace. Focus on performing your duty (svadharma) with dedication, excellence, and integrity. Let go of attachment to outcomes and trust that right action leads to right results. Your work is worship when done with consciousness."
        
        elif any(word in message_lower for word in ["meditation", "spiritual", "consciousness", "enlightenment"]):
            return "Your spiritual inquiry is beautiful and sacred. Meditation is the practice of returning to your true Self. Start with simple breath awareness, practice daily self-reflection (svadhyaya), and remember that enlightenment is not a destination but a recognition of what you already are - pure consciousness experiencing itself. Be patient and compassionate with yourself on this journey."
        
        elif any(word in message_lower for word in ["decision", "choice", "confused", "doubt"]):
            return "When facing decisions, Hindu wisdom guides us to consider dharma (righteousness), listen to our inner voice (antardarshan), and act from a place of love rather than fear. Ask yourself: 'What action would benefit all beings?' Trust your intuition, seek wise counsel, and remember that even 'wrong' choices become learning experiences that guide you closer to truth."
        
        elif any(word in message_lower for word in ["health", "sick", "illness", "body"]):
            return "The body is a sacred temple housing your divine consciousness. Hindu tradition teaches that health comes from balance in all aspects of life - proper diet (ahara), right action (achara), positive thinking (vichara), and spiritual practice (sadhana). Listen to your body with compassion, seek appropriate care, and remember that physical challenges often carry deeper teachings for the soul."
        
        else:
            return "Every question you ask is a step on your spiritual journey. Hindu philosophy teaches that you are both the seeker and the sought - a divine consciousness experiencing itself through human form. Trust in your inner wisdom, act with dharma, and remember that the divine within you is always guiding you toward truth, love, and liberation."
    
    async def _generate_growth_suggestions(self, message: str, user_context: Dict) -> List[str]:
        """Generate practical growth suggestions"""
        suggestions = []
        
        if user_context.get("spiritual_seeking"):
            suggestions.append("Practice daily meditation or mindfulness to deepen your spiritual connection")
            suggestions.append("Study sacred texts like the Bhagavad Gita for timeless wisdom")
        
        if user_context.get("personal_growth"):
            suggestions.append("Reflect on your dharma - your unique purpose and duties in life")
            suggestions.append("Practice self-inquiry: 'Who am I beyond my roles and identities?'")
        
        if user_context.get("emotional_state") == "struggling":
            suggestions.append("Practice pranayama (breathing exercises) to calm your mind")
            suggestions.append("Remember the principle of impermanence - all states of mind pass")
        
        if not suggestions:
            suggestions.append("Live each day with intention, compassion, and mindfulness")
            suggestions.append("Treat every interaction as an opportunity to practice love and kindness")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    async def _generate_ethical_guidance(self, message: str, dharma_engine) -> str:
        """Generate ethical guidance using dharma engine"""
        try:
            if hasattr(dharma_engine, 'get_ethical_guidance'):
                return await dharma_engine.get_ethical_guidance(message)
            else:
                return "Follow the path of dharma: speak truth, act with compassion, and serve the greater good."
        except Exception:
            return "Let your actions be guided by love, truth, and the welfare of all beings."
    
    async def _generate_spiritual_context(self, message: str, knowledge_base) -> str:
        """Generate spiritual context using knowledge base"""
        try:
            if hasattr(knowledge_base, 'get_spiritual_context'):
                return await knowledge_base.get_spiritual_context(message)
            else:
                return "Every moment is an opportunity for spiritual growth and self-realization."
        except Exception:
            return "You are a spiritual being having a human experience. Embrace both the journey and the destination."
    
    async def _extract_dharmic_insights(self, dharmic_knowledge: Dict[str, str]) -> List[str]:
        """Extract key dharmic insights"""
        insights = []
        
        for key, value in dharmic_knowledge.items():
            # Extract the essence of each wisdom
            if "dharma" in value.lower():
                insights.append("🔸 Follow your dharma - your righteous path in life")
            if "karma" in value.lower():
                insights.append("🔸 Focus on right action; let go of attachment to results")
            if "moksha" in value.lower():
                insights.append("🔸 Ultimate freedom comes through self-realization")
            if "compassion" in value.lower():
                insights.append("🔸 Compassion is the highest virtue")
        
        if not insights:
            insights = [
                "🔸 Live with awareness and intention",
                "🔸 Practice love and kindness in all interactions",
                "🔸 Seek truth and wisdom in every experience"
            ]
        
        return insights[:4]  # Return top 4 insights

# Create global instance
dharmic_ai = DharmicConversationalAI()

@router.post("/chat", response_model=DharmicChatResponse)
async def dharmic_chat(request: DharmicChatRequest):
    """
    Main endpoint for dharmic conversational AI
    
    This provides ChatGPT-style responses with deep dharmic wisdom integration:
    - Natural conversational responses
    - Hindu dharmic knowledge and wisdom
    - Personal growth guidance
    - Spiritual insights
    - Ethical living principles
    - Universal language for all humanity
    """
    return await dharmic_ai.generate_dharmic_response(request)

@router.get("/wisdom-topics")
async def get_wisdom_topics():
    """Get available dharmic wisdom topics"""
    return {
        "life_guidance": [
            "Life purpose and meaning",
            "Dealing with challenges and suffering", 
            "Making ethical decisions",
            "Finding inner peace",
            "Building meaningful relationships"
        ],
        "spiritual_growth": [
            "Meditation and mindfulness",
            "Self-inquiry and self-realization",
            "Understanding karma and dharma",
            "Path to enlightenment",
            "Sacred texts and teachings"
        ],
        "practical_dharma": [
            "Ethical living in modern world",
            "Balancing spiritual and material life",
            "Family and relationship dharma",
            "Work and career guidance",
            "Health and well-being"
        ]
    }

@router.post("/quick-wisdom")
async def get_quick_wisdom(question: str):
    """Get quick dharmic wisdom for any question"""
    try:
        # Simple dharmic response for quick questions
        dharmic_responses = {
            "purpose": "Your purpose is to realize your true Self while serving others with love and compassion.",
            "suffering": "Suffering teaches us compassion and detachment. Embrace it as a teacher, not an enemy.",
            "success": "True success is inner peace and the ability to help others on their journey.",
            "love": "Love is the recognition of the divine in all beings. Practice it unconditionally.",
            "fear": "Fear dissolves when you remember your eternal nature. You are consciousness itself.",
            "anger": "Anger clouds wisdom. Breathe deeply and respond from love, not reaction.",
            "happiness": "Happiness comes from within, not from external circumstances. Cultivate contentment.",
            "truth": "Truth is simple: you are divine consciousness expressing itself in human form."
        }
        
        question_lower = question.lower()
        for key, response in dharmic_responses.items():
            if key in question_lower:
                return {
                    "wisdom": response,
                    "source": "dharmic_guidance",
                    "practice": f"Meditate on this truth and apply it in your daily life."
                }
        
        return {
            "wisdom": "Every question is a sacred inquiry. Look within for the answer, and trust your inner wisdom.",
            "source": "universal_dharma",
            "practice": "Practice self-inquiry and mindful reflection."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
