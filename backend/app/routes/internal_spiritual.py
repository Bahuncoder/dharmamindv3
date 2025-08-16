"""
Internal Spiritual Wisdom Processing - DharmaMind Backend
===================================================            # Step 3: Emotional analysis and response selection
            emotional_context = await self.emotional_intelligence.process_emotional_content(
                query.message
            )==

This module provides internal spiritual wisdom responses using only the built-in
Chakra modules without relying on external LLM providers. It demonstrates the
power of our internal spiritual processing capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio

# Import Chakra modules
from backend.app.chakra_modules.consciousness_core import get_consciousness_core, ConsciousnessLevel
from backend.app.chakra_modules.knowledge_base import get_knowledge_base
from backend.app.chakra_modules.dharma_engine import get_dharma_engine
from backend.app.chakra_modules.emotional_intelligence import get_emotional_intelligence
from backend.app.chakra_modules.ai_core import get_ai_core

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/internal", tags=["Internal Spiritual Wisdom"])

class SpiritualQuery(BaseModel):
    message: str
    user_id: Optional[str] = "guest"
    context: Optional[Dict[str, Any]] = None
    tradition_preference: Optional[str] = None

class SpiritualResponse(BaseModel):
    response: str
    wisdom_source: str
    dharma_assessment: Dict[str, Any]
    consciousness_level: str
    emotional_tone: str
    confidence: float
    timestamp: datetime

class InternalSpiritualProcessor:
    """Internal spiritual wisdom processor using only Chakra modules"""
    
    def __init__(self):
        self.consciousness_core = get_consciousness_core()
        self.knowledge_base = get_knowledge_base()
        self.dharma_engine = get_dharma_engine()
        self.emotional_intelligence = get_emotional_intelligence()
        self.ai_core = get_ai_core()
        
        # Initialize emotional intelligence engine
        asyncio.create_task(self._initialize_modules())
        
        self.spiritual_templates = {
            "stress": [
                "🧘 In times of stress, remember that this too shall pass. Like clouds in the sky, difficulties are temporary visitors to your consciousness.",
                "💫 Stress arises when we resist the present moment. Practice accepting what is, while working skillfully toward what can be changed.",
                "🌸 Take three deep breaths and connect with your inner stillness. Within you lies an ocean of peace that stress cannot disturb.",
                "🕉️ As the Bhagavad Gita teaches: 'You are not your emotions, you are the observer of them.' Rest in this deeper truth."
            ],
            "anxiety": [
                "🌅 Anxiety often comes from projecting into an uncertain future. Gently return your awareness to this present moment, where peace dwells.",
                "🏔️ Like a mountain unmoved by changing weather, cultivate the stability of your deeper Self that anxiety cannot shake.",
                "🦋 Remember that transformation often comes through challenge. Your anxiety may be showing you what needs attention and care.",
                "☀️ Practice loving-kindness toward yourself. Anxiety is not your enemy, but a signal calling for compassion and understanding."
            ],
            "sadness": [
                "🌧️ Sadness is like rain - it nourishes the ground of your heart for new growth. Allow yourself to feel deeply and with compassion.",
                "🌱 In Buddhist wisdom, sadness reminds us of our interconnectedness. What you feel, the universe feels through you.",
                "🌙 Honor your sadness as a sacred visitor. It often carries gifts of wisdom and deeper empathy for all beings.",
                "💝 Your capacity to feel sadness is also your capacity for great love. They are two sides of the same open heart."
            ],
            "purpose": [
                "🌟 Your dharma (life purpose) reveals itself through what brings you joy, serves others, and aligns with your deepest values.",
                "🎯 As Rumi said: 'Let yourself be silently drawn by the strange pull of what you really love. It will not lead you astray.'",
                "🌍 Purpose often begins with serving something greater than yourself. How can your unique gifts contribute to the world's healing?",
                "🔥 Your purpose is not a destination but a way of being. Live with awareness, compassion, and authenticity in each moment."
            ],
            "relationships": [
                "❤️ All relationships are mirrors reflecting aspects of ourselves. Practice seeing others with the eyes of compassion.",
                "🤝 Healthy relationships honor both unity and individuality. Love without losing yourself; connect without possessing.",
                "🌸 Forgiveness is not about condoning harmful actions, but about freeing yourself from the burden of resentment.",
                "🕊️ True intimacy comes from being authentically yourself while allowing others the same freedom."
            ],
            "meditation": [
                "🧘‍♀️ Begin with three deep breaths. Inhale peace, exhale tension. Feel your body naturally relaxing with each breath.",
                "🌅 Find a comfortable position, close your eyes gently, and simply observe your breath. No need to change it, just witness.",
                "💫 In meditation, thoughts will come and go like clouds. Acknowledge them with kindness and return to your breath.",
                "🕉️ Start with just 5 minutes daily. Consistency is more valuable than duration. Let meditation become a sacred daily ritual.",
                "🌊 Meditation is not about emptying the mind, but about finding the stillness that already exists within you."
            ],
            "general": [
                "🕉️ Remember that you are a spiritual being having a human experience. Your essence is pure consciousness, beyond all temporary conditions.",
                "🌟 Whatever challenges you face, they are opportunities for growth and awakening. Trust in the wisdom of your journey.",
                "💎 Within your heart lies infinite wisdom, compassion, and strength. Connect with this inner sanctuary whenever you need guidance.",
                "🙏 Practice gratitude for this precious human life. Each moment is a gift to experience love, learn wisdom, and serve others."
            ]
        }
        
    async def _initialize_modules(self):
        """Initialize all Chakra modules that have initialize methods"""
        try:
            # Only initialize modules that have the initialize method
            if hasattr(self.emotional_intelligence, 'initialize'):
                await self.emotional_intelligence.initialize()
            if hasattr(self.consciousness_core, 'initialize'):
                await self.consciousness_core.initialize()
            if hasattr(self.knowledge_base, 'initialize'):
                await self.knowledge_base.initialize()
            if hasattr(self.ai_core, 'initialize'):
                await self.ai_core.initialize()
            logger.info("Chakra modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing modules: {str(e)}")
        
    async def process_spiritual_query(self, query: SpiritualQuery) -> SpiritualResponse:
        """Process spiritual query using internal Chakra modules"""
        try:
            # Step 1: Consciousness processing
            consciousness_event = await self.consciousness_core.process_input(
                query.message, 
                source="spiritual_inquiry"
            )
            
            # Step 2: Knowledge base search
            relevant_concepts = await self.knowledge_base.search_concepts(
                query.message, 
                tradition=query.tradition_preference,
                limit=5
            )
            
            # Step 3: Dharma assessment
            dharma_assessment = await self.dharma_engine.assess_dharma_compliance(
                query.message,
                context=query.context
            )
            
            # Step 4: Emotional analysis
            emotional_analysis = await self.emotional_intelligence.process_emotional_content(
                query.message
            )
            
            # Step 5: Generate spiritual response
            response_text = await self._generate_spiritual_response(
                query, consciousness_event, relevant_concepts, 
                dharma_assessment, emotional_analysis
            )
            
            # Step 6: Build response
            spiritual_response = SpiritualResponse(
                response=response_text,
                wisdom_source="DharmaMind Internal Chakra Modules",
                dharma_assessment={
                    "dharma_level": dharma_assessment.overall_level.value,
                    "dharma_score": dharma_assessment.dharma_score,
                    "positive_aspects": dharma_assessment.positive_aspects[:3]
                },
                consciousness_level=consciousness_event.level.name,
                emotional_tone=emotional_analysis.response_tone,
                confidence=0.85,  # High confidence in internal processing
                timestamp=datetime.now()
            )
            
            logger.info(f"Generated internal spiritual response for query: {query.message[:50]}...")
            return spiritual_response
            
        except Exception as e:
            logger.error(f"Error in spiritual processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Spiritual processing error: {str(e)}")
    
    async def _generate_spiritual_response(
        self, 
        query: SpiritualQuery,
        consciousness_event,
        relevant_concepts: List[Dict],
        dharma_assessment,
        emotional_analysis
    ) -> str:
        """Generate spiritual response using template-based wisdom"""
        
        message_lower = query.message.lower()
        
        # Detect query category with enhanced keyword detection
        category = "general"
        if any(word in message_lower for word in ["stress", "stressed", "overwhelming", "pressure", "overwhelmed", "burden", "stressful"]):
            category = "stress"
        elif any(word in message_lower for word in ["anxiety", "anxious", "worry", "worried", "fear", "fearful", "nervous", "panic"]):
            category = "anxiety"
        elif any(word in message_lower for word in ["sad", "sadness", "depressed", "grief", "loss", "sorrow", "mourning", "heartbroken"]):
            category = "sadness"
        elif any(word in message_lower for word in ["purpose", "meaning", "direction", "calling", "dharma", "mission", "path", "destiny", "powerful", "power", "strength", "strong"]):
            category = "purpose"
        elif any(word in message_lower for word in ["relationship", "love", "partner", "family", "friend", "marriage", "dating", "romance"]):
            category = "relationships"
        elif any(word in message_lower for word in ["meditation", "meditate", "breathing", "breath", "mindfulness", "peace", "calm", "serenity", "tranquil"]):
            category = "meditation"
        
        # Debug logging to see what category was selected
        logger.info(f"Selected category '{category}' for message: '{query.message[:50]}...'")
        
        # Get base response from templates
        import random
        base_response = random.choice(self.spiritual_templates[category])
        
        # Enhance with knowledge base insights
        if relevant_concepts:
            concept_wisdom = relevant_concepts[0].get('description', '')
            if concept_wisdom:
                base_response += f"\n\n📚 Ancient wisdom reminds us: {concept_wisdom}"
        
        # Add consciousness insights
        if consciousness_event.insights:
            insight = consciousness_event.insights[0]
            base_response += f"\n\n🧠 From deeper awareness: {insight}"
        
        # Add dharmic guidance if needed
        if dharma_assessment.recommendations:
            recommendation = dharma_assessment.recommendations[0]
            base_response += f"\n\n⚖️ Dharmic guidance: {recommendation}"
        
        # Add closing blessing
        base_response += "\n\n🙏 May this wisdom serve your highest good and the welfare of all beings."
        
        return base_response

# Create global processor instance
spiritual_processor = InternalSpiritualProcessor()

@router.post("/spiritual-wisdom", response_model=SpiritualResponse)
async def get_internal_spiritual_wisdom(query: SpiritualQuery):
    """
    Get spiritual wisdom using only internal DharmaMind Chakra modules
    
    This endpoint demonstrates our internal spiritual processing capabilities
    without relying on external LLM providers.
    """
    return await spiritual_processor.process_spiritual_query(query)

@router.get("/spiritual-stats")
async def get_spiritual_stats():
    """Get statistics about internal spiritual processing"""
    return {
        "internal_processing": True,
        "chakra_modules_active": 5,
        "wisdom_traditions": ["Hindu", "Buddhist", "Universal"],
        "response_categories": list(spiritual_processor.spiritual_templates.keys()),
        "processing_method": "Template-based with Chakra module enhancement",
        "confidence_level": "High (0.85)",
        "last_updated": datetime.now().isoformat()
    }

@router.post("/test-chakras")
async def test_chakra_modules():
    """Test all Chakra modules to verify they're working"""
    results = {}
    
    try:
        # Test consciousness core
        consciousness_core = get_consciousness_core()
        consciousness_event = await consciousness_core.process_input("test", "api_test")
        results["consciousness_core"] = {
            "status": "active",
            "level": consciousness_event.level.name,
            "insights_count": len(consciousness_event.insights)
        }
    except Exception as e:
        results["consciousness_core"] = {"status": "error", "error": str(e)}
    
    try:
        # Test knowledge base
        knowledge_base = get_knowledge_base()
        concepts = await knowledge_base.search_concepts("wisdom", limit=1)
        results["knowledge_base"] = {
            "status": "active",
            "concepts_found": len(concepts)
        }
    except Exception as e:
        results["knowledge_base"] = {"status": "error", "error": str(e)}
    
    try:
        # Test dharma engine
        dharma_engine = get_dharma_engine()
        assessment = await dharma_engine.assess_dharma_compliance("peaceful wisdom")
        results["dharma_engine"] = {
            "status": "active",
            "dharma_level": assessment.overall_level.name,
            "dharma_score": assessment.dharma_score
        }
    except Exception as e:
        results["dharma_engine"] = {"status": "error", "error": str(e)}
    
    try:
        # Test emotional intelligence
        emotional_intelligence = get_emotional_intelligence()
        emotion_analysis = await emotional_intelligence.analyze_emotional_context("I am happy")
        results["emotional_intelligence"] = {
            "status": "active",
            "analysis_keys": list(emotion_analysis.keys()) if emotion_analysis else []
        }
    except Exception as e:
        results["emotional_intelligence"] = {"status": "error", "error": str(e)}
    
    return {
        "test_timestamp": datetime.now().isoformat(),
        "modules_tested": len(results),
        "results": results
    }
