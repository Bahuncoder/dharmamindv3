"""
<<<<<<< HEAD
Dharmic LLM Processor Service
============================

Processes LLM responses through dharmic enhancement.
Temporary implementation for backward compatibility.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

=======
🕉️ Dharmic LLM Response Processor

This service processes external LLM responses (ChatGPT, Claude, etc.) through 
our complete Dharmic backend system to ensure all responses are:

1. Aligned with dharmic principles
2. Enhanced with spiritual wisdom
3. Filtered through appropriate spiritual modules
4. Validated against subscription limits
5. Enriched with scriptural references

The system ensures that any external AI response is transformed into 
dharmic guidance that serves the user's spiritual growth.

Features:
- External LLM response processing
- Dharmic alignment and enhancement
- Spiritual module integration
- Subscription-aware processing
- Knowledge base integration
- Response validation and filtering

May this service transform all AI responses into dharmic wisdom 🕉️
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..models.chat import ChatMessage, ModuleInfo
from ..models.subscription import SubscriptionTier
from ..services.llm_router import LLMRouter, LLMResponse, LLMProvider
from ..chakra_modules.darshana_engine import get_darshana_engine
from ..chakra_modules.system_orchestrator import SystemOrchestrator
from ..spiritual_modules import get_spiritual_router
from ..services.subscription_service import SubscriptionService
from ..config import settings

>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
logger = logging.getLogger(__name__)

class DharmicProcessingMode(str, Enum):
    """Dharmic processing modes"""
<<<<<<< HEAD
    BASIC = "basic"
    ENHANCED = "enhanced"
    DEEP = "deep"
    RISHI = "rishi"

class DharmicResponse:
    """Dharmic response container"""
    
    def __init__(self, response: str, insights: list, confidence: float = 0.9):
        self.response = response
        self.insights = insights
        self.confidence = confidence
        self.timestamp = datetime.utcnow().isoformat()

class DharmicLLMProcessor:
    """Processes LLM responses through dharmic enhancement"""
    
    def __init__(self):
        self.initialized = True
        logger.info("Dharmic LLM Processor initialized")
    
    async def process_response(
        self,
        response: str,
        mode: DharmicProcessingMode = DharmicProcessingMode.ENHANCED,
        context: Optional[Dict[str, Any]] = None
    ) -> DharmicResponse:
        """Process response through dharmic enhancement"""
        try:
            # Basic dharmic enhancement
            enhanced_response = f"🕉️ {response}\n\nMay this guidance serve your spiritual journey with wisdom and compassion."
            
            insights = [
                "Response enhanced with dharmic principles",
                "Spiritual context considered",
                "Compassionate guidance provided"
            ]
            
            return DharmicResponse(enhanced_response, insights, 0.9)
            
        except Exception as e:
            logger.error(f"Error in dharmic processing: {e}")
            return DharmicResponse(
                "I apologize for any difficulty. May you find peace and wisdom on your path.",
                ["Error in processing, basic compassionate response provided"],
                0.5
            )

# Global processor instance
_dharmic_llm_processor = None

def get_dharmic_llm_processor() -> DharmicLLMProcessor:
    """Get or create dharmic LLM processor instance"""
    global _dharmic_llm_processor
    
    if _dharmic_llm_processor is None:
        _dharmic_llm_processor = DharmicLLMProcessor()
    
    return _dharmic_llm_processor
=======
    LIGHT = "light"          # Basic dharmic alignment
    STANDARD = "standard"    # Full spiritual module processing  
    DEEP = "deep"           # Complete dharmic transformation
    PREMIUM = "premium"      # Advanced features with all modules

@dataclass
class DharmicResponse:
    """Enhanced dharmic response"""
    original_response: str
    dharmic_response: str
    spiritual_insights: List[str]
    scriptural_references: List[Dict[str, str]]
    dharmic_alignment_score: float
    processing_mode: DharmicProcessingMode
    modules_used: List[str]
    subscription_tier: SubscriptionTier
    metadata: Dict[str, Any]

class DharmicLLMProcessor:
    """Main service for processing external LLM responses through dharmic system"""
    
    def __init__(self):
        self.llm_router = None
        self.darshana_engine = None
        self.system_orchestrator = None
        self.spiritual_router = None
        self.subscription_service = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize all dharmic processing components"""
        try:
            logger.info("🕉️ Initializing Dharmic LLM Processor...")
            
            # Initialize core components
            self.llm_router = LLMRouter()
            await self.llm_router.initialize()
            
            self.darshana_engine = get_darshana_engine()
            self.system_orchestrator = SystemOrchestrator()
            self.spiritual_router = get_spiritual_router()
            self.subscription_service = SubscriptionService()
            
            self.is_initialized = True
            logger.info("✅ Dharmic LLM Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Dharmic LLM Processor: {e}")
            raise
    
    async def process_external_llm_response(
        self,
        external_response: str,
        original_query: str,
        user_id: str,
        subscription_tier: SubscriptionTier = SubscriptionTier.FREE,
        processing_mode: Optional[DharmicProcessingMode] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> DharmicResponse:
        """
        Process external LLM response through complete dharmic system
        
        Flow:
        1. Validate subscription and determine processing mode
        2. Analyze response for dharmic content and spiritual themes
        3. Route through appropriate spiritual modules
        4. Enhance with darshana engine philosophical framework
        5. Add scriptural references and spiritual insights
        6. Generate final dharmic response
        """
        
        if not self.is_initialized:
            await self.initialize()
            
        # Step 1: Determine processing mode based on subscription
        actual_mode = await self._determine_processing_mode(subscription_tier, processing_mode)
        
        # Step 2: Check subscription limits
        await self._validate_subscription_usage(user_id, subscription_tier, actual_mode)
        
        # Step 3: Analyze spiritual content in the response
        spiritual_analysis = await self._analyze_spiritual_content(
            external_response, original_query, user_context
        )
        
        # Step 4: Route through spiritual modules based on analysis
        spiritual_enhancement = await self._process_through_spiritual_modules(
            external_response, original_query, spiritual_analysis, actual_mode
        )
        
        # Step 5: Apply darshana engine philosophical framework
        philosophical_enhancement = await self._apply_darshana_framework(
            spiritual_enhancement, spiritual_analysis, actual_mode
        )
        
        # Step 6: Generate final dharmic response
        final_response = await self._generate_dharmic_response(
            original_response=external_response,
            spiritual_enhancement=spiritual_enhancement,
            philosophical_enhancement=philosophical_enhancement,
            spiritual_analysis=spiritual_analysis,
            processing_mode=actual_mode,
            subscription_tier=subscription_tier
        )
        
        # Step 7: Log usage for subscription tracking
        await self._log_usage(user_id, actual_mode, len(final_response.dharmic_response))
        
        return final_response
    
    async def _determine_processing_mode(
        self,
        subscription_tier: SubscriptionTier,
        requested_mode: Optional[DharmicProcessingMode]
    ) -> DharmicProcessingMode:
        """Determine actual processing mode based on subscription"""
        
        # Map subscription tiers to maximum processing modes
        tier_to_max_mode = {
            SubscriptionTier.FREE: DharmicProcessingMode.LIGHT,
            SubscriptionTier.PRO: DharmicProcessingMode.STANDARD,
            SubscriptionTier.MAX: DharmicProcessingMode.DEEP,
            SubscriptionTier.ENTERPRISE: DharmicProcessingMode.PREMIUM
        }
        
        max_allowed = tier_to_max_mode.get(subscription_tier, DharmicProcessingMode.LIGHT)
        
        # If no mode requested, use maximum allowed
        if not requested_mode:
            return max_allowed
            
        # Return the minimum of requested and allowed
        mode_hierarchy = [
            DharmicProcessingMode.LIGHT,
            DharmicProcessingMode.STANDARD, 
            DharmicProcessingMode.DEEP,
            DharmicProcessingMode.PREMIUM
        ]
        
        requested_level = mode_hierarchy.index(requested_mode)
        max_level = mode_hierarchy.index(max_allowed)
        
        return mode_hierarchy[min(requested_level, max_level)]
    
    async def _validate_subscription_usage(
        self,
        user_id: str,
        subscription_tier: SubscriptionTier,
        processing_mode: DharmicProcessingMode
    ):
        """Validate subscription usage limits"""
        
        # Check monthly chat limits
        monthly_usage = await self.subscription_service.get_current_usage(
            user_id, "dharmic_processing"
        )
        
        # Get limits based on subscription tier
        limits = {
            SubscriptionTier.FREE: 50,
            SubscriptionTier.PRO: -1,  # Unlimited
            SubscriptionTier.MAX: -1,  # Unlimited  
            SubscriptionTier.ENTERPRISE: -1  # Unlimited
        }
        
        limit = limits.get(subscription_tier, 50)
        
        if limit > 0 and monthly_usage >= limit:
            raise Exception(f"Monthly dharmic processing limit ({limit}) exceeded")
    
    async def _analyze_spiritual_content(
        self,
        response: str,
        query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze spiritual content in response and query"""
        
        # Spiritual keywords and themes
        spiritual_themes = {
            'dharma': ['dharma', 'duty', 'righteousness', 'purpose'],
            'karma': ['karma', 'action', 'consequences', 'cause'],
            'yoga': ['yoga', 'meditation', 'practice', 'union'],
            'jnana': ['knowledge', 'wisdom', 'understanding', 'awareness'],
            'bhakti': ['devotion', 'love', 'surrender', 'faith'],
            'moksha': ['liberation', 'freedom', 'enlightenment', 'release'],
            'seva': ['service', 'helping', 'selfless', 'giving'],
            'ahimsa': ['non-violence', 'peace', 'compassion', 'kindness']
        }
        
        # Analyze both query and response
        combined_text = f"{query} {response}".lower()
        
        detected_themes = {}
        for theme, keywords in spiritual_themes.items():
            count = sum(1 for keyword in keywords if keyword in combined_text)
            if count > 0:
                detected_themes[theme] = count
        
        # Determine primary spiritual path
        primary_path = max(detected_themes.items(), key=lambda x: x[1])[0] if detected_themes else 'general'
        
        # Check for life stage indicators
        life_stages = {
            'brahmacharya': ['student', 'learning', 'study', 'young'],
            'grihastha': ['family', 'work', 'responsibility', 'marriage'],
            'vanaprastha': ['midlife', 'guidance', 'mentoring', 'wisdom'],
            'sannyasa': ['retirement', 'spiritual', 'detachment', 'elderly']
        }
        
        detected_stage = None
        for stage, indicators in life_stages.items():
            if any(indicator in combined_text for indicator in indicators):
                detected_stage = stage
                break
        
        return {
            'spiritual_themes': detected_themes,
            'primary_path': primary_path,
            'life_stage': detected_stage,
            'dharmic_content_score': len(detected_themes) / len(spiritual_themes),
            'needs_enhancement': len(detected_themes) < 2  # Low spiritual content
        }
    
    async def _process_through_spiritual_modules(
        self,
        response: str,
        query: str,
        spiritual_analysis: Dict[str, Any],
        processing_mode: DharmicProcessingMode
    ) -> Dict[str, Any]:
        """Process through appropriate spiritual modules"""
        
        enhancement = {
            'spiritual_insights': [],
            'practical_guidance': [],
            'modules_used': [],
            'dharmic_principles': []
        }
        
        # Determine which modules to use based on processing mode
        if processing_mode == DharmicProcessingMode.LIGHT:
            # Basic dharmic alignment only
            primary_path = spiritual_analysis['primary_path']
            if primary_path != 'general':
                enhancement['modules_used'].append(f"{primary_path}_module")
                enhancement['spiritual_insights'].append(
                    f"From the path of {primary_path.title()}: {self._get_basic_insight(primary_path)}"
                )
        
        elif processing_mode in [DharmicProcessingMode.STANDARD, DharmicProcessingMode.DEEP, DharmicProcessingMode.PREMIUM]:
            # Use spiritual router for comprehensive processing
            spiritual_response = await self.spiritual_router.analyze_and_guide(
                query, response, spiritual_analysis
            )
            
            enhancement['spiritual_insights'] = spiritual_response.get('insights', [])
            enhancement['practical_guidance'] = spiritual_response.get('guidance', [])
            enhancement['modules_used'] = spiritual_response.get('modules_used', [])
            enhancement['dharmic_principles'] = spiritual_response.get('principles', [])
        
        return enhancement
    
    async def _apply_darshana_framework(
        self,
        spiritual_enhancement: Dict[str, Any],
        spiritual_analysis: Dict[str, Any],
        processing_mode: DharmicProcessingMode
    ) -> Dict[str, Any]:
        """Apply darshana engine philosophical framework"""
        
        if processing_mode == DharmicProcessingMode.LIGHT:
            # Skip philosophical processing for light mode
            return {'philosophical_framework': 'Basic dharmic principles applied'}
        
        # Use darshana engine for philosophical analysis
        philosophical_analysis = await self.darshana_engine.analyze_query(
            spiritual_analysis.get('primary_path', 'general'),
            spiritual_enhancement
        )
        
        return {
            'philosophical_framework': philosophical_analysis.get('framework', ''),
            'scriptural_context': philosophical_analysis.get('scriptures', []),
            'philosophical_principles': philosophical_analysis.get('principles', [])
        }
    
    async def _generate_dharmic_response(
        self,
        original_response: str,
        spiritual_enhancement: Dict[str, Any],
        philosophical_enhancement: Dict[str, Any],
        spiritual_analysis: Dict[str, Any],
        processing_mode: DharmicProcessingMode,
        subscription_tier: SubscriptionTier
    ) -> DharmicResponse:
        """Generate final dharmic response"""
        
        # Build enhanced response based on processing mode
        if processing_mode == DharmicProcessingMode.LIGHT:
            dharmic_response = await self._build_light_response(
                original_response, spiritual_enhancement
            )
        else:
            dharmic_response = await self._build_enhanced_response(
                original_response, spiritual_enhancement, philosophical_enhancement, processing_mode
            )
        
        # Calculate dharmic alignment score
        alignment_score = self._calculate_alignment_score(
            spiritual_analysis, spiritual_enhancement, processing_mode
        )
        
        # Compile scriptural references
        scriptural_refs = philosophical_enhancement.get('scriptural_context', [])
        
        return DharmicResponse(
            original_response=original_response,
            dharmic_response=dharmic_response,
            spiritual_insights=spiritual_enhancement.get('spiritual_insights', []),
            scriptural_references=scriptural_refs,
            dharmic_alignment_score=alignment_score,
            processing_mode=processing_mode,
            modules_used=spiritual_enhancement.get('modules_used', []),
            subscription_tier=subscription_tier,
            metadata={
                'processed_at': datetime.utcnow().isoformat(),
                'spiritual_analysis': spiritual_analysis,
                'processing_mode': processing_mode,
                'enhancement_applied': True
            }
        )
    
    async def _build_light_response(
        self,
        original_response: str,
        spiritual_enhancement: Dict[str, Any]
    ) -> str:
        """Build light dharmic response (Free tier)"""
        
        dharmic_addition = ""
        
        if spiritual_enhancement.get('spiritual_insights'):
            dharmic_addition += f"\n\n🕉️ **Dharmic Perspective:**\n"
            dharmic_addition += spiritual_enhancement['spiritual_insights'][0]
        
        return original_response + dharmic_addition
    
    async def _build_enhanced_response(
        self,
        original_response: str,
        spiritual_enhancement: Dict[str, Any],
        philosophical_enhancement: Dict[str, Any],
        processing_mode: DharmicProcessingMode
    ) -> str:
        """Build enhanced dharmic response (Pro+ tiers)"""
        
        enhanced_response = original_response
        
        # Add spiritual insights
        if spiritual_enhancement.get('spiritual_insights'):
            enhanced_response += f"\n\n🕉️ **Spiritual Wisdom:**\n"
            for insight in spiritual_enhancement['spiritual_insights'][:3]:
                enhanced_response += f"• {insight}\n"
        
        # Add practical guidance
        if spiritual_enhancement.get('practical_guidance'):
            enhanced_response += f"\n🌟 **Dharmic Guidance:**\n"
            for guidance in spiritual_enhancement['practical_guidance'][:3]:
                enhanced_response += f"• {guidance}\n"
        
        # Add philosophical framework (Deep+ modes)
        if processing_mode in [DharmicProcessingMode.DEEP, DharmicProcessingMode.PREMIUM]:
            if philosophical_enhancement.get('philosophical_framework'):
                enhanced_response += f"\n📚 **Philosophical Foundation:**\n"
                enhanced_response += philosophical_enhancement['philosophical_framework']
        
        # Add scriptural references (Premium mode)
        if processing_mode == DharmicProcessingMode.PREMIUM:
            if philosophical_enhancement.get('scriptural_context'):
                enhanced_response += f"\n📖 **Sacred Texts:**\n"
                for ref in philosophical_enhancement['scriptural_context'][:2]:
                    enhanced_response += f"• {ref.get('text', '')}\n"
        
        return enhanced_response
    
    def _calculate_alignment_score(
        self,
        spiritual_analysis: Dict[str, Any],
        spiritual_enhancement: Dict[str, Any],
        processing_mode: DharmicProcessingMode
    ) -> float:
        """Calculate dharmic alignment score"""
        
        base_score = spiritual_analysis.get('dharmic_content_score', 0.0)
        
        # Boost based on enhancement
        enhancement_boost = len(spiritual_enhancement.get('spiritual_insights', [])) * 0.1
        
        # Boost based on processing mode
        mode_boosts = {
            DharmicProcessingMode.LIGHT: 0.1,
            DharmicProcessingMode.STANDARD: 0.2,
            DharmicProcessingMode.DEEP: 0.3,
            DharmicProcessingMode.PREMIUM: 0.4
        }
        
        mode_boost = mode_boosts.get(processing_mode, 0.1)
        
        return min(1.0, base_score + enhancement_boost + mode_boost)
    
    def _get_basic_insight(self, spiritual_path: str) -> str:
        """Get basic insight for spiritual path"""
        
        insights = {
            'dharma': "Consider how your actions align with your life purpose and duty to others.",
            'karma': "Remember that every action creates consequences - choose mindfully.",
            'yoga': "Integration of mind, body, and spirit leads to inner harmony.",
            'jnana': "True wisdom comes from understanding the nature of reality and self.",
            'bhakti': "Devotion and love transform the heart and open spiritual doors.",
            'moksha': "Ultimate freedom comes from releasing attachment to temporary things.",
            'seva': "Selfless service purifies the heart and connects us to the divine.",
            'ahimsa': "Non-violence in thought, word, and deed creates lasting peace."
        }
        
        return insights.get(spiritual_path, "Apply dharmic principles of righteousness and compassion.")
    
    async def _log_usage(
        self,
        user_id: str,
        processing_mode: DharmicProcessingMode,
        response_length: int
    ):
        """Log usage for subscription tracking"""
        
        # Log dharmic processing usage
        await self.subscription_service.log_feature_usage(
            user_id,
            "dharmic_processing",
            metadata={
                'processing_mode': processing_mode,
                'response_length': response_length,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    async def health_check(self) -> bool:
        """Health check for dharmic processor"""
        try:
            return (
                self.is_initialized and
                self.llm_router and
                self.darshana_engine and
                self.system_orchestrator and
                self.spiritual_router
            )
        except:
            return False

# Singleton instance
_dharmic_processor = None

async def get_dharmic_llm_processor() -> DharmicLLMProcessor:
    """Get singleton dharmic LLM processor"""
    global _dharmic_processor
    if _dharmic_processor is None:
        _dharmic_processor = DharmicLLMProcessor()
        await _dharmic_processor.initialize()
    return _dharmic_processor
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
