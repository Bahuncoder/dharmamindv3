"""
Deep Contemplation System for DharmaMind platform

Advanced contemplative practices and deep spiritual inquiry system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ContemplationType(str, Enum):
    """Types of contemplative practices"""
    SELF_INQUIRY = "self_inquiry"
    DHARMIC_INQUIRY = "dharmic_inquiry"
    NATURE_CONTEMPLATION = "nature_contemplation"
    DEATH_CONTEMPLATION = "death_contemplation"
    LOVING_KINDNESS = "loving_kindness"
    WISDOM_CONTEMPLATION = "wisdom_contemplation"
    IMPERMANENCE = "impermanence"
    INTERCONNECTEDNESS = "interconnectedness"

class ContemplationLevel(str, Enum):
    """Depth levels of contemplation"""
    SURFACE = "surface"
    INTERMEDIATE = "intermediate"
    DEEP = "deep"
    PROFOUND = "profound"
    TRANSCENDENT = "transcendent"

class ContemplationTradition(str, Enum):
    """Contemplation traditions"""
    ADVAITA = "advaita"
    BUDDHIST = "buddhist"
    CHRISTIAN = "christian"
    SUFI = "sufi"
    JAIN = "jain"
    SECULAR = "secular"

class ContemplationDepth(str, Enum):
    """Contemplation depth levels (alias for ContemplationLevel)"""
    SURFACE = "surface"
    INTERMEDIATE = "intermediate"
    DEEP = "deep"
    PROFOUND = "profound"
    TRANSCENDENT = "transcendent"

class ContemplationRequest(BaseModel):
    """Request for contemplation guidance"""
    user_id: str = Field(..., description="User identifier")
    contemplation_type: ContemplationType = Field(..., description="Type of contemplation")
    depth_level: ContemplationLevel = Field(default=ContemplationLevel.INTERMEDIATE, description="Desired depth")
    duration_minutes: Optional[int] = Field(default=20, description="Session duration")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ContemplationResponse(BaseModel):
    """Response with contemplation guidance"""
    session_id: str = Field(..., description="Unique session identifier")
    contemplation_type: ContemplationType = Field(..., description="Type of contemplation")
    depth_level: ContemplationLevel = Field(..., description="Depth level")
    
    # Guidance content
    opening_guidance: str = Field(..., description="Opening contemplation guidance")
    core_inquiry: str = Field(..., description="Core contemplative inquiry")
    reflection_prompts: List[str] = Field(default_factory=list, description="Reflection prompts")
    closing_guidance: str = Field(..., description="Closing guidance")
    
    # Practice structure
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Contemplation stages")
    key_insights: List[str] = Field(default_factory=list, description="Key insights to explore")
    
    # Integration
    integration_practices: List[str] = Field(default_factory=list, description="Integration practices")
    follow_up_inquiries: List[str] = Field(default_factory=list, description="Follow-up inquiries")
    
    # Metadata
    estimated_duration: int = Field(..., description="Estimated duration in minutes")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

class ContemplationInsight(BaseModel):
    """Insight from contemplation session"""
    insight_id: str = Field(..., description="Unique insight identifier")
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    
    # Insight content
    insight_text: str = Field(..., description="The insight itself")
    depth_level: ContemplationLevel = Field(..., description="Depth of insight")
    category: str = Field(..., description="Insight category")
    
    # Context
    arose_during: str = Field(..., description="When the insight arose")
    triggered_by: Optional[str] = Field(default=None, description="What triggered the insight")
    
    # Quality
    clarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Clarity of insight")
    transformative_potential: float = Field(default=0.0, ge=0.0, le=1.0, description="Transformative potential")
    
    # Timestamps
    recorded_at: datetime = Field(default_factory=datetime.now, description="Recording timestamp")

class DeepContemplationSystem:
    """System for deep contemplative practices"""
    
    def __init__(self):
        self.contemplation_templates = {
            ContemplationType.SELF_INQUIRY: {
                "opening": "Settle into a comfortable position and bring your attention inward. Let the mind become quiet and receptive.",
                "core_inquiry": "Who am I beyond my thoughts, emotions, and roles?",
                "prompts": [
                    "What remains constant as thoughts and feelings change?",
                    "Who is the observer of all experiences?",
                    "What is the source of awareness itself?"
                ],
                "closing": "Rest in the spaciousness of pure awareness, beyond all identifications."
            },
            ContemplationType.DHARMIC_INQUIRY: {
                "opening": "Connect with the dharmic principles that guide righteous living and spiritual growth.",
                "core_inquiry": "How can I align my actions with the highest dharmic principles?",
                "prompts": [
                    "What does ahimsa (non-violence) mean in my daily life?",
                    "How can I practice satya (truthfulness) more deeply?",
                    "Where can I release attachment and practice aparigraha?"
                ],
                "closing": "May my life be a living expression of dharmic wisdom and compassion."
            },
            ContemplationType.IMPERMANENCE: {
                "opening": "Bring to mind the constant flow and change in all phenomena.",
                "core_inquiry": "How does understanding impermanence liberate the heart?",
                "prompts": [
                    "What am I clinging to that is inherently temporary?",
                    "How does accepting change bring peace?",
                    "What remains untouched by the dance of impermanence?"
                ],
                "closing": "Rest in the peace that comes from accepting the flowing nature of existence."
            }
        }
        
        self.active_sessions: Dict[str, ContemplationResponse] = {}
        self.user_insights: Dict[str, List[ContemplationInsight]] = {}
    
    async def initialize(self) -> bool:
        """Initialize the deep contemplation system"""
        try:
            logger.info("Deep Contemplation System initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Deep Contemplation System: {e}")
            return False
    
    async def start_contemplation_session(
        self,
        request: ContemplationRequest
    ) -> ContemplationResponse:
        """Start a new contemplation session"""
        try:
            session_id = f"session_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get template for contemplation type
            template = self.contemplation_templates.get(
                request.contemplation_type,
                self.contemplation_templates[ContemplationType.SELF_INQUIRY]
            )
            
            # Create contemplation stages based on duration
            stages = await self._create_contemplation_stages(
                request.contemplation_type,
                request.depth_level,
                request.duration_minutes or 20
            )
            
            # Generate key insights for this type
            key_insights = await self._generate_key_insights(
                request.contemplation_type,
                request.depth_level
            )
            
            # Create response
            response = ContemplationResponse(
                session_id=session_id,
                contemplation_type=request.contemplation_type,
                depth_level=request.depth_level,
                opening_guidance=template["opening"],
                core_inquiry=template["core_inquiry"],
                reflection_prompts=template["prompts"],
                closing_guidance=template["closing"],
                stages=stages,
                key_insights=key_insights,
                integration_practices=await self._get_integration_practices(request.contemplation_type),
                follow_up_inquiries=await self._get_follow_up_inquiries(request.contemplation_type),
                estimated_duration=request.duration_minutes or 20
            )
            
            # Store active session
            self.active_sessions[session_id] = response
            
            logger.info(f"Started contemplation session {session_id} for user {request.user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error starting contemplation session: {e}")
            raise
    
    async def _create_contemplation_stages(
        self,
        contemplation_type: ContemplationType,
        depth_level: ContemplationLevel,
        duration_minutes: int
    ) -> List[Dict[str, Any]]:
        """Create structured contemplation stages"""
        stages = []
        
        # Basic structure based on duration
        if duration_minutes <= 10:
            stages = [
                {"name": "Settling", "duration": 2, "guidance": "Settle into stillness and presence"},
                {"name": "Inquiry", "duration": 6, "guidance": "Engage with the core contemplative question"},
                {"name": "Integration", "duration": 2, "guidance": "Integrate insights and close with gratitude"}
            ]
        elif duration_minutes <= 20:
            stages = [
                {"name": "Preparation", "duration": 3, "guidance": "Prepare mind and heart for contemplation"},
                {"name": "Opening", "duration": 2, "guidance": "Open to the contemplative process"},
                {"name": "Deep Inquiry", "duration": 12, "guidance": "Engage deeply with the questions"},
                {"name": "Integration", "duration": 3, "guidance": "Integrate and close mindfully"}
            ]
        else:
            stages = [
                {"name": "Centering", "duration": 5, "guidance": "Center yourself in stillness"},
                {"name": "Opening", "duration": 5, "guidance": "Open to the contemplative space"},
                {"name": "Preliminary Inquiry", "duration": 8, "guidance": "Begin exploring the questions"},
                {"name": "Deep Contemplation", "duration": duration_minutes - 25, "guidance": "Rest in deep contemplative awareness"},
                {"name": "Insight Gathering", "duration": 4, "guidance": "Gather insights that have arisen"},
                {"name": "Integration", "duration": 3, "guidance": "Integrate and close with dedication"}
            ]
        
        return stages
    
    async def _generate_key_insights(
        self,
        contemplation_type: ContemplationType,
        depth_level: ContemplationLevel
    ) -> List[str]:
        """Generate key insights for contemplation type"""
        insight_templates = {
            ContemplationType.SELF_INQUIRY: [
                "The observer of thoughts is not itself a thought",
                "Awareness remains constant while experiences change",
                "The sense of 'I' is more fundamental than any identity"
            ],
            ContemplationType.DHARMIC_INQUIRY: [
                "Right action flows from understanding interconnection",
                "Compassion is wisdom in action",
                "Dharma is both path and destination"
            ],
            ContemplationType.IMPERMANENCE: [
                "Clinging to the permanent in the impermanent causes suffering",
                "Change is the only constant; accepting this brings peace",
                "Within impermanence lies the discovery of what is eternal"
            ]
        }
        
        base_insights = insight_templates.get(contemplation_type, [])
        
        # Add depth-specific insights
        if depth_level in [ContemplationLevel.DEEP, ContemplationLevel.PROFOUND]:
            base_insights.extend([
                "Direct experience transcends conceptual understanding",
                "The question and questioner are not separate"
            ])
        
        return base_insights
    
    async def _get_integration_practices(self, contemplation_type: ContemplationType) -> List[str]:
        """Get integration practices for contemplation type"""
        integration_map = {
            ContemplationType.SELF_INQUIRY: [
                "Throughout the day, ask 'Who is aware of this?'",
                "Notice the space of awareness in daily activities",
                "Practice witnessing thoughts without identification"
            ],
            ContemplationType.DHARMIC_INQUIRY: [
                "Reflect on dharmic choices in daily decisions",
                "Practice one act of compassion each day",
                "Study dharmic texts for deeper understanding"
            ],
            ContemplationType.IMPERMANENCE: [
                "Notice the changing nature of all experiences",
                "Practice letting go of small attachments",
                "Contemplate the temporary nature of challenges"
            ]
        }
        
        return integration_map.get(contemplation_type, [
            "Carry the insights into daily life",
            "Journal about the contemplation experience",
            "Share insights with a spiritual friend"
        ])
    
    async def _get_follow_up_inquiries(self, contemplation_type: ContemplationType) -> List[str]:
        """Get follow-up inquiries for deeper exploration"""
        follow_up_map = {
            ContemplationType.SELF_INQUIRY: [
                "What is the relationship between consciousness and its contents?",
                "How does self-knowledge affect relationships with others?",
                "What is the source of the sense of existence itself?"
            ],
            ContemplationType.DHARMIC_INQUIRY: [
                "How does understanding karma influence present choices?",
                "What is the relationship between individual and universal dharma?",
                "How can dharmic living transform society?"
            ],
            ContemplationType.IMPERMANENCE: [
                "What dies and what is born in each moment?",
                "How does impermanence reveal the preciousness of life?",
                "What is the relationship between time and timelessness?"
            ]
        }
        
        return follow_up_map.get(contemplation_type, [])
    
    async def record_insight(
        self,
        session_id: str,
        user_id: str,
        insight_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ContemplationInsight:
        """Record an insight from contemplation"""
        try:
            insight_id = f"insight_{session_id}_{datetime.now().strftime('%H%M%S')}"
            
            insight = ContemplationInsight(
                insight_id=insight_id,
                session_id=session_id,
                user_id=user_id,
                insight_text=insight_text,
                depth_level=ContemplationLevel.INTERMEDIATE,  # Could be determined by analysis
                category="general",
                arose_during="contemplation_session",
                clarity_score=0.8,  # Could be determined by analysis
                transformative_potential=0.7  # Could be determined by analysis
            )
            
            # Store insight
            if user_id not in self.user_insights:
                self.user_insights[user_id] = []
            self.user_insights[user_id].append(insight)
            
            logger.info(f"Recorded insight {insight_id} for user {user_id}")
            return insight
            
        except Exception as e:
            logger.error(f"Error recording insight: {e}")
            raise
    
    async def get_user_insights(self, user_id: str) -> List[ContemplationInsight]:
        """Get all insights for a user"""
        return self.user_insights.get(user_id, [])
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        return {
            "status": "healthy",
            "system": "deep_contemplation",
            "active_sessions": len(self.active_sessions),
            "total_insights": sum(len(insights) for insights in self.user_insights.values()),
            "contemplation_types": len(self.contemplation_templates)
        }

# Global system instance
_deep_contemplation_system: Optional[DeepContemplationSystem] = None

async def get_deep_contemplation_system() -> DeepContemplationSystem:
    """Get the global deep contemplation system instance"""
    global _deep_contemplation_system
    
    if _deep_contemplation_system is None:
        _deep_contemplation_system = DeepContemplationSystem()
        await _deep_contemplation_system.initialize()
    
    return _deep_contemplation_system

# Convenience functions for route compatibility
async def begin_contemplation_session(
    user_id: str,
    contemplation_type: ContemplationType,
    depth_level: ContemplationLevel = ContemplationLevel.INTERMEDIATE,
    duration_minutes: int = 20,
    context: Optional[Dict[str, Any]] = None
) -> ContemplationResponse:
    """Begin a new contemplation session"""
    system = await get_deep_contemplation_system()
    request = ContemplationRequest(
        user_id=user_id,
        contemplation_type=contemplation_type,
        depth_level=depth_level,
        duration_minutes=duration_minutes,
        context=context
    )
    return await system.start_contemplation_session(request)

async def guide_contemplation_deepening(
    session_id: str,
    current_insights: List[str],
    request_deeper_guidance: bool = True
) -> Dict[str, Any]:
    """Guide contemplation to deeper levels"""
    system = await get_deep_contemplation_system()
    
    # Get session from active sessions
    session = system.active_sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    # Generate deeper guidance
    deeper_prompts = [
        "What lies beneath the surface of this insight?",
        "How does this understanding change your relationship to experience?",
        "What questions arise from this realization?",
        "How might this wisdom transform daily life?"
    ]
    
    guidance = {
        "session_id": session_id,
        "deeper_prompts": deeper_prompts,
        "integration_suggestions": [
            "Sit with the insight without trying to understand it",
            "Notice how this awareness shifts your perspective",
            "Allow the wisdom to permeate your being"
        ],
        "next_level_inquiries": [
            "Who or what is aware of this insight?",
            "What is the source of this understanding?",
            "How does this connect to the larger web of existence?"
        ]
    }
    
    return guidance

async def complete_contemplation_session(
    session_id: str,
    insights: Optional[List[str]] = None,
    integration_notes: Optional[str] = None
) -> Dict[str, Any]:
    """Complete a contemplation session"""
    system = await get_deep_contemplation_system()
    
    # Get session from active sessions
    session = system.active_sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    # Record insights if provided
    recorded_insights = []
    if insights:
        for insight_text in insights:
            try:
                insight = await system.record_insight(
                    session_id=session_id,
                    user_id=session.session_id.split('_')[1],  # Extract user_id from session_id
                    insight_text=insight_text
                )
                recorded_insights.append(insight.insight_id)
            except Exception as e:
                logger.error(f"Error recording insight: {e}")
    
    # Mark session as completed
    completion_summary = {
        "session_id": session_id,
        "completed_at": datetime.now().isoformat(),
        "duration_minutes": session.estimated_duration,
        "contemplation_type": session.contemplation_type.value,
        "insights_recorded": len(recorded_insights),
        "integration_notes": integration_notes,
        "follow_up_recommendations": session.follow_up_inquiries,
        "suggested_next_session": {
            "recommended_after_days": 3,
            "suggested_type": session.contemplation_type.value,
            "suggested_depth": session.depth_level.value
        }
    }
    
    # Remove from active sessions
    if session_id in system.active_sessions:
        del system.active_sessions[session_id]
    
    return completion_summary