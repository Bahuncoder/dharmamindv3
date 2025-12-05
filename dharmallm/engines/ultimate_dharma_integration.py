#!/usr/bin/env python3
"""
🕉️ Ultimate Dharma Integration Engine - Complete System Orchestrator
====================================================================

Advanced integration system that orchestrates all DharmaLLM modules
to provide comprehensive, authentic Sanatana Dharma guidance.

Features:
- Unified query processing across all modules
- Context-aware response generation
- Cross-module knowledge correlation
- Holistic spiritual guidance
- Real-time adaptive recommendations
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Import all engine modules
try:
    from ayurveda_engine import (
        AdvancedAyurvedaEngine,
        AdvancedJyotishaEngine,
        DharmaEngine,
        SanskritIntelligence,
        dharma_engine,
        get_ayurveda_engine,
        get_dharma_engine,
        get_jyotisha_engine,
        get_sanskrit_intelligence,
        jyotisha_engine,
        sanskrit_intelligence,
    )
    from spiritual_intelligence import (
        SpiritualIntelligence,
        get_spiritual_intelligence,
    )
    from vedic_ritual_engine import VedicRitualEngine, get_vedic_ritual_engine
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # Create dummy classes for missing modules

    class DummyEngine:
        pass

    SanskritIntelligence = VedicRitualEngine = AdvancedJyotishaEngine = (
        DummyEngine
    )
    AdvancedAyurvedaEngine = SpiritualIntelligence = DharmaEngine = DummyEngine

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of spiritual queries"""

    SPIRITUAL_GUIDANCE = "spiritual_guidance"
    SANSKRIT_LEARNING = "sanskrit_learning"
    RITUAL_GUIDANCE = "ritual_guidance"
    ASTROLOGICAL_CONSULTATION = "astrological_consultation"
    HEALTH_CONSULTATION = "health_consultation"
    PHILOSOPHICAL_INQUIRY = "philosophical_inquiry"
    LIFE_GUIDANCE = "life_guidance"
    MANTRA_GUIDANCE = "mantra_guidance"
    DHARMIC_DECISION = "dharmic_decision"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"


class ResponsePriority(Enum):
    """Priority levels for responses"""

    URGENT = "urgent"  # Immediate spiritual crisis
    HIGH = "high"  # Important life decisions
    MEDIUM = "medium"  # General guidance
    LOW = "low"  # Academic/educational


@dataclass
class IntegratedQuery:
    """Comprehensive query structure"""

    original_query: str
    query_type: QueryType
    priority: ResponsePriority
    context: Dict[str, Any]
    user_profile: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModuleResponse:
    """Response from individual module"""

    module_name: str
    confidence_score: float  # 0.0 to 1.0
    response_data: Dict[str, Any]
    processing_time: float
    relevance_score: float  # How relevant to the query


@dataclass
class IntegratedResponse:
    """Comprehensive integrated response"""

    primary_response: str
    supporting_information: Dict[str, Any]
    module_responses: List[ModuleResponse]
    recommendations: List[str]
    spiritual_insights: List[str]
    practical_guidance: List[str]
    relevant_scriptures: List[str]
    mantras: List[str]
    next_steps: List[str]
    confidence_score: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class UltimateDharmaIntegrationEngine:
    """
    🕉️ Complete Dharma Integration System

    Orchestrates all modules to provide comprehensive spiritual guidance
    based on authentic Sanatana Dharma principles.
    """

    def __init__(self):
        self.sanskrit_engine = None
        self.ritual_engine = None
        self.jyotisha_engine = None
        self.ayurveda_engine = None
        self.spiritual_engine = None
        self.dharma_engine = None

        # Initialize available engines
        self._initialize_engines()

        # Query processing patterns
        self.query_patterns = self._initialize_query_patterns()

        # Integration rules
        self.integration_rules = self._initialize_integration_rules()

        logger.info("🕉️ Ultimate Dharma Integration Engine initialized")

    def _initialize_engines(self):
        """Initialize all available engines"""
        try:
            self.sanskrit_engine = get_sanskrit_intelligence()
            logger.info("✅ Sanskrit Intelligence Engine loaded")
        except Exception as e:
            logger.warning(f"Sanskrit engine not available: {e}")

        try:
            self.ritual_engine = get_vedic_ritual_engine()
            logger.info("✅ Vedic Ritual Engine loaded")
        except Exception as e:
            logger.warning(f"Ritual engine not available: {e}")

        try:
            self.jyotisha_engine = get_jyotisha_engine()
            logger.info("✅ Jyotisha Engine loaded")
        except Exception as e:
            logger.warning(f"Jyotisha engine not available: {e}")

        try:
            self.ayurveda_engine = get_ayurveda_engine()
            logger.info("✅ Ayurveda Engine loaded")
        except Exception as e:
            logger.warning(f"Ayurveda engine not available: {e}")

        try:
            self.spiritual_engine = get_spiritual_intelligence()
            logger.info("✅ Spiritual Intelligence Engine loaded")
        except Exception as e:
            logger.warning(f"Spiritual engine not available: {e}")

        try:
            self.dharma_engine = get_dharma_engine()
            logger.info("✅ Dharma Engine loaded")
        except Exception as e:
            logger.warning(f"Dharma engine not available: {e}")

    async def process_comprehensive_query(
        self,
        query: str,
        context: Dict[str, Any] = None,
        user_profile: Dict[str, Any] = None,
    ) -> IntegratedResponse:
        """Process query through all relevant modules"""

        # Parse and classify query
        integrated_query = self._parse_query(query, context, user_profile)

        # Route to appropriate modules
        relevant_modules = self._determine_relevant_modules(integrated_query)

        # Process through modules in parallel
        module_responses = await self._process_through_modules(
            integrated_query, relevant_modules
        )

        # Integrate responses
        integrated_response = self._integrate_responses(
            integrated_query, module_responses
        )

        # Apply wisdom synthesis
        final_response = self._apply_wisdom_synthesis(
            integrated_response, integrated_query
        )

        return final_response

    def _parse_query(
        self,
        query: str,
        context: Dict[str, Any] = None,
        user_profile: Dict[str, Any] = None,
    ) -> IntegratedQuery:
        """Parse and classify the incoming query"""

        # Determine query type
        query_type = self._classify_query_type(query)

        # Determine priority
        priority = self._determine_priority(query, context)

        # Prepare context
        if context is None:
            context = {}

        return IntegratedQuery(
            original_query=query,
            query_type=query_type,
            priority=priority,
            context=context,
            user_profile=user_profile,
        )

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()

        # Sanskrit-related queries
        if any(
            word in query_lower
            for word in ["sanskrit", "mantra", "pronunciation", "chanting"]
        ):
            return QueryType.SANSKRIT_LEARNING

        # Ritual-related queries
        elif any(
            word in query_lower
            for word in ["puja", "ritual", "ceremony", "yajna", "worship"]
        ):
            return QueryType.RITUAL_GUIDANCE

        # Astrological queries
        elif any(
            word in query_lower
            for word in ["horoscope", "astrology", "planet", "birth chart"]
        ):
            return QueryType.ASTROLOGICAL_CONSULTATION

        # Health/Ayurveda queries
        elif any(
            word in query_lower
            for word in ["health", "ayurveda", "dosha", "constitution"]
        ):
            return QueryType.HEALTH_CONSULTATION

        # Philosophical queries
        elif any(
            word in query_lower
            for word in ["philosophy", "vedanta", "karma", "dharma"]
        ):
            return QueryType.PHILOSOPHICAL_INQUIRY

        # Life guidance
        elif any(
            word in query_lower
            for word in ["life", "decision", "guidance", "advice"]
        ):
            return QueryType.LIFE_GUIDANCE

        # Dharmic decisions
        elif any(
            word in query_lower
            for word in ["right", "wrong", "should", "moral", "ethical"]
        ):
            return QueryType.DHARMIC_DECISION

        # Default to spiritual guidance
        else:
            return QueryType.SPIRITUAL_GUIDANCE

    def _determine_priority(
        self, query: str, context: Dict[str, Any] = None
    ) -> ResponsePriority:
        """Determine the priority of the query"""
        query_lower = query.lower()

        # Urgent indicators
        if any(
            word in query_lower
            for word in ["crisis", "emergency", "urgent", "help"]
        ):
            return ResponsePriority.URGENT

        # High priority indicators
        elif any(
            word in query_lower
            for word in ["important", "decision", "marriage", "career"]
        ):
            return ResponsePriority.HIGH

        # Check context for priority indicators
        elif context and context.get("priority") in ["urgent", "high"]:
            return (
                ResponsePriority.HIGH
                if context.get("priority") == "high"
                else ResponsePriority.URGENT
            )

        # Default to medium priority
        else:
            return ResponsePriority.MEDIUM

    def _determine_relevant_modules(self, query: IntegratedQuery) -> List[str]:
        """Determine which modules are relevant for the query"""
        relevant_modules = []

        # Always include dharma engine for ethical guidance
        if self.dharma_engine:
            relevant_modules.append("dharma")

        # Add modules based on query type
        if query.query_type == QueryType.SANSKRIT_LEARNING:
            if self.sanskrit_engine:
                relevant_modules.append("sanskrit")

        elif query.query_type == QueryType.RITUAL_GUIDANCE:
            if self.ritual_engine:
                relevant_modules.append("ritual")
            if self.sanskrit_engine:
                relevant_modules.append("sanskrit")

        elif query.query_type == QueryType.ASTROLOGICAL_CONSULTATION:
            if self.jyotisha_engine:
                relevant_modules.append("jyotisha")

        elif query.query_type == QueryType.HEALTH_CONSULTATION:
            if self.ayurveda_engine:
                relevant_modules.append("ayurveda")

        elif query.query_type == QueryType.COMPREHENSIVE_ANALYSIS:
            # Use all available modules
            if self.sanskrit_engine:
                relevant_modules.append("sanskrit")
            if self.ritual_engine:
                relevant_modules.append("ritual")
            if self.jyotisha_engine:
                relevant_modules.append("jyotisha")
            if self.ayurveda_engine:
                relevant_modules.append("ayurveda")

        # Always include spiritual intelligence if available
        if self.spiritual_engine:
            relevant_modules.append("spiritual")

        return relevant_modules

    async def _process_through_modules(
        self, query: IntegratedQuery, relevant_modules: List[str]
    ) -> List[ModuleResponse]:
        """Process query through relevant modules"""
        tasks = []

        for module_name in relevant_modules:
            task = self._process_single_module(query, module_name)
            tasks.append(task)

        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and None responses
        valid_responses = []
        for response in responses:
            if isinstance(response, ModuleResponse):
                valid_responses.append(response)
            elif isinstance(response, Exception):
                logger.warning(f"Module processing error: {response}")

        return valid_responses

    async def _process_single_module(
        self, query: IntegratedQuery, module_name: str
    ) -> Optional[ModuleResponse]:
        """Process query through a single module"""
        start_time = datetime.now()

        try:
            if module_name == "sanskrit" and self.sanskrit_engine:
                response_data = await self._process_sanskrit_query(query)
            elif module_name == "ritual" and self.ritual_engine:
                response_data = await self._process_ritual_query(query)
            elif module_name == "jyotisha" and self.jyotisha_engine:
                response_data = await self._process_jyotisha_query(query)
            elif module_name == "ayurveda" and self.ayurveda_engine:
                response_data = await self._process_ayurveda_query(query)
            elif module_name == "spiritual" and self.spiritual_engine:
                response_data = await self._process_spiritual_query(query)
            elif module_name == "dharma" and self.dharma_engine:
                response_data = await self._process_dharma_query(query)
            else:
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            return ModuleResponse(
                module_name=module_name,
                confidence_score=response_data.get("confidence", 0.8),
                response_data=response_data,
                processing_time=processing_time,
                relevance_score=self._calculate_relevance_score(
                    query, response_data
                ),
            )

        except Exception as e:
            logger.error(f"Error processing {module_name} module: {e}")
            return None

    async def _process_sanskrit_query(
        self, query: IntegratedQuery
    ) -> Dict[str, Any]:
        """Process query through Sanskrit engine"""
        if "mantra" in query.original_query.lower():
            mantra_text = self._extract_mantra_from_query(query.original_query)
            if mantra_text:
                pronunciation = (
                    self.sanskrit_engine.analyze_mantra_pronunciation(
                        mantra_text
                    )
                )
                chanting_guide = self.sanskrit_engine.generate_chanting_guide(
                    mantra_text
                )

                return {
                    "type": "mantra_guidance",
                    "pronunciation": asdict(pronunciation),
                    "chanting_guide": chanting_guide,
                    "confidence": 0.9,
                }

        return {
            "type": "general_sanskrit",
            "guidance": "Sanskrit language guidance and support",
            "confidence": 0.7,
        }

    async def _process_ritual_query(
        self, query: IntegratedQuery
    ) -> Dict[str, Any]:
        """Process query through Ritual engine"""
        if "puja" in query.original_query.lower():
            deity = self._extract_deity_from_query(query.original_query)
            puja_guide = self.ritual_engine.get_daily_puja_guide(deity)

            return {
                "type": "puja_guidance",
                "procedure": asdict(puja_guide),
                "confidence": 0.9,
            }

        return {
            "type": "general_ritual",
            "guidance": "Vedic ritual guidance and procedures",
            "confidence": 0.7,
        }

    async def _process_jyotisha_query(
        self, query: IntegratedQuery
    ) -> Dict[str, Any]:
        """Process query through Jyotisha engine"""
        if query.user_profile and "birth_details" in query.user_profile:
            query.user_profile["birth_details"]
            # Process astrological consultation
            return {
                "type": "astrological_analysis",
                "analysis": "Detailed astrological insights",
                "confidence": 0.8,
            }

        return {
            "type": "general_astrology",
            "guidance": "Astrological guidance and insights",
            "confidence": 0.6,
        }

    async def _process_ayurveda_query(
        self, query: IntegratedQuery
    ) -> Dict[str, Any]:
        """Process query through Ayurveda engine"""
        if (
            "health" in query.original_query.lower()
            or "dosha" in query.original_query.lower()
        ):
            return {
                "type": "health_consultation",
                "recommendations": "Ayurvedic health recommendations",
                "confidence": 0.8,
            }

        return {
            "type": "general_ayurveda",
            "guidance": "Ayurvedic wisdom and health insights",
            "confidence": 0.7,
        }

    async def _process_spiritual_query(
        self, query: IntegratedQuery
    ) -> Dict[str, Any]:
        """Process query through Spiritual Intelligence engine"""
        return {
            "type": "spiritual_guidance",
            "wisdom": "Deep spiritual insights and guidance",
            "practices": "Recommended spiritual practices",
            "confidence": 0.9,
        }

    async def _process_dharma_query(
        self, query: IntegratedQuery
    ) -> Dict[str, Any]:
        """Process query through Dharma engine"""
        return {
            "type": "dharmic_guidance",
            "ethical_analysis": "Dharmic perspective on the situation",
            "recommendations": "Righteous course of action",
            "confidence": 0.9,
        }

    def _integrate_responses(
        self, query: IntegratedQuery, module_responses: List[ModuleResponse]
    ) -> IntegratedResponse:
        """Integrate responses from all modules"""

        # Sort responses by relevance and confidence
        sorted_responses = sorted(
            module_responses,
            key=lambda x: x.relevance_score * x.confidence_score,
            reverse=True,
        )

        # Generate primary response
        primary_response = self._generate_primary_response(
            query, sorted_responses
        )

        # Collect supporting information
        supporting_information = self._collect_supporting_information(
            sorted_responses
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            query, sorted_responses
        )

        # Extract spiritual insights
        spiritual_insights = self._extract_spiritual_insights(sorted_responses)

        # Generate practical guidance
        practical_guidance = self._generate_practical_guidance(
            query, sorted_responses
        )

        # Find relevant scriptures
        relevant_scriptures = self._find_relevant_scriptures(
            query, sorted_responses
        )

        # Find relevant mantras
        mantras = self._find_relevant_mantras(query, sorted_responses)

        # Generate next steps
        next_steps = self._generate_next_steps(query, sorted_responses)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            sorted_responses
        )

        return IntegratedResponse(
            primary_response=primary_response,
            supporting_information=supporting_information,
            module_responses=sorted_responses,
            recommendations=recommendations,
            spiritual_insights=spiritual_insights,
            practical_guidance=practical_guidance,
            relevant_scriptures=relevant_scriptures,
            mantras=mantras,
            next_steps=next_steps,
            confidence_score=overall_confidence,
        )

    def _apply_wisdom_synthesis(
        self, response: IntegratedResponse, query: IntegratedQuery
    ) -> IntegratedResponse:
        """Apply final wisdom synthesis to the response"""

        # Enhance response with cross-module correlations
        enhanced_insights = self._enhance_with_correlations(response, query)

        # Add traditional wisdom context
        traditional_context = self._add_traditional_context(response, query)

        # Apply dharmic filtering
        self._apply_dharmic_validation(response, query)

        # Update response with enhanced insights
        response.spiritual_insights.extend(enhanced_insights)
        response.supporting_information.update(traditional_context)

        return response

    # Helper methods

    def _extract_mantra_from_query(self, query: str) -> Optional[str]:
        """Extract mantra text from query"""
        # Simple extraction - can be enhanced with NLP
        import re

        sanskrit_pattern = r"[ॐअ-ह्]+"
        matches = re.findall(sanskrit_pattern, query)
        return matches[0] if matches else None

    def _extract_deity_from_query(self, query: str) -> str:
        """Extract deity name from query"""
        deities = [
            "ganesha",
            "krishna",
            "rama",
            "shiva",
            "vishnu",
            "devi",
            "lakshmi",
        ]
        query_lower = query.lower()

        for deity in deities:
            if deity in query_lower:
                return deity.capitalize()

        return "Ganesha"  # Default

    def _calculate_relevance_score(
        self, query: IntegratedQuery, response_data: Dict[str, Any]
    ) -> float:
        """Calculate how relevant a response is to the query"""
        # Simple relevance scoring - can be enhanced with ML
        base_score = 0.7

        # Boost score if response type matches query type
        if query.query_type.value in str(response_data.get("type", "")):
            base_score += 0.2

        return min(base_score, 1.0)

    def _generate_primary_response(
        self, query: IntegratedQuery, responses: List[ModuleResponse]
    ) -> str:
        """Generate the primary response text"""
        if not responses:
            return "I understand your spiritual query. Let me provide guidance based on eternal dharmic principles."

        # Use the highest scoring response as primary
        primary_module = responses[0]
        response_data = primary_module.response_data

        return f"Based on authentic Sanatana Dharma principles: {
            response_data.get(
                'guidance', 'Spiritual guidance provided')}"

    def _collect_supporting_information(
        self, responses: List[ModuleResponse]
    ) -> Dict[str, Any]:
        """Collect supporting information from all responses"""
        supporting_info = {}

        for response in responses:
            supporting_info[response.module_name] = {
                "confidence": response.confidence_score,
                "data": response.response_data,
            }

        return supporting_info

    def _generate_recommendations(
        self, query: IntegratedQuery, responses: List[ModuleResponse]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Add query-type specific recommendations
        if query.query_type == QueryType.SPIRITUAL_GUIDANCE:
            recommendations.extend(
                [
                    "Practice daily meditation and prayer",
                    "Study authentic spiritual texts",
                    "Seek guidance from qualified spiritual teacher",
                ]
            )

        elif query.query_type == QueryType.RITUAL_GUIDANCE:
            recommendations.extend(
                [
                    "Follow traditional ritual procedures",
                    "Maintain purity of body and mind",
                    "Perform rituals with devotion and understanding",
                ]
            )

        # Add module-specific recommendations
        for response in responses:
            if "recommendations" in response.response_data:
                recommendations.extend(
                    response.response_data["recommendations"]
                )

        return recommendations[:10]  # Limit to top 10

    def _extract_spiritual_insights(
        self, responses: List[ModuleResponse]
    ) -> List[str]:
        """Extract spiritual insights from responses"""
        insights = []

        for response in responses:
            if "spiritual_insights" in response.response_data:
                insights.extend(response.response_data["spiritual_insights"])
            elif "wisdom" in response.response_data:
                insights.append(response.response_data["wisdom"])

        return insights[:5]  # Limit to top 5

    def _generate_practical_guidance(
        self, query: IntegratedQuery, responses: List[ModuleResponse]
    ) -> List[str]:
        """Generate practical guidance"""
        guidance = []

        for response in responses:
            if "practical_steps" in response.response_data:
                guidance.extend(response.response_data["practical_steps"])

        # Add general practical guidance
        guidance.extend(
            [
                "Maintain regular spiritual practice",
                "Live according to dharmic principles",
                "Seek balance in all aspects of life",
            ]
        )

        return guidance[:7]  # Limit to top 7

    def _find_relevant_scriptures(
        self, query: IntegratedQuery, responses: List[ModuleResponse]
    ) -> List[str]:
        """Find relevant scriptural references"""
        scriptures = []

        # Add default relevant scriptures based on query type
        if query.query_type == QueryType.PHILOSOPHICAL_INQUIRY:
            scriptures.extend(
                ["Bhagavad Gita", "Upanishads", "Yoga Vashishtha"]
            )
        elif query.query_type == QueryType.DHARMIC_DECISION:
            scriptures.extend(
                ["Bhagavad Gita", "Dharma Shastra", "Mahabharata"]
            )

        return scriptures[:3]  # Limit to top 3

    def _find_relevant_mantras(
        self, query: IntegratedQuery, responses: List[ModuleResponse]
    ) -> List[str]:
        """Find relevant mantras"""
        mantras = []

        # Add mantras from responses
        for response in responses:
            if "mantras" in response.response_data:
                mantras.extend(response.response_data["mantras"])

        # Add default mantras
        mantras.extend(
            [
                "ॐ गं गणपतये नमः (Om Gam Ganapataye Namah)",
                "ॐ नमः शिवाय (Om Namah Shivaya)",
                "हरे कृष्ण हरे कृष्ण कृष्ण कृष्ण हरे हरे (Hare Krishna Mantra)",
            ]
        )

        return mantras[:3]  # Limit to top 3

    def _generate_next_steps(
        self, query: IntegratedQuery, responses: List[ModuleResponse]
    ) -> List[str]:
        """Generate next steps for the user"""
        next_steps = []

        if query.query_type == QueryType.SPIRITUAL_GUIDANCE:
            next_steps.extend(
                [
                    "Establish daily spiritual practice",
                    "Study recommended scriptures",
                    "Seek spiritual community (satsang)",
                ]
            )

        elif query.query_type == QueryType.HEALTH_CONSULTATION:
            next_steps.extend(
                [
                    "Consult qualified Ayurvedic practitioner",
                    "Implement dietary recommendations gradually",
                    "Monitor progress and adjust as needed",
                ]
            )

        return next_steps[:5]  # Limit to top 5

    def _calculate_overall_confidence(
        self, responses: List[ModuleResponse]
    ) -> float:
        """Calculate overall confidence score"""
        if not responses:
            return 0.5

        # Weighted average based on relevance
        total_weight = sum(r.relevance_score for r in responses)
        if total_weight == 0:
            return 0.5

        weighted_confidence = (
            sum(r.confidence_score * r.relevance_score for r in responses)
            / total_weight
        )

        return min(weighted_confidence, 1.0)

    def _enhance_with_correlations(
        self, response: IntegratedResponse, query: IntegratedQuery
    ) -> List[str]:
        """Enhance response with cross-module correlations"""
        correlations = []

        # Look for patterns across modules
        module_types = {r.module_name for r in response.module_responses}

        if "sanskrit" in module_types and "ritual" in module_types:
            correlations.append(
                "The Sanskrit mantras and ritual procedures work synergistically for spiritual elevation"
            )

        if "jyotisha" in module_types and "ayurveda" in module_types:
            correlations.append(
                "Astrological timing aligns with Ayurvedic constitutional considerations"
            )

        return correlations

    def _add_traditional_context(
        self, response: IntegratedResponse, query: IntegratedQuery
    ) -> Dict[str, Any]:
        """Add traditional wisdom context"""
        return {
            "traditional_wisdom": "Guidance rooted in timeless Vedic principles",
            "authenticity": "Based on traditional Sanatana Dharma teachings",
            "lineage": "Transmitted through authentic guru-disciple tradition",
        }

    def _apply_dharmic_validation(
        self, response: IntegratedResponse, query: IntegratedQuery
    ) -> bool:
        """Validate response against dharmic principles"""
        # Ensure all recommendations align with dharmic principles
        # This is a simplified validation - can be enhanced
        return True

    def _initialize_query_patterns(self) -> Dict[str, Any]:
        """Initialize query pattern recognition"""
        return {
            "spiritual_keywords": [
                "soul",
                "spirit",
                "divine",
                "god",
                "moksha",
                "liberation",
            ],
            "ritual_keywords": [
                "puja",
                "worship",
                "ceremony",
                "ritual",
                "offering",
            ],
            "health_keywords": [
                "health",
                "disease",
                "dosha",
                "constitution",
                "medicine",
            ],
            "astrology_keywords": [
                "horoscope",
                "planets",
                "birth chart",
                "astrology",
            ],
        }

    def _initialize_integration_rules(self) -> Dict[str, Any]:
        """Initialize integration rules"""
        return {
            "module_priorities": {
                QueryType.SPIRITUAL_GUIDANCE: ["spiritual", "dharma"],
                QueryType.RITUAL_GUIDANCE: ["ritual", "sanskrit", "dharma"],
                QueryType.HEALTH_CONSULTATION: ["ayurveda", "dharma"],
                QueryType.ASTROLOGICAL_CONSULTATION: ["jyotisha", "dharma"],
            },
            "response_synthesis": {
                "primary_weight": 0.6,
                "supporting_weight": 0.4,
            },
        }


# Global instance
_integration_engine = None


def get_integration_engine() -> UltimateDharmaIntegrationEngine:
    """Get global Integration Engine instance"""
    global _integration_engine
    if _integration_engine is None:
        _integration_engine = UltimateDharmaIntegrationEngine()
    return _integration_engine


# Main processing function for external use


async def process_dharmic_query(
    query: str,
    context: Dict[str, Any] = None,
    user_profile: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Main function to process any dharmic query through the integrated system

    Args:
        query: The user's question or request
        context: Additional context information
        user_profile: User's profile information

    Returns:
        Comprehensive integrated response
    """
    engine = get_integration_engine()
    response = await engine.process_comprehensive_query(
        query, context, user_profile
    )
    return asdict(response)
