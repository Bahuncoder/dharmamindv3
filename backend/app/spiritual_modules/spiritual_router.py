"""
Spiritual Modules Router - Enhanced Chakra Integration
=====================================================

This module provides intelligent routing and integration across
all spiritual chakra modules for comprehensive dharmic guidance.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiritualPath(Enum):
    """Spiritual paths available in the system"""
    JNANA = "jnana"  # Knowledge and wisdom
    KARMA = "karma"  # Action and duty  
    BHAKTI = "bhakti"  # Love and devotion
    SEVA = "seva"  # Service and compassion
    DHARMA = "dharma"  # Righteousness and duty
    MOKSHA = "moksha"  # Liberation and transcendence
    YOGA = "yoga"  # Union and practice
    AHIMSA = "ahimsa"  # Non-violence and compassion

class QueryIntent(Enum):
    """Types of spiritual inquiries"""
    KNOWLEDGE_SEEKING = "knowledge_seeking"
    ACTION_GUIDANCE = "action_guidance"
    DEVOTIONAL_PRACTICE = "devotional_practice"
    SERVICE_CALLING = "service_calling"
    RIGHTEOUS_LIVING = "righteous_living"
    LIBERATION_INQUIRY = "liberation_inquiry"
    PRACTICE_GUIDANCE = "practice_guidance"
    COMPASSION_CULTIVATION = "compassion_cultivation"
    GENERAL_WISDOM = "general_wisdom"
    INTEGRATED_PATH = "integrated_path"

class GuidanceMode(Enum):
    """Modes of guidance delivery"""
    SINGLE_PATH = "single_path"  # Focus on one primary path
    MULTI_PATH = "multi_path"  # Integrate multiple paths
    ADAPTIVE = "adaptive"  # Dynamically adjust based on user needs

@dataclass
class PathActivation:
    """Represents activation level and relevance of each spiritual path"""
    path: SpiritualPath
    activation_level: float  # 0-1 scale
    relevance_score: float  # 0-1 scale
    keywords_matched: List[str]
    reasoning: str

@dataclass
class IntegratedGuidance:
    """Comprehensive guidance from multiple spiritual paths"""
    primary_guidance: str
    primary_path: SpiritualPath
    secondary_perspectives: Dict[SpiritualPath, str]
    synthesis: str
    practical_steps: List[str]
    spiritual_principles: List[str]
    sanskrit_references: List[str] = field(default_factory=list)
    recommended_practices: List[str] = field(default_factory=list)
    dharmic_principles: List[str] = field(default_factory=list)

@dataclass
class SpiritualRouterResponse:
    """Complete response from spiritual router"""
    integrated_guidance: IntegratedGuidance
    individual_responses: Dict[SpiritualPath, Any]
    routing_analysis: List[PathActivation]
    user_journey_insights: List[str]
    next_recommended_inquiry: str

class SpiritualRouter:
    """Advanced spiritual path routing and integration system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Path detection patterns
        self.path_keywords = {
            SpiritualPath.JNANA: {
                'primary': ['knowledge', 'wisdom', 'understanding', 'learn', 'study', 'truth', 'reality', 'consciousness', 'awareness', 'self-inquiry', 'meditation', 'contemplation'],
                'sanskrit': ['jnana', 'jñāna', 'vidya', 'viveka', 'vichara', 'atman', 'brahman', 'vedanta'],
                'concepts': ['who am i', 'nature of self', 'ultimate reality', 'enlightenment', 'awakening', 'realization']
            },
            SpiritualPath.KARMA: {
                'primary': ['action', 'duty', 'work', 'responsibility', 'ethics', 'dharma', 'decision', 'choice', 'behavior', 'conduct'],
                'sanskrit': ['karma', 'dharma', 'yajna', 'nishkama', 'karmayoga', 'svadharma'],
                'concepts': ['what should i do', 'right action', 'moral dilemma', 'work life', 'family duty', 'social responsibility']
            },
            SpiritualPath.BHAKTI: {
                'primary': ['love', 'devotion', 'faith', 'surrender', 'worship', 'prayer', 'divine', 'god', 'heart', 'emotion'],
                'sanskrit': ['bhakti', 'prema', 'shraddha', 'ishvara', 'guru', 'mantra', 'kirtan'],
                'concepts': ['devotional practice', 'love for divine', 'surrender to god', 'faith and prayer', 'heart opening']
            },
            SpiritualPath.SEVA: {
                'primary': ['service', 'help', 'compassion', 'kindness', 'charity', 'giving', 'volunteer', 'care', 'support'],
                'sanskrit': ['seva', 'karuna', 'daya', 'ahimsa', 'dana'],
                'concepts': ['how to help', 'serve others', 'show compassion', 'express love', 'make difference', 'community service']
            },
            SpiritualPath.DHARMA: {
                'primary': ['dharma', 'righteousness', 'duty', 'moral', 'ethical', 'right', 'justice', 'order', 'law'],
                'sanskrit': ['dharma', 'rita', 'satya', 'nyaya', 'swadharma'],
                'concepts': ['right living', 'moral duty', 'ethical behavior', 'righteous path', 'natural order']
            },
            SpiritualPath.MOKSHA: {
                'primary': ['liberation', 'freedom', 'transcendence', 'enlightenment', 'awakening', 'realization', 'spiritual goal', 'ultimate purpose'],
                'sanskrit': ['moksha', 'mukti', 'kaivalya', 'nirvana', 'samadhi', 'jivanmukta'],
                'concepts': ['spiritual goal', 'ultimate purpose', 'end of suffering', 'final realization', 'highest attainment']
            },
            SpiritualPath.YOGA: {
                'primary': ['yoga', 'practice', 'meditation', 'asana', 'pranayama', 'union', 'discipline', 'control'],
                'sanskrit': ['yoga', 'dhyana', 'dharana', 'samadhi', 'yama', 'niyama', 'asana', 'pranayama'],
                'concepts': ['spiritual practice', 'yoga practice', 'meditation technique', 'mind control', 'spiritual discipline']
            },
            SpiritualPath.AHIMSA: {
                'primary': ['non-violence', 'compassion', 'kindness', 'peaceful', 'gentle', 'harmless', 'protection'],
                'sanskrit': ['ahimsa', 'karuna', 'maitri', 'daya'],
                'concepts': ['non-violence', 'peaceful living', 'compassionate action', 'protecting life', 'gentle approach']
            }
        }
        
        # Context patterns for life stages and situations
        self.context_patterns = {
            'life_stage': {
                'student': ['study', 'learn', 'education', 'school', 'college', 'brahmachari'],
                'householder': ['family', 'work', 'career', 'marriage', 'children', 'grihastha'],
                'seeker': ['spiritual', 'meditation', 'practice', 'teacher', 'path', 'vanaprastha'],
                'renunciant': ['renunciation', 'sannyasa', 'monk', 'ascetic', 'withdrawal']
            },
            'emotional_state': {
                'confused': ['confused', 'lost', 'uncertain', 'don\'t know', 'unclear'],
                'suffering': ['pain', 'hurt', 'suffering', 'grief', 'sadness', 'depression'],
                'seeking': ['seeking', 'searching', 'looking for', 'want to find', 'need guidance'],
                'peaceful': ['peaceful', 'content', 'happy', 'grateful', 'blessed'],
                'devoted': ['devoted', 'faithful', 'loving', 'surrendered', 'trusting']
            }
        }
        
        self.logger.info("🕉️ Spiritual Router initialized - Universal dharma guidance system activated")
    
    async def route_spiritual_query(self, query: str, user_context: Optional[Dict[str, Any]] = None,
                                  guidance_mode: GuidanceMode = GuidanceMode.ADAPTIVE) -> SpiritualRouterResponse:
        """Route spiritual query to appropriate paths and provide integrated guidance"""
        
        try:
            self.logger.debug(f"🔄 Routing spiritual query: {query[:50]}...")
            
            # Analyze query for spiritual path activations
            activations = self.analyze_query_for_paths(query, user_context or {})
            
            # Determine routing strategy
            routing_strategy = self.determine_routing_strategy(activations, guidance_mode)
            
            # Get guidance from relevant paths
            individual_responses = await self.get_path_responses(
                query, activations, routing_strategy, user_context or {}
            )
            
            # Integrate responses into unified guidance
            integrated_guidance = await self.integrate_path_responses(
                query, individual_responses, activations
            )
            
            # Generate user journey insights
            journey_insights = self.generate_user_journey_insights(query, activations, user_context or {})
            
            # Recommend next inquiry
            next_inquiry = self.recommend_next_inquiry(query, activations, integrated_guidance)
            
            response = SpiritualRouterResponse(
                integrated_guidance=integrated_guidance,
                individual_responses=individual_responses,
                routing_analysis=activations,
                user_journey_insights=journey_insights,
                next_recommended_inquiry=next_inquiry
            )
            
            self.logger.debug(f"✅ Spiritual routing completed - Primary path: {integrated_guidance.primary_path.value}")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error in spiritual routing: {str(e)}")
            return self.create_fallback_response(query)
    
    def analyze_query_for_paths(self, query: str, user_context: Dict[str, Any]) -> List[PathActivation]:
        """Analyze query to determine which spiritual paths should be activated"""
        
        query_lower = query.lower()
        activations = []
        
        for path in SpiritualPath:
            keywords = self.path_keywords[path]
            matched_keywords = []
            relevance_score = 0.0
            
            # Check primary keywords
            for keyword in keywords['primary']:
                if keyword in query_lower:
                    matched_keywords.append(keyword)
                    relevance_score += 0.1
            
            # Check Sanskrit terms (higher weight)
            for sanskrit_term in keywords['sanskrit']:
                if sanskrit_term in query_lower:
                    matched_keywords.append(sanskrit_term)
                    relevance_score += 0.15
            
            # Check concept patterns (highest weight)
            for concept in keywords['concepts']:
                if concept in query_lower:
                    matched_keywords.append(concept)
                    relevance_score += 0.2
            
            # Apply context modifiers
            relevance_score = self.apply_context_modifiers(
                relevance_score, path, query_lower, user_context
            )
            
            # Calculate activation level
            activation_level = min(relevance_score, 1.0)
            
            # Generate reasoning
            reasoning = self.generate_activation_reasoning(
                path, matched_keywords, relevance_score, user_context
            )
            
            activation = PathActivation(
                path=path,
                activation_level=activation_level,
                relevance_score=relevance_score,
                keywords_matched=matched_keywords,
                reasoning=reasoning
            )
            
            activations.append(activation)
        
        # Sort by relevance score
        activations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return activations
    
    def apply_context_modifiers(self, base_score: float, path: SpiritualPath,
                              query_lower: str, user_context: Dict[str, Any]) -> float:
        """Apply contextual modifiers to relevance scores"""
        
        modified_score = base_score
        
        # Life stage modifiers
        life_stage = user_context.get('life_stage', 'unknown')
        stage_modifiers = {
            SpiritualPath.JNANA: {'student': 1.2, 'seeker': 1.3, 'renunciant': 1.1},
            SpiritualPath.KARMA: {'householder': 1.3, 'student': 1.1},
            SpiritualPath.BHAKTI: {'householder': 1.2, 'seeker': 1.3},
            SpiritualPath.SEVA: {'householder': 1.2, 'seeker': 1.1},
            SpiritualPath.DHARMA: {'householder': 1.3, 'student': 1.2},
            SpiritualPath.MOKSHA: {'seeker': 1.4, 'renunciant': 1.2},
            SpiritualPath.YOGA: {'seeker': 1.3, 'student': 1.1},
            SpiritualPath.AHIMSA: {'householder': 1.1, 'seeker': 1.2}
        }
        
        if life_stage in stage_modifiers.get(path, {}):
            modified_score *= stage_modifiers[path][life_stage]
        
        # Emotional state modifiers
        emotional_state = user_context.get('emotional_state', 'neutral')
        emotion_modifiers = {
            SpiritualPath.JNANA: {'confused': 1.3, 'seeking': 1.2},
            SpiritualPath.KARMA: {'confused': 1.2, 'suffering': 1.1},
            SpiritualPath.BHAKTI: {'devoted': 1.4, 'seeking': 1.2},
            SpiritualPath.SEVA: {'suffering': 1.2, 'peaceful': 1.2},
            SpiritualPath.DHARMA: {'confused': 1.3, 'suffering': 1.1},
            SpiritualPath.MOKSHA: {'seeking': 1.4, 'suffering': 1.2},
            SpiritualPath.YOGA: {'seeking': 1.3, 'confused': 1.1},
            SpiritualPath.AHIMSA: {'suffering': 1.3, 'peaceful': 1.1}
        }
        
        if emotional_state in emotion_modifiers.get(path, {}):
            modified_score *= emotion_modifiers[path][emotional_state]
        
        return modified_score
    
    def generate_activation_reasoning(self, path: SpiritualPath, matched_keywords: List[str],
                                    relevance_score: float, user_context: Dict[str, Any]) -> str:
        """Generate reasoning for path activation"""
        
        if not matched_keywords:
            return f"{path.value.title()} path has minimal relevance to this query"
        
        primary_reason = f"Activated due to keywords: {', '.join(matched_keywords[:3])}"
        
        if relevance_score > 0.5:
            confidence = "High"
        elif relevance_score > 0.3:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        context_note = ""
        if user_context.get('life_stage'):
            context_note = f" (Enhanced by {user_context['life_stage']} life stage)"
        
        return f"{confidence} relevance. {primary_reason}{context_note}"
    
    def determine_routing_strategy(self, activations: List[PathActivation],
                                 guidance_mode: GuidanceMode) -> str:
        """Determine routing strategy based on activations and mode"""
        
        if guidance_mode == GuidanceMode.SINGLE_PATH:
            return "single_primary"
        elif guidance_mode == GuidanceMode.MULTI_PATH:
            return "all_relevant"
        else:  # ADAPTIVE
            high_relevance = [a for a in activations if a.relevance_score > 0.4]
            
            if len(high_relevance) == 1:
                return "single_primary"
            elif len(high_relevance) <= 3:
                return "multi_focused"
            else:
                return "integrated_synthesis"
    
    async def get_path_responses(self, query: str, activations: List[PathActivation],
                               strategy: str, user_context: Dict[str, Any]) -> Dict[SpiritualPath, Any]:
        """Get responses from relevant spiritual paths"""
        
        responses = {}
        
        # Determine which paths to query
        if strategy == "single_primary":
            target_paths = [activations[0].path] if activations else [SpiritualPath.JNANA]
        elif strategy == "multi_focused":
            target_paths = [a.path for a in activations if a.relevance_score > 0.3][:3]
        else:  # integrated_synthesis or all_relevant
            target_paths = [a.path for a in activations if a.relevance_score > 0.1]
        
        # Query each relevant path
        for path in target_paths:
            try:
                response = await self.query_specific_path(path, query, user_context)
                if response:
                    responses[path] = response
            except Exception as e:
                self.logger.error(f"❌ Error querying {path.value} path: {str(e)}")
        
        return responses
    
    async def query_specific_path(self, path: SpiritualPath, query: str,
                                user_context: Dict[str, Any]) -> Any:
        """Query a specific spiritual path (placeholder for actual implementations)"""
        
        try:
            # This would integrate with actual spiritual module implementations
            # For now, return a basic response structure
            
            path_guidance = {
                SpiritualPath.JNANA: f"From the path of knowledge: {query} can be understood through self-inquiry and contemplation of the nature of consciousness and reality.",
                SpiritualPath.KARMA: f"From the path of action: {query} requires understanding your dharmic duty and performing actions without attachment to results.",
                SpiritualPath.BHAKTI: f"From the path of devotion: {query} can be approached through surrendering to the divine with love and faith.",
                SpiritualPath.SEVA: f"From the path of service: {query} can be addressed through selfless service to others and seeing the divine in all beings.",
                SpiritualPath.DHARMA: f"From the path of righteousness: {query} requires understanding and following the principles of righteous living.",
                SpiritualPath.MOKSHA: f"From the path of liberation: {query} points toward the ultimate goal of freedom from all forms of suffering.",
                SpiritualPath.YOGA: f"From the path of union: {query} can be approached through disciplined practice and the eight-fold path of yoga.",
                SpiritualPath.AHIMSA: f"From the path of non-violence: {query} should be considered with utmost compassion and minimal harm to all beings."
            }
            
            return {
                'guidance': path_guidance.get(path, "Wisdom from the eternal dharma"),
                'practices': [f"{path.value.title()} practice", f"{path.value.title()} study"],
                'principles': [f"{path.value.title()} principle", "Universal love and compassion"]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in path query for {path.value}: {str(e)}")
            return None
    
    async def integrate_path_responses(self, query: str, individual_responses: Dict[SpiritualPath, Any],
                                     activations: List[PathActivation]) -> IntegratedGuidance:
        """Integrate responses from multiple paths into unified guidance"""
        
        if not individual_responses:
            return self.create_fallback_guidance(query)
        
        # Determine primary path
        primary_path = max(activations, key=lambda x: x.relevance_score).path
        if primary_path not in individual_responses:
            primary_path = list(individual_responses.keys())[0]
        
        # Extract primary guidance
        primary_response = individual_responses[primary_path]
        primary_guidance = primary_response.get('guidance', 'Dharmic wisdom guides your path')
        
        # Extract secondary perspectives
        secondary_perspectives = {}
        for path, response in individual_responses.items():
            if path != primary_path:
                perspective = response.get('guidance', f'{path.value.title()} wisdom')
                secondary_perspectives[path] = f"**{path.value.title()} Perspective**: {perspective}"
        
        # Generate synthesis
        synthesis = await self.generate_synthesis(
            query, primary_guidance, secondary_perspectives, primary_path
        )
        
        # Extract practical elements
        practical_steps = self.extract_practical_steps(individual_responses)
        spiritual_principles = self.extract_spiritual_principles(individual_responses)
        recommended_practices = self.extract_recommended_practices(individual_responses)
        dharmic_principles = self.extract_dharmic_principles(individual_responses)
        
        return IntegratedGuidance(
            primary_guidance=primary_guidance,
            primary_path=primary_path,
            secondary_perspectives=secondary_perspectives,
            synthesis=synthesis,
            practical_steps=practical_steps,
            spiritual_principles=spiritual_principles,
            recommended_practices=recommended_practices,
            dharmic_principles=dharmic_principles
        )
    
    async def generate_synthesis(self, query: str, primary_guidance: str,
                               secondary_perspectives: Dict[SpiritualPath, str],
                               primary_path: SpiritualPath) -> str:
        """Generate synthesis of multiple path perspectives"""
        
        synthesis_template = f"""
        **Integrated Dharmic Guidance:**
        
        Your inquiry touches the heart of dharmic living. The path of {primary_path.value} provides the primary guidance, while other paths offer complementary wisdom for the same spiritual truth.
        
        {chr(10).join(secondary_perspectives.values()) if secondary_perspectives else "Focus on developing this primary path while remaining open to insights from other approaches."}
        
        **Universal Principle**: All authentic spiritual paths lead to the same ultimate reality - the realization of our true divine nature and the establishment of righteousness in the world. Let knowledge inform action, action express devotion, devotion manifest as service, and all practice aim toward liberation while upholding dharma.
        
        **Integration Wisdom**: Use the understanding from your primary path to inform your approach, while allowing insights from other paths to enrich and deepen your practice. Remember that diversity in spiritual expression serves the unity of truth.
        """
        
        return synthesis_template.strip()
    
    def extract_practical_steps(self, responses: Dict[SpiritualPath, Any]) -> List[str]:
        """Extract practical steps from all responses"""
        steps = []
        for path, response in responses.items():
            if isinstance(response, dict) and 'practices' in response:
                steps.extend(response['practices'][:2])
        return list(set(steps))  # Remove duplicates
    
    def extract_spiritual_principles(self, responses: Dict[SpiritualPath, Any]) -> List[str]:
        """Extract spiritual principles from all responses"""
        principles = []
        for path, response in responses.items():
            if isinstance(response, dict) and 'principles' in response:
                principles.extend(response['principles'][:2])
        return list(set(principles))  # Remove duplicates
    
    def extract_recommended_practices(self, responses: Dict[SpiritualPath, Any]) -> List[str]:
        """Extract recommended practices from all responses"""
        practices = []
        for path, response in responses.items():
            if isinstance(response, dict) and 'practices' in response:
                practices.extend(response['practices'][:2])
        return list(set(practices))  # Remove duplicates
    
    def extract_dharmic_principles(self, responses: Dict[SpiritualPath, Any]) -> List[str]:
        """Extract dharmic principles from all responses"""
        return [
            "Truth (Satya) - Speak and live truthfully",
            "Non-violence (Ahimsa) - Harm no living being",
            "Duty (Dharma) - Fulfill your righteous obligations", 
            "Service (Seva) - Serve others selflessly",
            "Devotion (Bhakti) - Cultivate love for the divine",
            "Wisdom (Jnana) - Seek knowledge of ultimate reality",
            "Practice (Yoga) - Maintain spiritual discipline",
            "Liberation (Moksha) - Aim for freedom from suffering"
        ]
    
    def generate_user_journey_insights(self, query: str, activations: List[PathActivation],
                                     user_context: Dict[str, Any]) -> List[str]:
        """Generate insights about user's spiritual journey"""
        
        insights = []
        
        # Analyze activation patterns
        primary_activation = activations[0] if activations else None
        if primary_activation:
            path_insights = {
                SpiritualPath.JNANA: "Your inquiry shows a strong orientation toward understanding and wisdom",
                SpiritualPath.KARMA: "Your question reflects a practical approach to righteous living",
                SpiritualPath.BHAKTI: "Your inquiry demonstrates a heart-oriented approach to spirituality",
                SpiritualPath.SEVA: "Your question shows a compassionate desire to serve others",
                SpiritualPath.DHARMA: "Your inquiry reflects a deep commitment to righteous living",
                SpiritualPath.MOKSHA: "Your question indicates a profound longing for spiritual liberation",
                SpiritualPath.YOGA: "Your inquiry shows dedication to spiritual practice and discipline",
                SpiritualPath.AHIMSA: "Your question demonstrates a compassionate and non-violent approach"
            }
            insights.append(path_insights.get(primary_activation.path, "Your spiritual inquiry is sincere and valuable"))
        
        # Multiple high activations suggest integration
        high_activations = [a for a in activations if a.relevance_score > 0.4]
        if len(high_activations) > 1:
            insights.append("Your question touches multiple spiritual paths, indicating an integrated approach to dharma")
        
        # Context-based insights
        life_stage = user_context.get('life_stage')
        if life_stage:
            stage_insights = {
                'student': "As a student, focus on building strong foundations in understanding and ethical practice",
                'householder': "Your householder stage offers rich opportunities for dharmic living and selfless service",
                'seeker': "Your seeking nature indicates readiness for deeper spiritual practices and realization",
                'renunciant': "Your renunciant path calls for complete dedication to liberation and wisdom"
            }
            insights.append(stage_insights.get(life_stage, "Your life stage supports your spiritual development"))
        
        return insights
    
    def recommend_next_inquiry(self, query: str, activations: List[PathActivation],
                             guidance: IntegratedGuidance) -> str:
        """Recommend next area of inquiry for user"""
        
        primary_path = guidance.primary_path
        
        next_inquiries = {
            SpiritualPath.JNANA: [
                "How can I apply this understanding in my daily life?",
                "What practices will deepen my spiritual knowledge?",
                "How do I distinguish between intellectual understanding and direct realization?"
            ],
            SpiritualPath.KARMA: [
                "How can I purify my motivation for action?",
                "What is my unique dharmic contribution to the world?",
                "How do I balance spiritual practice with worldly responsibilities?"
            ],
            SpiritualPath.BHAKTI: [
                "How can I deepen my devotional practice?",
                "What is the relationship between love and surrender?",
                "How do I maintain devotion in challenging times?"
            ],
            SpiritualPath.SEVA: [
                "How can I serve more effectively and selflessly?",
                "What forms of service align with my spiritual development?",
                "How do I maintain spiritual perspective while serving others?"
            ],
            SpiritualPath.DHARMA: [
                "How do I determine what is dharmic in complex situations?",
                "What is the relationship between personal and universal dharma?",
                "How can I live more righteously in daily life?"
            ],
            SpiritualPath.MOKSHA: [
                "What obstacles prevent me from deeper realization?",
                "How do I integrate spiritual insights with daily living?",
                "What is the next stage in my spiritual development?"
            ],
            SpiritualPath.YOGA: [
                "How can I deepen my spiritual practice?",
                "What practices are most suitable for my temperament?",
                "How do I maintain consistency in spiritual discipline?"
            ],
            SpiritualPath.AHIMSA: [
                "How can I live more compassionately?",
                "What is the deepest meaning of non-violence?",
                "How do I balance truth-telling with non-harm?"
            ]
        }
        
        recommendations = next_inquiries.get(primary_path, next_inquiries[SpiritualPath.JNANA])
        
        # Choose based on query sophistication
        if len(query.split()) > 10:  # More sophisticated query
            return recommendations[2]  # More advanced recommendation
        elif 'how' in query.lower():
            return recommendations[1]  # Practical recommendation
        else:
            return recommendations[0]  # Basic recommendation
    
    def create_fallback_guidance(self, query: str) -> IntegratedGuidance:
        """Create fallback guidance when no path responses available"""
        
        return IntegratedGuidance(
            primary_guidance=f"Your sincere inquiry '{query}' is honored in the tradition of dharmic wisdom. All authentic spiritual questions ultimately lead to self-understanding and the realization of our divine nature.",
            primary_path=SpiritualPath.DHARMA,
            secondary_perspectives={},
            synthesis="Begin with establishing righteousness in daily life, cultivate understanding through study and contemplation, express this understanding through compassionate action, and recognize that the ultimate goal is liberation while serving all beings.",
            practical_steps=["Establish daily spiritual practice", "Study authentic dharmic texts", "Serve others selflessly", "Live ethically and truthfully"],
            spiritual_principles=["Truth is eternal and one", "All beings are divine in essence", "Service to others is service to the divine", "Liberation is our highest potential"],
            recommended_practices=["Daily meditation", "Study of sacred texts", "Selfless service", "Ethical living"],
            dharmic_principles=["Uphold truth in thought, word, and deed", "Practice non-violence toward all beings", "Fulfill your duties selflessly", "Seek the highest spiritual realization"]
        )
    
    def create_fallback_response(self, query: str) -> SpiritualRouterResponse:
        """Create fallback response when routing fails"""
        
        fallback_guidance = self.create_fallback_guidance(query)
        
        return SpiritualRouterResponse(
            integrated_guidance=fallback_guidance,
            individual_responses={},
            routing_analysis=[],
            user_journey_insights=["Your sincere spiritual inquiry is honored and holds great value"],
            next_recommended_inquiry="What specific aspect of dharmic living would you like to explore further?"
        )
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get spiritual router status"""
        
        return {
            "router": "SpiritualRouter",
            "spiritual_paths": [path.value for path in SpiritualPath],
            "total_paths": len(SpiritualPath),
            "query_intents": [intent.value for intent in QueryIntent],
            "guidance_modes": [mode.value for mode in GuidanceMode],
            "status": "active",
            "wisdom_tradition": "Hindu Dharma",
            "approach": "Integrated spiritual guidance"
        }

# Global spiritual router instance
_spiritual_router = None

def get_spiritual_router() -> SpiritualRouter:
    """Get global spiritual router instance"""
    global _spiritual_router
    if _spiritual_router is None:
        _spiritual_router = SpiritualRouter()
    return _spiritual_router

# Export main classes
__all__ = [
    "SpiritualRouter",
    "get_spiritual_router", 
    "SpiritualRouterResponse",
    "IntegratedGuidance",
    "SpiritualPath",
    "QueryIntent",
    "GuidanceMode"
]
