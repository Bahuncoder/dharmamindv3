"""
🧘 Spiritual AI Modules Tests
============================

Comprehensive tests for DharmaMind's spiritual AI components:
- System Orchestrator
- Consciousness Core
- Dharma Engine  
- Knowledge Base
- Emotional Intelligence
- Spiritual Guidance Processing
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from app.chakra_modules.system_orchestrator import SystemOrchestrator
from app.chakra_modules.consciousness_core import ConsciousnessCore
from app.chakra_modules.dharma_engine import DharmaEngine
from app.chakra_modules.knowledge_base import KnowledgeBase
from app.chakra_modules.emotional_intelligence import EmotionalIntelligence


@pytest.mark.spiritual
class TestSystemOrchestrator:
    """Test the central system orchestrator."""
    
    @pytest.fixture
    async def orchestrator(self, redis_client):
        """Create system orchestrator instance."""
        orchestrator = SystemOrchestrator()
        await orchestrator.initialize()
        return orchestrator
    
    async def test_orchestrator_initialization(self, orchestrator):
        """Test system orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.state == "ready"
        assert hasattr(orchestrator, 'components')
        assert len(orchestrator.components) > 0
    
    async def test_process_spiritual_guidance_simple(self, orchestrator, spiritual_test_data):
        """Test basic spiritual guidance processing."""
        query = spiritual_test_data["simple_query"]
        
        response = await orchestrator.process_spiritual_guidance(
            user_query=query["message"],
            user_context=query["context"]
        )
        
        assert response is not None
        assert "guidance" in response
        assert "dharmic_validation" in response
        assert "confidence" in response
        
        # Validate response structure
        assert isinstance(response["guidance"], str)
        assert isinstance(response["dharmic_validation"], bool)
        assert 0.0 <= response["confidence"] <= 1.0
        
        # Validate content quality
        assert len(response["guidance"]) > 20  # Meaningful response
        assert any(word in response["guidance"].lower() for word in 
                  ["breath", "present", "calm", "peace", "mindful"])
    
    async def test_process_spiritual_guidance_deep(self, orchestrator, spiritual_test_data):
        """Test deep philosophical query processing.""" 
        query = spiritual_test_data["deep_query"]
        
        response = await orchestrator.process_spiritual_guidance(
            user_query=query["message"],
            user_context=query["context"]
        )
        
        assert response is not None
        assert response["confidence"] >= 0.7  # High confidence for philosophical topics
        
        # Check for philosophical depth indicators
        guidance = response["guidance"].lower()
        assert any(word in guidance for word in 
                  ["consciousness", "awareness", "observer", "witness", "being"])
    
    async def test_process_practice_recommendation(self, orchestrator, spiritual_test_data):
        """Test meditation practice recommendations."""
        query = spiritual_test_data["practice_query"]
        
        response = await orchestrator.process_spiritual_guidance(
            user_query=query["message"],
            user_context=query["context"]
        )
        
        assert response is not None
        assert "suggested_practices" in response
        assert len(response["suggested_practices"]) > 0
        
        # Validate practice suggestions are relevant
        practices = response["suggested_practices"]
        valid_practices = ["breathing_meditation", "mindfulness", "loving_kindness", 
                          "body_scan", "walking_meditation", "concentration"]
        assert any(practice in valid_practices for practice in practices)
    
    async def test_dharmic_scripture_query(self, orchestrator, spiritual_test_data):
        """Test dharmic scripture-based queries."""
        query = spiritual_test_data["dharmic_query"]
        
        response = await orchestrator.process_spiritual_guidance(
            user_query=query["message"],
            user_context=query["context"]
        )
        
        assert response is not None
        assert response["dharmic_validation"] is True
        assert response["confidence"] >= 0.8  # High confidence for scripture
        
        # Check for dharmic concepts
        guidance = response["guidance"].lower()
        assert any(concept in guidance for concept in 
                  ["dharma", "duty", "righteous", "karma", "gita"])
    
    async def test_consciousness_analysis(self, orchestrator):
        """Test consciousness state analysis."""
        user_input = {
            "message": "I feel scattered and unfocused today",
            "mood_indicators": ["scattered", "unfocused", "restless"],
            "context": {"mental_state": "agitated"}
        }
        
        analysis = await orchestrator.analyze_consciousness(user_input)
        
        assert analysis is not None
        assert "consciousness_level" in analysis
        assert "emotional_state" in analysis
        assert "mental_clarity" in analysis
        assert "spiritual_readiness" in analysis
        
        # Validate analysis makes sense for scattered state
        assert analysis["mental_clarity"] < 0.7  # Low clarity for scattered state
        assert analysis["consciousness_level"] in ["scattered", "agitated", "distracted"]
    
    async def test_orchestrator_error_handling(self, orchestrator):
        """Test orchestrator handles errors gracefully."""
        # Test with malformed input
        with patch.object(orchestrator, 'consciousness_core') as mock_core:
            mock_core.analyze.side_effect = Exception("Simulated error")
            
            response = await orchestrator.process_spiritual_guidance(
                user_query="Test query",
                user_context={}
            )
            
            # Should handle error gracefully
            assert response is not None
            assert "error" in response or response["confidence"] < 0.5
    
    async def test_orchestrator_performance(self, orchestrator, benchmark_timer):
        """Test orchestrator response time performance."""
        timer = benchmark_timer()
        
        timer.start()
        response = await orchestrator.process_spiritual_guidance(
            user_query="How should I approach meditation?",
            user_context={"experience": "beginner"}
        )
        execution_time = timer.stop()
        
        assert response is not None
        assert execution_time < 2.0  # Should respond within 2 seconds


@pytest.mark.spiritual  
class TestConsciousnessCore:
    """Test consciousness analysis core."""
    
    @pytest.fixture
    def consciousness_core(self):
        """Create consciousness core instance."""
        return ConsciousnessCore()
    
    async def test_analyze_emotional_state(self, consciousness_core):
        """Test emotional state analysis."""
        emotional_input = {
            "message": "I am feeling anxious and worried about the future",
            "tone_indicators": ["anxious", "worried", "fearful"]
        }
        
        analysis = await consciousness_core.analyze_emotional_state(emotional_input)
        
        assert analysis is not None
        assert "primary_emotion" in analysis
        assert "emotional_intensity" in analysis
        assert "recommended_response" in analysis
        
        # Should detect anxiety
        assert analysis["primary_emotion"] in ["anxiety", "worry", "fear"]
        assert analysis["emotional_intensity"] > 0.5  # Moderate to high intensity
        
        # Should recommend calming practices
        recommendation = analysis["recommended_response"].lower()
        assert any(word in recommendation for word in 
                  ["breath", "calm", "present", "ground", "center"])
    
    async def test_analyze_mental_clarity(self, consciousness_core):
        """Test mental clarity assessment."""
        clarity_input = {
            "message": "I feel very clear and focused today",
            "focus_indicators": ["clear", "focused", "alert", "present"]
        }
        
        analysis = await consciousness_core.analyze_mental_clarity(clarity_input)
        
        assert analysis is not None
        assert "clarity_level" in analysis
        assert "focus_quality" in analysis
        
        # Should detect high clarity
        assert analysis["clarity_level"] >= 0.7
        assert analysis["focus_quality"] == "high"
    
    async def test_spiritual_readiness_assessment(self, consciousness_core):
        """Test spiritual readiness evaluation."""
        readiness_input = {
            "user_context": {
                "meditation_experience": 100,  # Days
                "spiritual_level": "intermediate",
                "recent_practice": True,
                "emotional_stability": 0.8
            }
        }
        
        assessment = await consciousness_core.assess_spiritual_readiness(readiness_input)
        
        assert assessment is not None
        assert "readiness_score" in assessment
        assert "readiness_factors" in assessment
        assert "recommended_practices" in assessment
        
        # Should show good readiness for intermediate practitioner
        assert assessment["readiness_score"] >= 0.6
        assert len(assessment["recommended_practices"]) > 0


@pytest.mark.spiritual
class TestDharmaEngine:
    """Test dharmic validation and guidance engine."""
    
    @pytest.fixture
    def dharma_engine(self):
        """Create dharma engine instance."""
        return DharmaEngine()
    
    async def test_validate_dharmic_content(self, dharma_engine):
        """Test dharmic content validation."""
        # Test positive dharmic content
        positive_content = "Practice compassion towards all beings and cultivate inner peace"
        
        validation = await dharma_engine.validate_content(positive_content)
        
        assert validation is not None
        assert "is_dharmic" in validation
        assert "dharmic_score" in validation
        assert "dharmic_principles" in validation
        
        assert validation["is_dharmic"] is True
        assert validation["dharmic_score"] >= 0.7
        assert len(validation["dharmic_principles"]) > 0
    
    async def test_reject_harmful_content(self, dharma_engine):
        """Test rejection of harmful or non-dharmic content."""
        harmful_content = "Seek revenge against your enemies and hold onto anger"
        
        validation = await dharma_engine.validate_content(harmful_content)
        
        assert validation is not None
        assert validation["is_dharmic"] is False
        assert validation["dharmic_score"] < 0.5
        assert "warning" in validation
    
    async def test_scripture_integration(self, dharma_engine):
        """Test integration of scriptural wisdom."""
        query = "What does Buddhism teach about suffering?"
        
        response = await dharma_engine.get_scriptural_guidance(query)
        
        assert response is not None
        assert "teaching" in response
        assert "source" in response
        assert "application" in response
        
        # Should reference Four Noble Truths or similar
        teaching = response["teaching"].lower()
        assert any(concept in teaching for concept in 
                  ["suffering", "dukkha", "four noble truths", "attachment"])
    
    async def test_ethical_guidance(self, dharma_engine):
        """Test ethical decision-making guidance."""
        ethical_dilemma = {
            "situation": "Should I tell the truth if it might hurt someone?",
            "context": "personal_relationship",
            "stakeholders": ["self", "friend"]
        }
        
        guidance = await dharma_engine.provide_ethical_guidance(ethical_dilemma)
        
        assert guidance is not None
        assert "recommendation" in guidance
        assert "ethical_principle" in guidance
        assert "considerations" in guidance
        
        # Should emphasize truth and compassion
        recommendation = guidance["recommendation"].lower()
        assert any(concept in recommendation for concept in 
                  ["truth", "compassion", "kindness", "wisdom", "harm"])


@pytest.mark.spiritual
class TestKnowledgeBase:
    """Test spiritual knowledge base functionality."""
    
    @pytest.fixture
    def knowledge_base(self):
        """Create knowledge base instance."""
        return KnowledgeBase()
    
    async def test_search_spiritual_teachings(self, knowledge_base):
        """Test searching spiritual teachings."""
        search_query = "meditation techniques for beginners"
        
        results = await knowledge_base.search_teachings(search_query)
        
        assert results is not None
        assert len(results) > 0
        
        # Validate result structure
        for result in results:
            assert "title" in result
            assert "content" in result
            assert "relevance_score" in result
            assert "source" in result
            
            # Should be relevant to meditation
            content = result["content"].lower()
            assert any(word in content for word in 
                      ["meditat", "breath", "mindful", "concentrat"])
    
    async def test_get_practice_instructions(self, knowledge_base):
        """Test retrieving practice instructions."""
        practice = "loving_kindness_meditation"
        
        instructions = await knowledge_base.get_practice_instructions(practice)
        
        assert instructions is not None
        assert "steps" in instructions
        assert "duration" in instructions
        assert "benefits" in instructions
        assert "prerequisites" in instructions
        
        # Validate steps are comprehensive
        assert len(instructions["steps"]) >= 3
        assert all(isinstance(step, str) and len(step) > 10 
                  for step in instructions["steps"])
    
    async def test_conceptual_understanding(self, knowledge_base):
        """Test conceptual explanation generation."""
        concept = "impermanence"
        
        explanation = await knowledge_base.explain_concept(concept)
        
        assert explanation is not None
        assert "definition" in explanation
        assert "examples" in explanation
        assert "practical_application" in explanation
        
        # Should explain impermanence accurately
        definition = explanation["definition"].lower()
        assert any(word in definition for word in 
                  ["change", "temporary", "passing", "transient"])


@pytest.mark.spiritual
class TestEmotionalIntelligence:
    """Test emotional intelligence and response system."""
    
    @pytest.fixture
    def emotional_intelligence(self):
        """Create emotional intelligence instance."""
        return EmotionalIntelligence()
    
    async def test_emotion_detection(self, emotional_intelligence):
        """Test emotion detection from text."""
        emotional_inputs = [
            ("I am so happy and grateful today!", "joy"),
            ("I feel lost and don't know what to do", "confusion"),
            ("This situation makes me really angry", "anger"), 
            ("I'm worried about my future", "anxiety"),
            ("I feel peaceful and content", "peace")
        ]
        
        for text, expected_emotion in emotional_inputs:
            result = await emotional_intelligence.detect_emotions(text)
            
            assert result is not None
            assert "primary_emotion" in result
            assert "confidence" in result
            assert "emotional_context" in result
            
            # Should detect the primary emotion correctly
            detected = result["primary_emotion"].lower()
            assert expected_emotion.lower() in detected or detected in expected_emotion.lower()
    
    async def test_empathetic_response_generation(self, emotional_intelligence):
        """Test generation of empathetic responses."""
        emotional_context = {
            "primary_emotion": "sadness",
            "intensity": 0.8,
            "triggers": ["loss", "grief"],
            "user_message": "I lost someone close to me"
        }
        
        response = await emotional_intelligence.generate_empathetic_response(emotional_context)
        
        assert response is not None
        assert "response_text" in response
        assert "therapeutic_approach" in response
        assert "suggested_practices" in response
        
        # Should be compassionate and appropriate for grief
        response_text = response["response_text"].lower()
        assert any(word in response_text for word in 
                  ["sorry", "grief", "loss", "time", "heal", "support"])
        
        # Should suggest appropriate practices
        practices = response["suggested_practices"]
        assert any(practice in practices for practice in 
                  ["loving_kindness", "grief_meditation", "gentle_breathing"])
    
    async def test_emotional_regulation_guidance(self, emotional_intelligence):
        """Test emotional regulation guidance."""
        regulation_request = {
            "current_emotion": "anger",
            "intensity": 0.9,
            "situation": "work_conflict",
            "user_goal": "calm_down"
        }
        
        guidance = await emotional_intelligence.provide_regulation_guidance(regulation_request)
        
        assert guidance is not None
        assert "immediate_techniques" in guidance
        assert "long_term_practices" in guidance
        assert "mindfulness_approaches" in guidance
        
        # Should provide immediate calming techniques
        immediate = guidance["immediate_techniques"]
        assert len(immediate) >= 2
        assert any("breath" in technique.lower() for technique in immediate)


@pytest.mark.integration
@pytest.mark.spiritual
class TestSpiritualAIIntegration:
    """Integration tests for spiritual AI component interactions."""
    
    async def test_full_guidance_pipeline(self, mock_system_orchestrator, spiritual_test_data):
        """Test complete spiritual guidance pipeline."""
        # Test the full pipeline from user input to guidance output
        user_query = spiritual_test_data["simple_query"]
        
        # Mock the orchestrator to test integration
        response = await mock_system_orchestrator.process_spiritual_guidance(
            user_query["message"], user_query["context"]
        )
        
        assert response is not None
        assert all(key in response for key in 
                  ["guidance", "dharmic_validation", "confidence", 
                   "spiritual_context", "suggested_practices"])
    
    async def test_component_communication(self, orchestrator):
        """Test communication between spiritual AI components."""
        # This would test that components properly communicate
        # and share context with each other
        
        test_context = {
            "user_id": "test_user_123",
            "session_id": "session_456",
            "emotional_state": "calm",
            "spiritual_level": "intermediate"
        }
        
        # Process through multiple components
        consciousness_analysis = await orchestrator.consciousness_core.analyze(test_context)
        dharmic_validation = await orchestrator.dharma_engine.validate_context(test_context)
        
        # Components should maintain context consistency
        assert consciousness_analysis is not None
        assert dharmic_validation is not None
    
    async def test_response_coherence(self, orchestrator, spiritual_test_data):
        """Test that responses are coherent across multiple queries."""
        queries = [
            spiritual_test_data["simple_query"],
            spiritual_test_data["practice_query"],
            spiritual_test_data["deep_query"]
        ]
        
        responses = []
        for query in queries:
            response = await orchestrator.process_spiritual_guidance(
                query["message"], query["context"]
            )
            responses.append(response)
        
        # All responses should be coherent and dharmic
        for response in responses:
            assert response["dharmic_validation"] is True
            assert response["confidence"] >= 0.6
            assert len(response["guidance"]) > 20
    
    @pytest.mark.slow
    async def test_performance_under_load(self, orchestrator):
        """Test performance with multiple concurrent requests."""
        # Simulate multiple users asking for guidance simultaneously
        concurrent_queries = [
            ("How do I deal with stress?", {"mood": "stressed"}),
            ("What is mindfulness?", {"experience": "beginner"}),
            ("How to cultivate compassion?", {"practice": "loving_kindness"}),
            ("What is the nature of mind?", {"depth": "philosophical"}),
            ("How to meditate properly?", {"time": "morning"})
        ] * 10  # 50 total queries
        
        # Execute all queries concurrently
        tasks = [
            orchestrator.process_spiritual_guidance(query, context)
            for query, context in concurrent_queries
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        # Validate results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= len(concurrent_queries) * 0.9  # 90% success rate
        
        # Performance should be reasonable
        total_time = end_time - start_time
        avg_time_per_query = total_time / len(concurrent_queries)
        assert avg_time_per_query < 3.0  # Average under 3 seconds per query
        
        # All successful responses should be valid
        for result in successful_results:
            assert result["confidence"] >= 0.5
            assert result["dharmic_validation"] is True
