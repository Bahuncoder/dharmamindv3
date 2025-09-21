"""
🧪 Unit Tests for Revolutionary Emotional Intelligence
====================================================

Comprehensive unit tests for the enhanced emotional intelligence system:
- Emotional state detection accuracy
- Cultural awareness and adaptation
- Sanskrit concept integration
- Response generation quality
- Performance benchmarks
- Edge case handling

These tests ensure the spiritual guidance AI provides accurate, culturally
sensitive, and enlightening emotional support.
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Import test framework
from tests import (
    test_framework, DharmaMindTestFramework, 
    spiritual_test_data, user_test_data,
    test_emotional_cache
)

class TestRevolutionaryEmotionalIntelligence:
    """Unit tests for revolutionary emotional intelligence system"""
    
    @pytest.mark.asyncio
    async def test_basic_emotional_detection(self, test_emotional_cache, spiritual_test_data):
        """Test basic emotional state detection"""
        async with test_framework.test_context("basic_emotional_detection", "emotional_ai") as test_result:
            # Test with various spiritual queries
            for query_data in spiritual_test_data['queries'][:3]:
                text = query_data['text']
                expected_emotions = query_data['expected_emotions']
                
                # Analyze emotional state
                start_time = time.time()
                analysis = await test_emotional_cache.analyze_emotional_state(text, "test_user")
                duration = time.time() - start_time
                
                # Performance assertion
                test_framework.assert_performance_threshold('emotional_analysis', duration)
                
                # Accuracy assertion
                test_framework.assert_emotional_analysis_accuracy(analysis, expected_emotions)
                
                test_result.details = {
                    'queries_tested': len(spiritual_test_data['queries'][:3]),
                    'avg_analysis_time': duration,
                    'sample_analysis': analysis
                }
    
    @pytest.mark.asyncio
    async def test_emotional_analysis_accuracy(self, test_emotional_cache):
        """Test emotional analysis accuracy with specific scenarios"""
        async with test_framework.test_context("emotional_analysis_accuracy", "emotional_ai") as test_result:
            test_cases = [
                {
                    'text': "I am feeling deep peace during my meditation practice",
                    'expected_emotions': ['peace', 'tranquility', 'meditation'],
                    'expected_confidence': 0.7
                },
                {
                    'text': "I struggle with anger and resentment, seeking spiritual guidance",
                    'expected_emotions': ['anger', 'struggle', 'seeking'],
                    'expected_confidence': 0.6
                },
                {
                    'text': "I feel overwhelming gratitude for this spiritual journey",
                    'expected_emotions': ['gratitude', 'joy', 'spiritual_growth'],
                    'expected_confidence': 0.7
                },
                {
                    'text': "What is the nature of consciousness and ultimate reality?",
                    'expected_emotions': ['wisdom', 'seeking', 'curiosity'],
                    'expected_confidence': 0.6
                }
            ]
            
            successful_analyses = 0
            total_confidence = 0
            
            for test_case in test_cases:
                analysis = await test_emotional_cache.analyze_emotional_state(
                    test_case['text'], "accuracy_test_user"
                )
                
                # Check basic structure
                assert 'emotional_states' in analysis
                assert 'confidence' in analysis
                assert isinstance(analysis['emotional_states'], list)
                assert isinstance(analysis['confidence'], (int, float))
                
                # Check confidence threshold
                confidence = analysis['confidence']
                total_confidence += confidence
                
                if confidence >= test_case['expected_confidence']:
                    successful_analyses += 1
                
                # Check for expected emotional overlap
                detected_emotions = analysis['emotional_states']
                expected_emotions = test_case['expected_emotions']
                overlap = set(detected_emotions) & set(expected_emotions)
                
                # At least one expected emotion should be detected
                assert len(overlap) > 0, f"No overlap for: {test_case['text']}"
            
            avg_confidence = total_confidence / len(test_cases)
            accuracy_rate = successful_analyses / len(test_cases)
            
            # Overall accuracy assertions
            assert avg_confidence >= 0.6, f"Low average confidence: {avg_confidence}"
            assert accuracy_rate >= 0.75, f"Low accuracy rate: {accuracy_rate}"
            
            test_result.details = {
                'test_cases': len(test_cases),
                'successful_analyses': successful_analyses,
                'avg_confidence': avg_confidence,
                'accuracy_rate': accuracy_rate
            }
    
    @pytest.mark.asyncio
    async def test_cultural_awareness(self, test_emotional_cache):
        """Test cultural awareness and adaptation in emotional analysis"""
        async with test_framework.test_context("cultural_awareness", "emotional_ai") as test_result:
            cultural_contexts = [
                {
                    'context': 'indian_traditional',
                    'text': "I seek moksha through dedicated sadhana",
                    'expected_concepts': ['moksha', 'sadhana', 'liberation']
                },
                {
                    'context': 'western_contemporary',
                    'text': "I want to improve my mindfulness and well-being",
                    'expected_concepts': ['mindfulness', 'well-being', 'self-improvement']
                },
                {
                    'context': 'buddhist_influenced',
                    'text': "I practice meditation to reduce suffering",
                    'expected_concepts': ['meditation', 'suffering', 'compassion']
                }
            ]
            
            cultural_adaptations = 0
            
            for context_data in cultural_contexts:
                # Create user context
                user_context = {
                    'cultural_context': context_data['context'],
                    'spiritual_tradition': context_data['context'].split('_')[0]
                }
                
                analysis = await test_emotional_cache.analyze_emotional_state(
                    context_data['text'], 
                    f"cultural_test_{context_data['context']}", 
                    user_context
                )
                
                # Check for cultural adaptation
                cultural_context = analysis.get('cultural_context', {})
                if cultural_context:
                    cultural_adaptations += 1
                
                # Check for appropriate concept references
                recommended_practices = analysis.get('recommended_practices', [])
                assert len(recommended_practices) > 0, f"No practices recommended for {context_data['context']}"
            
            adaptation_rate = cultural_adaptations / len(cultural_contexts)
            assert adaptation_rate >= 0.5, f"Low cultural adaptation rate: {adaptation_rate}"
            
            test_result.details = {
                'contexts_tested': len(cultural_contexts),
                'cultural_adaptations': cultural_adaptations,
                'adaptation_rate': adaptation_rate
            }
    
    @pytest.mark.asyncio
    async def test_sanskrit_concept_integration(self, test_emotional_cache):
        """Test integration of Sanskrit spiritual concepts"""
        async with test_framework.test_context("sanskrit_concept_integration", "emotional_ai") as test_result:
            sanskrit_test_cases = [
                {
                    'text': "I seek ultimate peace and bliss",
                    'expected_concepts': ['shanti', 'ananda', 'peace']
                },
                {
                    'text': "I want to understand divine love and devotion",
                    'expected_concepts': ['prema', 'bhakti', 'love']
                },
                {
                    'text': "I am curious about pure knowledge and wisdom",
                    'expected_concepts': ['jnana', 'vidya', 'wisdom']
                }
            ]
            
            sanskrit_integrations = 0
            
            for test_case in sanskrit_test_cases:
                analysis = await test_emotional_cache.analyze_emotional_state(
                    test_case['text'], "sanskrit_test_user"
                )
                
                # Check for Sanskrit concepts
                sanskrit_concepts = analysis.get('sanskrit_concepts', [])
                if len(sanskrit_concepts) > 0:
                    sanskrit_integrations += 1
                
                # Verify analysis structure includes spiritual guidance
                assert 'guidance_priority' in analysis
                assert 'recommended_practices' in analysis
            
            integration_rate = sanskrit_integrations / len(sanskrit_test_cases)
            assert integration_rate >= 0.6, f"Low Sanskrit integration rate: {integration_rate}"
            
            test_result.details = {
                'test_cases': len(sanskrit_test_cases),
                'sanskrit_integrations': sanskrit_integrations,
                'integration_rate': integration_rate
            }
    
    @pytest.mark.asyncio
    async def test_empathetic_response_generation(self, test_emotional_cache):
        """Test empathetic response generation quality"""
        async with test_framework.test_context("empathetic_response_generation", "emotional_ai") as test_result:
            emotional_profiles = [
                {
                    'emotional_states': ['seeking', 'confusion'],
                    'confidence': 0.8,
                    'user_type': 'beginner'
                },
                {
                    'emotional_states': ['peace', 'gratitude'],
                    'confidence': 0.9,
                    'user_type': 'intermediate'
                },
                {
                    'emotional_states': ['wisdom', 'contemplation'],
                    'confidence': 0.85,
                    'user_type': 'advanced'
                }
            ]
            
            successful_responses = 0
            total_personalization = 0
            
            for profile in emotional_profiles:
                response = await test_emotional_cache.generate_empathetic_response(
                    profile, f"response_test_{profile['user_type']}"
                )
                
                # Check response structure
                assert 'message' in response
                assert 'tone' in response
                assert 'practices' in response
                assert 'confidence' in response
                
                # Check response quality
                message = response['message']
                assert len(message) > 10, "Response message too short"
                assert 'I understand' in message or 'I sense' in message, "Response lacks empathy"
                
                # Check personalization
                personalization_level = response.get('personalization_level', 0)
                total_personalization += personalization_level
                
                if response['confidence'] >= 0.7:
                    successful_responses += 1
            
            response_quality = successful_responses / len(emotional_profiles)
            avg_personalization = total_personalization / len(emotional_profiles)
            
            assert response_quality >= 0.8, f"Low response quality: {response_quality}"
            assert avg_personalization >= 0.3, f"Low personalization: {avg_personalization}"
            
            test_result.details = {
                'profiles_tested': len(emotional_profiles),
                'successful_responses': successful_responses,
                'response_quality': response_quality,
                'avg_personalization': avg_personalization
            }
    
    @pytest.mark.asyncio
    async def test_user_pattern_recognition(self, test_emotional_cache):
        """Test user emotional pattern recognition and learning"""
        async with test_framework.test_context("user_pattern_recognition", "emotional_ai") as test_result:
            user_id = "pattern_test_user"
            
            # Simulate user interaction pattern
            interaction_texts = [
                "I am seeking inner peace",
                "I want to understand meditation better",
                "I feel calm during my practice",
                "I am curious about mindfulness",
                "I seek tranquility in daily life"
            ]
            
            # Analyze multiple interactions to build pattern
            for text in interaction_texts:
                await test_emotional_cache.analyze_emotional_state(text, user_id)
                await asyncio.sleep(0.01)  # Small delay to simulate real interactions
            
            # Check if user pattern was created
            user_pattern = await test_emotional_cache._get_user_pattern(user_id)
            
            assert user_pattern is not None, "User pattern not created"
            assert len(user_pattern.dominant_emotions) > 0, "No dominant emotions identified"
            assert user_pattern.pattern_confidence > 0, "No pattern confidence"
            assert len(user_pattern.interaction_history) > 0, "No interaction history"
            
            # Test pattern-based personalization
            final_analysis = await test_emotional_cache.analyze_emotional_state(
                "I continue my spiritual journey", user_id
            )
            
            # Should have higher confidence due to established pattern
            assert final_analysis['confidence'] >= 0.6, "Pattern learning not improving confidence"
            
            test_result.details = {
                'interactions_processed': len(interaction_texts),
                'dominant_emotions': user_pattern.dominant_emotions,
                'pattern_confidence': user_pattern.pattern_confidence,
                'interaction_history_length': len(user_pattern.interaction_history)
            }
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, test_emotional_cache):
        """Test emotional intelligence performance benchmarks"""
        async with test_framework.test_context("performance_benchmarks", "emotional_ai") as test_result:
            # Performance test data
            test_texts = [
                "I seek spiritual guidance",
                "I am grateful for this wisdom",
                "I struggle with inner conflict",
                "I want to find peace",
                "I am curious about enlightenment"
            ]
            
            # Benchmark response times
            response_times = []
            cache_hits = 0
            
            for i, text in enumerate(test_texts):
                # First analysis (should be slower)
                start_time = time.time()
                analysis1 = await test_emotional_cache.analyze_emotional_state(
                    text, f"perf_user_{i}"
                )
                first_time = time.time() - start_time
                
                # Second analysis (should hit cache)
                start_time = time.time()
                analysis2 = await test_emotional_cache.analyze_emotional_state(
                    text, f"perf_user_{i}"
                )
                second_time = time.time() - start_time
                
                response_times.extend([first_time, second_time])
                
                # Check if second analysis was faster (cache hit)
                if second_time < first_time:
                    cache_hits += 1
                
                # Performance assertions
                test_framework.assert_performance_threshold('emotional_analysis', first_time)
                test_framework.assert_performance_threshold('emotional_analysis', second_time)
            
            avg_response_time = sum(response_times) / len(response_times)
            cache_hit_ratio = cache_hits / len(test_texts)
            
            # Performance targets
            assert avg_response_time <= 0.1, f"Average response time too high: {avg_response_time}"
            assert cache_hit_ratio >= 0.6, f"Low cache hit ratio: {cache_hit_ratio}"
            
            test_result.details = {
                'texts_tested': len(test_texts),
                'avg_response_time': avg_response_time,
                'cache_hit_ratio': cache_hit_ratio,
                'total_analyses': len(response_times)
            }
    
    @pytest.mark.asyncio
    async def test_edge_cases_and_error_handling(self, test_emotional_cache):
        """Test edge cases and error handling"""
        async with test_framework.test_context("edge_cases_error_handling", "emotional_ai") as test_result:
            edge_cases = [
                "",  # Empty string
                "   ",  # Whitespace only
                "a",  # Single character
                "?" * 1000,  # Very long string
                "Hello 123 !@# $%^",  # Mixed content
                "Namaskar sat-chit-ananda",  # Sanskrit terms
                None,  # None value (will be converted to string)
            ]
            
            successful_handles = 0
            
            for i, test_input in enumerate(edge_cases):
                try:
                    # Convert None to empty string
                    text_input = str(test_input) if test_input is not None else ""
                    
                    analysis = await test_emotional_cache.analyze_emotional_state(
                        text_input, f"edge_case_user_{i}"
                    )
                    
                    # Should always return a valid response structure
                    assert isinstance(analysis, dict), "Invalid response type"
                    assert 'emotional_states' in analysis, "Missing emotional_states"
                    assert 'confidence' in analysis, "Missing confidence"
                    
                    # Confidence should be reasonable for edge cases
                    confidence = analysis['confidence']
                    assert 0 <= confidence <= 1, f"Invalid confidence range: {confidence}"
                    
                    successful_handles += 1
                    
                except Exception as e:
                    # Log the error but don't fail the test - edge cases should be handled gracefully
                    print(f"Edge case error for '{test_input}': {e}")
            
            # Most edge cases should be handled successfully
            handle_rate = successful_handles / len(edge_cases)
            assert handle_rate >= 0.7, f"Low edge case handling rate: {handle_rate}"
            
            test_result.details = {
                'edge_cases_tested': len(edge_cases),
                'successful_handles': successful_handles,
                'handle_rate': handle_rate
            }
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, test_emotional_cache):
        """Test concurrent emotional analysis operations"""
        async with test_framework.test_context("concurrent_analysis", "emotional_ai") as test_result:
            # Create concurrent analysis tasks
            texts = [
                "I seek wisdom through meditation",
                "I feel grateful for this journey", 
                "I want to understand consciousness",
                "I am learning about compassion",
                "I practice mindfulness daily"
            ]
            
            # Run concurrent analyses
            start_time = time.time()
            tasks = []
            
            for i, text in enumerate(texts):
                task = test_emotional_cache.analyze_emotional_state(text, f"concurrent_user_{i}")
                tasks.append(task)
            
            # Wait for all analyses to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Check results
            successful_analyses = 0
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Task {i}: {result}")
                else:
                    if isinstance(result, dict) and 'emotional_states' in result:
                        successful_analyses += 1
                    else:
                        errors.append(f"Task {i}: Invalid result format")
            
            success_rate = successful_analyses / len(texts)
            avg_time_per_analysis = total_time / len(texts)
            
            # Concurrent operations should not significantly degrade performance
            assert success_rate >= 0.9, f"Low concurrent success rate: {success_rate}"
            assert avg_time_per_analysis <= 0.2, f"Concurrent operations too slow: {avg_time_per_analysis}"
            assert len(errors) == 0, f"Concurrent analysis errors: {errors}"
            
            test_result.details = {
                'concurrent_analyses': len(texts),
                'successful_analyses': successful_analyses,
                'success_rate': success_rate,
                'total_time': total_time,
                'avg_time_per_analysis': avg_time_per_analysis,
                'errors': errors
            }

# Additional test utilities for emotional intelligence
class TestEmotionalIntelligenceUtils:
    """Utility functions for emotional intelligence testing"""
    
    @staticmethod
    def validate_emotional_response_structure(response: Dict[str, Any]) -> bool:
        """Validate emotional response structure"""
        required_fields = [
            'emotional_states', 'confidence', 'complexity_score',
            'recommended_practices', 'analysis_timestamp'
        ]
        
        for field in required_fields:
            if field not in response:
                return False
        
        # Type validations
        if not isinstance(response['emotional_states'], list):
            return False
        if not isinstance(response['confidence'], (int, float)):
            return False
        if not isinstance(response['recommended_practices'], list):
            return False
        
        return True
    
    @staticmethod
    def calculate_emotional_overlap(detected: List[str], expected: List[str]) -> float:
        """Calculate overlap between detected and expected emotions"""
        if not expected:
            return 1.0
        
        overlap = set(detected) & set(expected)
        return len(overlap) / len(expected)
    
    @staticmethod
    def assess_spiritual_guidance_quality(response: Dict[str, Any]) -> Dict[str, float]:
        """Assess spiritual guidance response quality"""
        scores = {
            'completeness': 0.0,
            'relevance': 0.0,
            'personalization': 0.0,
            'spiritual_depth': 0.0
        }
        
        # Completeness score
        required_components = ['emotional_analysis', 'practices', 'concepts']
        present_components = sum(1 for comp in required_components if comp in response)
        scores['completeness'] = present_components / len(required_components)
        
        # Relevance score (based on confidence)
        scores['relevance'] = response.get('confidence', 0.0)
        
        # Personalization score
        scores['personalization'] = response.get('personalization_level', 0.0)
        
        # Spiritual depth score (based on Sanskrit concepts and practices)
        sanskrit_concepts = response.get('sanskrit_concepts', [])
        practices = response.get('recommended_practices', [])
        depth_indicators = len(sanskrit_concepts) + len(practices)
        scores['spiritual_depth'] = min(1.0, depth_indicators / 5)  # Normalize to 0-1
        
        return scores

# Export test utilities
__all__ = [
    'TestRevolutionaryEmotionalIntelligence',
    'TestEmotionalIntelligenceUtils'
]