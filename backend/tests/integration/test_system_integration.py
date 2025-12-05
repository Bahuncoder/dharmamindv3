"""
🧪 Integration Tests for DharmaMind Complete System
================================================

Comprehensive integration tests for the complete DharmaMind platform:
- End-to-end spiritual guidance workflows
- Authentication and security integration
- Cache system integration across modules
- Performance optimization validation
- User experience flow testing
- Multi-module interaction testing

These tests validate that all components work harmoniously to provide
seamless spiritual guidance experiences.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Import test framework
from tests import (
    test_framework, DharmaMindTestFramework,
    spiritual_test_data, user_test_data,
    test_cache_manager, test_knowledge_cache, test_emotional_cache
)

class TestDharmaMindSystemIntegration:
    """Integration tests for complete DharmaMind system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_spiritual_guidance(self, test_knowledge_cache, test_emotional_cache, spiritual_test_data):
        """Test complete end-to-end spiritual guidance workflow"""
        async with test_framework.test_context("end_to_end_spiritual_guidance", "integration") as test_result:
            # Test complete guidance workflow for each query
            successful_workflows = 0
            workflow_metrics = []
            
            for query_data in spiritual_test_data['queries']:
                workflow_start = time.time()
                
                query = query_data['text']
                expected_emotions = query_data['expected_emotions']
                expected_concepts = query_data['expected_concepts']
                user_id = f"integration_user_{hash(query) % 1000}"
                
                # Step 1: Emotional Analysis
                emotional_start = time.time()
                emotional_analysis = await test_emotional_cache.analyze_emotional_state(
                    query, user_id
                )
                emotional_time = time.time() - emotional_start
                
                # Step 2: Knowledge Retrieval
                knowledge_start = time.time()
                knowledge_results = await test_knowledge_cache.search_spiritual_knowledge(
                    query, user_id, emotional_analysis.get('emotional_states', [])
                )
                knowledge_time = time.time() - knowledge_start
                
                # Step 3: Response Generation
                response_start = time.time()
                empathetic_response = await test_emotional_cache.generate_empathetic_response(
                    emotional_analysis, user_id
                )
                response_time = time.time() - response_start
                
                # Step 4: Integration Validation
                workflow_time = time.time() - workflow_start
                
                # Validate each component
                assert emotional_analysis is not None, f"Emotional analysis failed for: {query}"
                assert knowledge_results is not None, f"Knowledge search failed for: {query}"
                assert empathetic_response is not None, f"Response generation failed for: {query}"
                
                # Validate integration quality
                test_framework.assert_emotional_analysis_accuracy(emotional_analysis, expected_emotions)
                test_framework.assert_knowledge_retrieval_accuracy(knowledge_results, expected_concepts)
                
                # Performance validation
                test_framework.assert_performance_threshold('emotional_analysis', emotional_time)
                test_framework.assert_performance_threshold('knowledge_search', knowledge_time)
                test_framework.assert_performance_threshold('spiritual_guidance', workflow_time)
                
                workflow_metrics.append({
                    'query': query[:50],
                    'total_time': workflow_time,
                    'emotional_time': emotional_time,
                    'knowledge_time': knowledge_time,
                    'response_time': response_time,
                    'emotional_confidence': emotional_analysis.get('confidence', 0),
                    'concepts_found': len(knowledge_results.get('concepts', [])),
                    'practices_recommended': len(knowledge_results.get('recommended_practices', []))
                })
                
                successful_workflows += 1
            
            # Calculate overall metrics
            workflow_success_rate = successful_workflows / len(spiritual_test_data['queries'])
            avg_workflow_time = sum(m['total_time'] for m in workflow_metrics) / len(workflow_metrics)
            avg_emotional_confidence = sum(m['emotional_confidence'] for m in workflow_metrics) / len(workflow_metrics)
            avg_concepts_found = sum(m['concepts_found'] for m in workflow_metrics) / len(workflow_metrics)
            
            # Integration success criteria
            assert workflow_success_rate >= 0.9, f"Low workflow success rate: {workflow_success_rate}"
            assert avg_workflow_time <= 0.5, f"Workflows too slow: {avg_workflow_time}s"
            assert avg_emotional_confidence >= 0.6, f"Low emotional confidence: {avg_emotional_confidence}"
            assert avg_concepts_found >= 2, f"Too few concepts found: {avg_concepts_found}"
            
            test_result.details = {
                'workflows_tested': len(spiritual_test_data['queries']),
                'successful_workflows': successful_workflows,
                'workflow_success_rate': workflow_success_rate,
                'avg_workflow_time': avg_workflow_time,
                'avg_emotional_confidence': avg_emotional_confidence,
                'avg_concepts_found': avg_concepts_found,
                'workflow_metrics': workflow_metrics
            }
    
    @pytest.mark.asyncio
    async def test_cache_integration_across_modules(self, test_cache_manager, test_knowledge_cache, test_emotional_cache):
        """Test cache integration across all modules"""
        async with test_framework.test_context("cache_integration_across_modules", "integration") as test_result:
            from app.services.cache_service import CacheCategory
            
            # Test data for different cache categories
            test_data = {
                CacheCategory.KNOWLEDGE_BASE: {
                    'key': 'integration_knowledge_test',
                    'data': {'concept': 'meditation', 'definition': 'Focused awareness practice'}
                },
                CacheCategory.EMOTIONAL_ANALYSIS: {
                    'key': 'integration_emotional_test',
                    'data': {'emotions': ['peace', 'seeking'], 'confidence': 0.85}
                },
                CacheCategory.USER_PREFERENCES: {
                    'key': 'integration_user_test',
                    'data': {'tradition': 'vedantic', 'experience': 'intermediate'}
                },
                CacheCategory.SPIRITUAL_GUIDANCE: {
                    'key': 'integration_guidance_test',
                    'data': {'guidance': 'Practice mindfulness daily', 'practices': ['meditation']}
                }
            }
            
            # Test cross-module cache operations
            cache_operations_successful = 0
            
            for category, test_info in test_data.items():
                try:
                    # Set data in cache
                    await test_cache_manager.set(category, test_info['key'], test_info['data'])
                    
                    # Retrieve data from cache
                    retrieved_data = await test_cache_manager.get(category, test_info['key'])
                    
                    # Validate data integrity
                    assert retrieved_data is not None, f"Failed to retrieve from {category}"
                    assert retrieved_data == test_info['data'], f"Data corruption in {category}"
                    
                    cache_operations_successful += 1
                    
                except Exception as e:
                    print(f"Cache operation failed for {category}: {e}")
            
            # Test cache statistics across modules
            cache_stats = await test_cache_manager.get_cache_statistics()
            
            # Validate cache statistics
            assert isinstance(cache_stats, dict), "Invalid cache statistics format"
            
            # Test cache isolation between categories
            isolation_test_key = "isolation_test"
            
            await test_cache_manager.set(CacheCategory.KNOWLEDGE_BASE, isolation_test_key, "knowledge_data")
            await test_cache_manager.set(CacheCategory.EMOTIONAL_ANALYSIS, isolation_test_key, "emotional_data")
            
            knowledge_data = await test_cache_manager.get(CacheCategory.KNOWLEDGE_BASE, isolation_test_key)
            emotional_data = await test_cache_manager.get(CacheCategory.EMOTIONAL_ANALYSIS, isolation_test_key)
            
            assert knowledge_data == "knowledge_data", "Cache isolation failed for knowledge"
            assert emotional_data == "emotional_data", "Cache isolation failed for emotional"
            
            operation_success_rate = cache_operations_successful / len(test_data)
            
            assert operation_success_rate >= 0.9, f"Low cache operation success rate: {operation_success_rate}"
            
            test_result.details = {
                'categories_tested': len(test_data),
                'successful_operations': cache_operations_successful,
                'operation_success_rate': operation_success_rate,
                'cache_isolation_verified': True,
                'cache_statistics': cache_stats
            }
    
    @pytest.mark.asyncio
    async def test_user_session_workflow(self, test_emotional_cache, test_knowledge_cache, user_test_data):
        """Test complete user session workflow with multiple interactions"""
        async with test_framework.test_context("user_session_workflow", "integration") as test_result:
            # Simulate a complete user session
            user_profile = user_test_data['test_users'][0]
            user_id = user_profile['user_id']
            
            # Session interaction sequence
            session_interactions = [
                {
                    'query': "I am new to meditation and seeking guidance",
                    'expected_progression': 'introduction'
                },
                {
                    'query': "I tried meditation but my mind wanders constantly",
                    'expected_progression': 'challenge_addressing'
                },
                {
                    'query': "I felt some peace during today's practice",
                    'expected_progression': 'progress_acknowledgment'
                },
                {
                    'query': "What deeper practices can I explore now?",
                    'expected_progression': 'advancement_guidance'
                },
                {
                    'query': "I want to understand the philosophy behind meditation",
                    'expected_progression': 'deeper_learning'
                }
            ]
            
            session_metrics = []
            user_pattern_evolution = []
            
            for i, interaction in enumerate(session_interactions):
                interaction_start = time.time()
                
                # Analyze emotional state with user context
                emotional_analysis = await test_emotional_cache.analyze_emotional_state(
                    interaction['query'], user_id, user_profile['profile']
                )
                
                # Search knowledge with emotional context
                knowledge_results = await test_knowledge_cache.search_spiritual_knowledge(
                    interaction['query'], 
                    user_id, 
                    emotional_analysis.get('emotional_states', [])
                )
                
                # Generate personalized response
                response = await test_emotional_cache.generate_empathetic_response(
                    emotional_analysis, user_id
                )
                
                interaction_time = time.time() - interaction_start
                
                # Check for session continuity and personalization improvement
                personalization_level = response.get('personalization_level', 0)
                
                # Track user pattern evolution
                user_pattern = await test_emotional_cache._get_user_pattern(user_id)
                if user_pattern:
                    user_pattern_evolution.append({
                        'interaction': i + 1,
                        'dominant_emotions': user_pattern.dominant_emotions[:3],
                        'confidence': user_pattern.pattern_confidence,
                        'history_length': len(user_pattern.interaction_history)
                    })
                
                session_metrics.append({
                    'interaction': i + 1,
                    'query': interaction['query'][:50],
                    'response_time': interaction_time,
                    'emotional_confidence': emotional_analysis.get('confidence', 0),
                    'concepts_provided': len(knowledge_results.get('concepts', [])),
                    'personalization_level': personalization_level,
                    'progression_stage': interaction['expected_progression']
                })
                
                # Validate session progression
                assert emotional_analysis['confidence'] >= 0.5, f"Low confidence in interaction {i+1}"
                assert len(knowledge_results['concepts']) > 0, f"No concepts in interaction {i+1}"
                assert personalization_level >= 0.2, f"Low personalization in interaction {i+1}"
            
            # Analyze session progression
            personalization_improvement = (
                session_metrics[-1]['personalization_level'] - 
                session_metrics[0]['personalization_level']
            )
            
            avg_response_time = sum(m['response_time'] for m in session_metrics) / len(session_metrics)
            pattern_confidence_growth = (
                user_pattern_evolution[-1]['confidence'] - user_pattern_evolution[0]['confidence']
                if len(user_pattern_evolution) >= 2 else 0
            )
            
            # Session quality assertions
            assert personalization_improvement >= 0, "No personalization improvement over session"
            assert avg_response_time <= 0.3, f"Session responses too slow: {avg_response_time}s"
            assert pattern_confidence_growth >= 0, "User pattern confidence not improving"
            
            test_result.details = {
                'interactions_completed': len(session_interactions),
                'personalization_improvement': personalization_improvement,
                'avg_response_time': avg_response_time,
                'pattern_confidence_growth': pattern_confidence_growth,
                'session_metrics': session_metrics,
                'user_pattern_evolution': user_pattern_evolution
            }
    
    @pytest.mark.asyncio
    async def test_concurrent_user_handling(self, test_emotional_cache, test_knowledge_cache):
        """Test system handling of concurrent users"""
        async with test_framework.test_context("concurrent_user_handling", "integration") as test_result:
            # Simulate multiple concurrent users
            concurrent_users = 10
            queries_per_user = 3
            
            async def simulate_user_session(user_id: int):
                """Simulate a single user session"""
                user_queries = [
                    f"User {user_id}: How can I find inner peace?",
                    f"User {user_id}: I am seeking spiritual guidance",
                    f"User {user_id}: What practices do you recommend?"
                ]
                
                user_results = []
                for query in user_queries:
                    # Emotional analysis
                    emotional_analysis = await test_emotional_cache.analyze_emotional_state(
                        query, f"concurrent_user_{user_id}"
                    )
                    
                    # Knowledge search
                    knowledge_results = await test_knowledge_cache.search_spiritual_knowledge(
                        query, f"concurrent_user_{user_id}", 
                        emotional_analysis.get('emotional_states', [])
                    )
                    
                    user_results.append({
                        'emotional_analysis': emotional_analysis,
                        'knowledge_results': knowledge_results
                    })
                
                return user_results
            
            # Run concurrent user sessions
            start_time = time.time()
            
            tasks = [simulate_user_session(i) for i in range(concurrent_users)]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Analyze concurrent performance
            successful_users = 0
            total_queries_processed = 0
            errors = []
            
            for i, user_results in enumerate(all_results):
                if isinstance(user_results, Exception):
                    errors.append(f"User {i}: {user_results}")
                else:
                    if len(user_results) == queries_per_user:
                        successful_users += 1
                        total_queries_processed += len(user_results)
                        
                        # Validate each query result
                        for result in user_results:
                            assert 'emotional_analysis' in result
                            assert 'knowledge_results' in result
                            assert result['emotional_analysis'] is not None
                            assert result['knowledge_results'] is not None
            
            # Calculate performance metrics
            user_success_rate = successful_users / concurrent_users
            queries_per_second = total_queries_processed / total_time
            avg_time_per_user = total_time / concurrent_users
            
            # Concurrent performance assertions
            assert user_success_rate >= 0.9, f"Low concurrent user success rate: {user_success_rate}"
            assert queries_per_second >= 10, f"Low query throughput: {queries_per_second} qps"
            assert avg_time_per_user <= 5.0, f"Concurrent user sessions too slow: {avg_time_per_user}s"
            assert len(errors) == 0, f"Concurrent user errors: {errors}"
            
            test_result.details = {
                'concurrent_users': concurrent_users,
                'queries_per_user': queries_per_user,
                'successful_users': successful_users,
                'user_success_rate': user_success_rate,
                'total_queries_processed': total_queries_processed,
                'queries_per_second': queries_per_second,
                'total_time': total_time,
                'avg_time_per_user': avg_time_per_user,
                'errors': errors
            }
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, test_emotional_cache, test_knowledge_cache, test_cache_manager):
        """Test system error recovery and resilience"""
        async with test_framework.test_context("error_recovery_resilience", "integration") as test_result:
            error_scenarios = []
            recovery_successes = 0
            
            # Scenario 1: Cache failure simulation
            try:
                # Simulate cache unavailability by setting invalid TTL
                with patch.object(test_cache_manager, 'get', side_effect=Exception("Cache unavailable")):
                    # System should still function without cache
                    result = await test_emotional_cache.analyze_emotional_state(
                        "Test query during cache failure", "resilience_user_1"
                    )
                    
                    # Should get a response even without cache
                    assert result is not None, "No fallback when cache fails"
                    assert 'emotional_states' in result, "Invalid fallback response structure"
                    
                recovery_successes += 1
                error_scenarios.append("cache_failure_recovery")
                
            except Exception as e:
                error_scenarios.append(f"cache_failure_failed: {e}")
            
            # Scenario 2: Partial service failure
            try:
                # Simulate knowledge service failure
                with patch.object(test_knowledge_cache, 'search_spiritual_knowledge', 
                                side_effect=Exception("Knowledge service down")):
                    
                    # Emotional analysis should still work
                    emotional_result = await test_emotional_cache.analyze_emotional_state(
                        "Test emotional analysis during knowledge failure", "resilience_user_2"
                    )
                    
                    assert emotional_result is not None, "Emotional analysis failed during knowledge service failure"
                    
                recovery_successes += 1
                error_scenarios.append("partial_service_failure_recovery")
                
            except Exception as e:
                error_scenarios.append(f"partial_service_failure_failed: {e}")
            
            # Scenario 3: Invalid input handling
            try:
                invalid_inputs = [None, "", "   ", {"invalid": "input"}, 123]
                
                for invalid_input in invalid_inputs:
                    # Convert to string and process
                    input_str = str(invalid_input) if invalid_input is not None else ""
                    
                    result = await test_emotional_cache.analyze_emotional_state(
                        input_str, "resilience_user_3"
                    )
                    
                    # Should return valid response structure even for invalid input
                    assert isinstance(result, dict), f"Invalid response type for input: {invalid_input}"
                    assert 'emotional_states' in result, f"Missing emotional_states for input: {invalid_input}"
                    assert 'confidence' in result, f"Missing confidence for input: {invalid_input}"
                
                recovery_successes += 1
                error_scenarios.append("invalid_input_handling")
                
            except Exception as e:
                error_scenarios.append(f"invalid_input_handling_failed: {e}")
            
            # Scenario 4: Timeout handling
            try:
                # Simulate slow operations
                original_sleep = asyncio.sleep
                
                async def slow_operation(*args, **kwargs):
                    await original_sleep(0.1)  # Simulate slow operation
                    return {"emotional_states": ["patience"], "confidence": 0.7}
                
                with patch.object(test_emotional_cache, 'analyze_emotional_state', side_effect=slow_operation):
                    start_time = time.time()
                    
                    # Should handle slow operations gracefully
                    result = await asyncio.wait_for(
                        test_emotional_cache.analyze_emotional_state("Timeout test", "resilience_user_4"),
                        timeout=1.0
                    )
                    
                    operation_time = time.time() - start_time
                    assert operation_time < 1.0, "Operation took too long"
                    assert result is not None, "No result from slow operation"
                
                recovery_successes += 1
                error_scenarios.append("timeout_handling")
                
            except asyncio.TimeoutError:
                error_scenarios.append("timeout_handling_failed: Operation timed out")
            except Exception as e:
                error_scenarios.append(f"timeout_handling_failed: {e}")
            
            # Calculate resilience metrics
            total_scenarios = 4
            resilience_rate = recovery_successes / total_scenarios
            
            # System should recover from most error scenarios
            assert resilience_rate >= 0.75, f"Low error recovery rate: {resilience_rate}"
            
            test_result.details = {
                'total_error_scenarios': total_scenarios,
                'recovery_successes': recovery_successes,
                'resilience_rate': resilience_rate,
                'error_scenarios': error_scenarios
            }
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, test_emotional_cache, test_knowledge_cache):
        """Test system performance under sustained load"""
        async with test_framework.test_context("performance_under_load", "integration") as test_result:
            # Load test parameters
            load_duration = 10  # seconds
            target_qps = 20  # queries per second
            
            # Generate test queries
            base_queries = [
                "How can I find inner peace?",
                "What is the nature of consciousness?",
                "I am seeking spiritual guidance",
                "How to practice meditation?",
                "What is enlightenment?"
            ]
            
            # Performance tracking
            start_time = time.time()
            completed_operations = 0
            response_times = []
            errors = []
            
            async def perform_operation(query: str, user_id: str):
                """Perform a complete spiritual guidance operation"""
                operation_start = time.time()
                
                try:
                    # Emotional analysis
                    emotional_analysis = await test_emotional_cache.analyze_emotional_state(query, user_id)
                    
                    # Knowledge search
                    knowledge_results = await test_knowledge_cache.search_spiritual_knowledge(
                        query, user_id, emotional_analysis.get('emotional_states', [])
                    )
                    
                    operation_time = time.time() - operation_start
                    response_times.append(operation_time)
                    
                    return True
                    
                except Exception as e:
                    errors.append(str(e))
                    return False
            
            # Run load test
            operations = []
            operation_interval = 1.0 / target_qps
            
            while (time.time() - start_time) < load_duration:
                # Select random query and create unique user ID
                query = base_queries[completed_operations % len(base_queries)]
                user_id = f"load_test_user_{completed_operations}"
                
                # Start operation
                operation = perform_operation(query, user_id)
                operations.append(operation)
                completed_operations += 1
                
                # Wait for next operation
                await asyncio.sleep(operation_interval)
            
            # Wait for all operations to complete
            results = await asyncio.gather(*operations, return_exceptions=True)
            
            # Analyze load test results
            total_time = time.time() - start_time
            successful_operations = sum(1 for r in results if r is True)
            actual_qps = completed_operations / total_time
            success_rate = successful_operations / completed_operations
            
            # Performance statistics
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                # Calculate percentiles
                sorted_times = sorted(response_times)
                p95_response_time = sorted_times[int(0.95 * len(sorted_times))]
                p99_response_time = sorted_times[int(0.99 * len(sorted_times))]
            else:
                avg_response_time = max_response_time = min_response_time = 0
                p95_response_time = p99_response_time = 0
            
            # Performance assertions
            assert success_rate >= 0.95, f"Low success rate under load: {success_rate}"
            assert avg_response_time <= 0.5, f"High average response time: {avg_response_time}s"
            assert p95_response_time <= 1.0, f"High P95 response time: {p95_response_time}s"
            assert len(errors) / completed_operations <= 0.05, f"Too many errors: {len(errors)}/{completed_operations}"
            
            test_result.details = {
                'load_duration': load_duration,
                'target_qps': target_qps,
                'actual_qps': actual_qps,
                'completed_operations': completed_operations,
                'successful_operations': successful_operations,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'min_response_time': min_response_time,
                'p95_response_time': p95_response_time,
                'p99_response_time': p99_response_time,
                'error_count': len(errors),
                'errors': errors[:10]  # Include first 10 errors for debugging
            }

# Additional integration test utilities
class TestSystemIntegrationUtils:
    """Utility functions for system integration testing"""
    
    @staticmethod
    def validate_complete_guidance_response(response: Dict[str, Any]) -> bool:
        """Validate complete spiritual guidance response structure"""
        required_components = [
            'emotional_analysis',
            'knowledge_insights', 
            'empathetic_response',
            'integrated_guidance'
        ]
        
        for component in required_components:
            if component not in response:
                return False
        
        # Validate emotional analysis component
        emotional = response['emotional_analysis']
        if not isinstance(emotional.get('emotional_states'), list):
            return False
        if not isinstance(emotional.get('confidence'), (int, float)):
            return False
        
        # Validate knowledge insights component
        knowledge = response['knowledge_insights']
        if not isinstance(knowledge.get('concepts'), list):
            return False
        if not isinstance(knowledge.get('recommended_practices'), list):
            return False
        
        # Validate integrated guidance component
        guidance = response['integrated_guidance']
        if not isinstance(guidance.get('primary_message'), str):
            return False
        if not isinstance(guidance.get('spiritual_practices'), list):
            return False
        
        return True
    
    @staticmethod
    def calculate_system_performance_score(metrics: Dict[str, Any]) -> float:
        """Calculate overall system performance score"""
        score = 0.0
        max_score = 5.0
        
        # Response time score (0-1)
        avg_time = metrics.get('avg_response_time', 1.0)
        if avg_time <= 0.1:
            score += 1.0
        elif avg_time <= 0.3:
            score += 0.7
        elif avg_time <= 0.5:
            score += 0.5
        elif avg_time <= 1.0:
            score += 0.3
        
        # Success rate score (0-1)
        success_rate = metrics.get('success_rate', 0.0)
        score += success_rate
        
        # Cache performance score (0-1)
        cache_hit_ratio = metrics.get('cache_hit_ratio', 0.0)
        score += cache_hit_ratio
        
        # Quality score (0-1)
        avg_confidence = metrics.get('avg_confidence', 0.0)
        score += avg_confidence
        
        # Scalability score (0-1)
        concurrent_success = metrics.get('concurrent_success_rate', 0.0)
        score += concurrent_success
        
        return score / max_score
    
    @staticmethod
    def assess_user_experience_quality(session_metrics: List[Dict]) -> Dict[str, float]:
        """Assess user experience quality from session metrics"""
        if not session_metrics:
            return {'overall': 0.0, 'consistency': 0.0, 'progression': 0.0}
        
        # Consistency assessment
        response_times = [m.get('response_time', 1.0) for m in session_metrics]
        time_variance = max(response_times) - min(response_times)
        consistency_score = max(0.0, 1.0 - (time_variance / 2.0))  # Penalize high variance
        
        # Progression assessment
        personalization_levels = [m.get('personalization_level', 0.0) for m in session_metrics]
        if len(personalization_levels) > 1:
            progression_score = max(0.0, personalization_levels[-1] - personalization_levels[0])
        else:
            progression_score = personalization_levels[0] if personalization_levels else 0.0
        
        # Overall quality
        avg_confidence = sum(m.get('emotional_confidence', 0.0) for m in session_metrics) / len(session_metrics)
        avg_concepts = sum(m.get('concepts_provided', 0) for m in session_metrics) / len(session_metrics)
        
        overall_score = (avg_confidence + min(1.0, avg_concepts / 3.0) + consistency_score + progression_score) / 4.0
        
        return {
            'overall': overall_score,
            'consistency': consistency_score,
            'progression': progression_score,
            'avg_confidence': avg_confidence,
            'avg_concepts': avg_concepts
        }

# Export test utilities
__all__ = [
    'TestDharmaMindSystemIntegration',
    'TestSystemIntegrationUtils'
]