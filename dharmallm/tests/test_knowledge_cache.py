"""
🧪 Unit Tests for Optimized Knowledge Cache
=========================================

Comprehensive unit tests for the spiritual knowledge caching system:
- Knowledge search accuracy and relevance
- Concept retrieval and definition quality
- Embedding generation and caching
- Performance optimization validation
- Cache hit ratio and response times
- Spiritual content integrity

These tests ensure the knowledge base provides accurate, relevant, and
lightning-fast spiritual wisdom and guidance.
"""

import pytest
import asyncio
import time
import hashlib
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Import test framework
from tests import (
    test_framework, DharmaMindTestFramework,
    spiritual_test_data, user_test_data,
    test_knowledge_cache, test_cache_manager
)

class TestOptimizedKnowledgeCache:
    """Unit tests for optimized knowledge cache system"""
    
    @pytest.mark.asyncio
    async def test_spiritual_knowledge_search(self, test_knowledge_cache, spiritual_test_data):
        """Test spiritual knowledge search functionality"""
        async with test_framework.test_context("spiritual_knowledge_search", "knowledge_cache") as test_result:
            # Test with various spiritual queries
            successful_searches = 0
            total_concepts_found = 0
            
            for query_data in spiritual_test_data['queries']:
                query = query_data['text']
                expected_concepts = query_data['expected_concepts']
                expected_practices = query_data['expected_practices']
                
                # Perform knowledge search
                start_time = time.time()
                results = await test_knowledge_cache.search_spiritual_knowledge(
                    query, "test_user", context_emotions=query_data['expected_emotions']
                )
                duration = time.time() - start_time
                
                # Performance assertion
                test_framework.assert_performance_threshold('knowledge_search', duration)
                
                # Validate search results structure
                assert isinstance(results, dict), "Search results must be a dictionary"
                assert 'concepts' in results, "Missing concepts in search results"
                assert 'recommended_practices' in results, "Missing practices in search results"
                assert 'related_concepts' in results, "Missing related concepts"
                
                # Check result quality
                concepts = results['concepts']
                practices = results['recommended_practices']
                
                if len(concepts) > 0:
                    successful_searches += 1
                    total_concepts_found += len(concepts)
                
                # Validate knowledge retrieval accuracy
                test_framework.assert_knowledge_retrieval_accuracy(results, expected_concepts)
            
            search_success_rate = successful_searches / len(spiritual_test_data['queries'])
            avg_concepts_per_query = total_concepts_found / len(spiritual_test_data['queries'])
            
            assert search_success_rate >= 0.8, f"Low search success rate: {search_success_rate}"
            assert avg_concepts_per_query >= 2, f"Too few concepts per query: {avg_concepts_per_query}"
            
            test_result.details = {
                'queries_tested': len(spiritual_test_data['queries']),
                'successful_searches': successful_searches,
                'search_success_rate': search_success_rate,
                'avg_concepts_per_query': avg_concepts_per_query
            }
    
    @pytest.mark.asyncio
    async def test_concept_definition_retrieval(self, test_knowledge_cache, spiritual_test_data):
        """Test spiritual concept definition retrieval"""
        async with test_framework.test_context("concept_definition_retrieval", "knowledge_cache") as test_result:
            concepts = spiritual_test_data['concepts']
            successful_retrievals = 0
            
            for concept_name, expected_data in concepts.items():
                # Get concept definition
                start_time = time.time()
                definition = await test_knowledge_cache.get_concept_definition(concept_name)
                duration = time.time() - start_time
                
                # Performance assertion
                test_framework.assert_performance_threshold('knowledge_search', duration)
                
                # Validate definition structure
                assert isinstance(definition, dict), f"Definition for '{concept_name}' must be a dictionary"
                assert 'definition' in definition, f"Missing definition for '{concept_name}'"
                
                # Check definition quality
                def_text = definition['definition']
                assert len(def_text) > 10, f"Definition too short for '{concept_name}'"
                assert concept_name.lower() not in def_text.lower() or len(def_text) > 50, f"Circular definition for '{concept_name}'"
                
                # Check for related concepts
                if 'related_concepts' in definition:
                    related = definition['related_concepts']
                    assert isinstance(related, list), f"Related concepts must be a list for '{concept_name}'"
                
                successful_retrievals += 1
            
            retrieval_success_rate = successful_retrievals / len(concepts)
            assert retrieval_success_rate >= 0.9, f"Low concept retrieval rate: {retrieval_success_rate}"
            
            test_result.details = {
                'concepts_tested': len(concepts),
                'successful_retrievals': successful_retrievals,
                'retrieval_success_rate': retrieval_success_rate
            }
    
    @pytest.mark.asyncio
    async def test_embedding_generation_and_caching(self, test_knowledge_cache):
        """Test embedding generation and caching functionality"""
        async with test_framework.test_context("embedding_generation_caching", "knowledge_cache") as test_result:
            test_texts = [
                "meditation practice",
                "spiritual enlightenment", 
                "consciousness awareness",
                "divine love compassion",
                "inner peace tranquility"
            ]
            
            # Test embedding generation and caching
            cache_hits = 0
            embedding_times = []
            
            for text in test_texts:
                # First call - should generate and cache
                start_time = time.time()
                embedding1 = await test_knowledge_cache.get_cached_embedding(text)
                first_time = time.time() - start_time
                
                # Second call - should hit cache
                start_time = time.time()
                embedding2 = await test_knowledge_cache.get_cached_embedding(text)
                second_time = time.time() - start_time
                
                embedding_times.extend([first_time, second_time])
                
                # Validate embeddings
                assert embedding1 is not None, f"No embedding generated for '{text}'"
                assert embedding2 is not None, f"No cached embedding for '{text}'"
                assert embedding1 == embedding2, f"Cached embedding mismatch for '{text}'"
                
                # Check for cache hit (second call should be faster)
                if second_time < first_time:
                    cache_hits += 1
            
            cache_hit_ratio = cache_hits / len(test_texts)
            avg_embedding_time = sum(embedding_times) / len(embedding_times)
            
            # Performance assertions
            assert avg_embedding_time <= 0.05, f"Embedding generation too slow: {avg_embedding_time}"
            assert cache_hit_ratio >= 0.7, f"Low embedding cache hit ratio: {cache_hit_ratio}"
            
            test_result.details = {
                'texts_tested': len(test_texts),
                'cache_hits': cache_hits,
                'cache_hit_ratio': cache_hit_ratio,
                'avg_embedding_time': avg_embedding_time
            }
    
    @pytest.mark.asyncio
    async def test_popular_query_caching(self, test_knowledge_cache):
        """Test popular query result caching"""
        async with test_framework.test_context("popular_query_caching", "knowledge_cache") as test_result:
            # Simulate popular queries
            popular_queries = [
                "What is meditation?",
                "How to find inner peace?",
                "What is the nature of consciousness?",
                "How to practice mindfulness?",
                "What is enlightenment?"
            ]
            
            cache_performance = []
            
            for query in popular_queries:
                # First search - should cache results
                start_time = time.time()
                results1 = await test_knowledge_cache.search_spiritual_knowledge(query, "cache_test_user")
                first_time = time.time() - start_time
                
                # Second search - should hit cache
                start_time = time.time()
                results2 = await test_knowledge_cache.search_spiritual_knowledge(query, "cache_test_user")
                second_time = time.time() - start_time
                
                # Validate cache behavior
                assert results1 is not None, f"No results for first search: {query}"
                assert results2 is not None, f"No results for cached search: {query}"
                
                # Results should be identical
                assert results1['concepts'] == results2['concepts'], f"Cache inconsistency for: {query}"
                
                cache_performance.append({
                    'query': query,
                    'first_time': first_time,
                    'second_time': second_time,
                    'cache_speedup': first_time / second_time if second_time > 0 else float('inf')
                })
            
            # Calculate cache performance metrics
            avg_speedup = sum(p['cache_speedup'] for p in cache_performance if p['cache_speedup'] != float('inf'))
            avg_speedup = avg_speedup / len([p for p in cache_performance if p['cache_speedup'] != float('inf')])
            
            avg_cache_time = sum(p['second_time'] for p in cache_performance) / len(cache_performance)
            
            # Cache should provide significant speedup
            assert avg_speedup >= 2.0, f"Insufficient cache speedup: {avg_speedup}x"
            assert avg_cache_time <= 0.05, f"Cached queries too slow: {avg_cache_time}s"
            
            test_result.details = {
                'queries_tested': len(popular_queries),
                'avg_speedup': avg_speedup,
                'avg_cache_time': avg_cache_time,
                'performance_details': cache_performance
            }
    
    @pytest.mark.asyncio
    async def test_context_aware_search(self, test_knowledge_cache):
        """Test context-aware spiritual knowledge search"""
        async with test_framework.test_context("context_aware_search", "knowledge_cache") as test_result:
            # Test how context emotions influence search results
            base_query = "How can I find peace?"
            
            context_scenarios = [
                {
                    'emotions': ['anger', 'frustration'],
                    'expected_focus': 'emotional_regulation'
                },
                {
                    'emotions': ['sadness', 'grief'],
                    'expected_focus': 'healing_comfort'
                },
                {
                    'emotions': ['seeking', 'curiosity'],
                    'expected_focus': 'spiritual_exploration'
                },
                {
                    'emotions': ['peace', 'gratitude'],
                    'expected_focus': 'maintaining_cultivation'
                }
            ]
            
            context_adaptations = 0
            unique_results = set()
            
            for scenario in context_scenarios:
                results = await test_knowledge_cache.search_spiritual_knowledge(
                    base_query, 
                    f"context_user_{scenario['expected_focus']}", 
                    context_emotions=scenario['emotions']
                )
                
                # Check if context influenced results
                concepts = results.get('concepts', [])
                practices = results.get('recommended_practices', [])
                
                # Convert results to a hashable format for uniqueness check
                result_signature = str(sorted(concepts)) + str(sorted(practices))
                unique_results.add(result_signature)
                
                # Check for emotional context adaptation
                if any(emotion in str(results).lower() for emotion in scenario['emotions']):
                    context_adaptations += 1
                
                # Validate result structure
                assert len(concepts) > 0, f"No concepts for context: {scenario['emotions']}"
                assert len(practices) > 0, f"No practices for context: {scenario['emotions']}"
            
            adaptation_rate = context_adaptations / len(context_scenarios)
            result_diversity = len(unique_results) / len(context_scenarios)
            
            # Context should influence results
            assert result_diversity >= 0.5, f"Low result diversity: {result_diversity}"
            
            test_result.details = {
                'scenarios_tested': len(context_scenarios),
                'context_adaptations': context_adaptations,
                'adaptation_rate': adaptation_rate,
                'result_diversity': result_diversity,
                'unique_results': len(unique_results)
            }
    
    @pytest.mark.asyncio
    async def test_knowledge_performance_metrics(self, test_knowledge_cache):
        """Test knowledge cache performance metrics collection"""
        async with test_framework.test_context("knowledge_performance_metrics", "knowledge_cache") as test_result:
            # Perform various operations to generate metrics
            test_operations = [
                ("search", "meditation practice"),
                ("concept", "consciousness"),
                ("search", "inner peace"),
                ("embedding", "spiritual wisdom"),
                ("concept", "dharma")
            ]
            
            for operation, query in test_operations:
                if operation == "search":
                    await test_knowledge_cache.search_spiritual_knowledge(query, "metrics_user")
                elif operation == "concept":
                    await test_knowledge_cache.get_concept_definition(query)
                elif operation == "embedding":
                    await test_knowledge_cache.get_cached_embedding(query)
            
            # Generate performance report
            report = await test_knowledge_cache.get_knowledge_performance_report()
            
            # Validate report structure
            assert isinstance(report, dict), "Performance report must be a dictionary"
            
            required_metrics = [
                'total_queries', 'cache_hit_ratio', 'average_query_time',
                'concept_retrievals', 'embedding_generations'
            ]
            
            for metric in required_metrics:
                assert metric in report, f"Missing metric: {metric}"
            
            # Validate metric values
            assert report['total_queries'] >= len(test_operations), "Incorrect query count"
            assert 0 <= report['cache_hit_ratio'] <= 1, "Invalid cache hit ratio"
            assert report['average_query_time'] > 0, "Invalid average query time"
            
            test_result.details = {
                'operations_performed': len(test_operations),
                'performance_report': report
            }
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_and_updates(self, test_knowledge_cache, test_cache_manager):
        """Test cache invalidation and update mechanisms"""
        async with test_framework.test_context("cache_invalidation_updates", "knowledge_cache") as test_result:
            # Test data
            test_concept = "test_meditation_concept"
            initial_definition = "Initial meditation definition"
            updated_definition = "Updated meditation definition with more details"
            
            # Store initial concept definition
            from app.services.cache_service import CacheCategory
            await test_cache_manager.set(
                CacheCategory.KNOWLEDGE_BASE,
                f"concept_definition:{test_concept}",
                {"definition": initial_definition}
            )
            
            # Retrieve initial definition
            definition1 = await test_knowledge_cache.get_concept_definition(test_concept)
            assert definition1['definition'] == initial_definition, "Initial definition not cached"
            
            # Update the definition
            await test_cache_manager.set(
                CacheCategory.KNOWLEDGE_BASE,
                f"concept_definition:{test_concept}",
                {"definition": updated_definition}
            )
            
            # Retrieve updated definition
            definition2 = await test_knowledge_cache.get_concept_definition(test_concept)
            assert definition2['definition'] == updated_definition, "Definition update failed"
            
            # Test cache expiration by setting very short TTL
            short_ttl_key = "short_ttl_test"
            await test_cache_manager.set(
                CacheCategory.KNOWLEDGE_BASE,
                short_ttl_key,
                {"data": "temporary"},
                ttl_seconds=1
            )
            
            # Should exist immediately
            immediate_result = await test_cache_manager.get(CacheCategory.KNOWLEDGE_BASE, short_ttl_key)
            assert immediate_result is not None, "Short TTL item not found immediately"
            
            # Wait for expiration
            await asyncio.sleep(2)
            
            # Should be expired
            expired_result = await test_cache_manager.get(CacheCategory.KNOWLEDGE_BASE, short_ttl_key)
            assert expired_result is None, "Short TTL item did not expire"
            
            test_result.details = {
                'update_test': 'passed',
                'expiration_test': 'passed',
                'cache_consistency': 'verified'
            }
    
    @pytest.mark.asyncio
    async def test_concurrent_knowledge_operations(self, test_knowledge_cache):
        """Test concurrent knowledge cache operations"""
        async with test_framework.test_context("concurrent_knowledge_operations", "knowledge_cache") as test_result:
            # Create concurrent operation tasks
            queries = [
                "What is mindfulness?",
                "How to practice yoga?", 
                "What is consciousness?",
                "How to meditate?",
                "What is enlightenment?"
            ]
            
            concepts = ["meditation", "yoga", "dharma", "karma", "moksha"]
            embeddings = ["peace", "love", "wisdom", "compassion", "awareness"]
            
            # Run concurrent operations
            start_time = time.time()
            tasks = []
            
            # Add search tasks
            for i, query in enumerate(queries):
                task = test_knowledge_cache.search_spiritual_knowledge(query, f"concurrent_user_{i}")
                tasks.append(("search", task))
            
            # Add concept tasks
            for concept in concepts:
                task = test_knowledge_cache.get_concept_definition(concept)
                tasks.append(("concept", task))
            
            # Add embedding tasks
            for embedding_text in embeddings:
                task = test_knowledge_cache.get_cached_embedding(embedding_text)
                tasks.append(("embedding", task))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            successful_operations = 0
            errors = []
            
            for i, result in enumerate(results):
                operation_type = tasks[i][0]
                
                if isinstance(result, Exception):
                    errors.append(f"{operation_type} {i}: {result}")
                else:
                    if operation_type == "search" and isinstance(result, dict) and 'concepts' in result:
                        successful_operations += 1
                    elif operation_type == "concept" and isinstance(result, dict) and 'definition' in result:
                        successful_operations += 1
                    elif operation_type == "embedding" and result is not None:
                        successful_operations += 1
                    else:
                        errors.append(f"{operation_type} {i}: Invalid result format")
            
            success_rate = successful_operations / len(tasks)
            avg_time_per_operation = total_time / len(tasks)
            
            # Concurrent operations should not significantly degrade performance
            assert success_rate >= 0.9, f"Low concurrent success rate: {success_rate}"
            assert avg_time_per_operation <= 0.2, f"Concurrent operations too slow: {avg_time_per_operation}"
            assert len(errors) == 0, f"Concurrent operation errors: {errors}"
            
            test_result.details = {
                'total_operations': len(tasks),
                'successful_operations': successful_operations,
                'success_rate': success_rate,
                'total_time': total_time,
                'avg_time_per_operation': avg_time_per_operation,
                'errors': errors
            }
    
    @pytest.mark.asyncio
    async def test_spiritual_content_accuracy(self, test_knowledge_cache, spiritual_test_data):
        """Test accuracy and appropriateness of spiritual content"""
        async with test_framework.test_context("spiritual_content_accuracy", "knowledge_cache") as test_result:
            # Test spiritual concepts for accuracy
            spiritual_concepts = spiritual_test_data['concepts']
            accuracy_scores = []
            
            for concept_name, expected_data in spiritual_concepts.items():
                # Get concept from cache
                result = await test_knowledge_cache.get_concept_definition(concept_name)
                
                # Validate spiritual appropriateness
                definition = result.get('definition', '')
                
                # Check definition quality
                quality_score = self._assess_spiritual_definition_quality(
                    concept_name, definition, expected_data
                )
                accuracy_scores.append(quality_score)
                
                # Ensure definition is spiritual/philosophical in nature
                spiritual_keywords = [
                    'spiritual', 'consciousness', 'awareness', 'divine', 'sacred',
                    'meditation', 'peace', 'wisdom', 'enlightenment', 'practice'
                ]
                
                has_spiritual_context = any(
                    keyword in definition.lower() for keyword in spiritual_keywords
                )
                
                assert has_spiritual_context or len(definition) > 50, f"Definition lacks spiritual context: {concept_name}"
            
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            
            assert avg_accuracy >= 0.7, f"Low spiritual content accuracy: {avg_accuracy}"
            
            test_result.details = {
                'concepts_evaluated': len(spiritual_concepts),
                'avg_accuracy_score': avg_accuracy,
                'accuracy_scores': accuracy_scores
            }
    
    def _assess_spiritual_definition_quality(self, concept: str, definition: str, expected_data: Dict) -> float:
        """Assess the quality of a spiritual definition"""
        score = 0.0
        max_score = 4.0
        
        # Length appropriateness (0-1 points)
        if 20 <= len(definition) <= 200:
            score += 1.0
        elif 10 <= len(definition) <= 300:
            score += 0.5
        
        # Related concept coverage (0-1 points)
        expected_related = expected_data.get('related_concepts', [])
        if expected_related:
            covered_concepts = sum(
                1 for related in expected_related 
                if related.lower() in definition.lower()
            )
            score += min(1.0, covered_concepts / len(expected_related))
        else:
            score += 0.5  # Give partial credit if no expected related concepts
        
        # Avoids circular definition (0-1 points)
        if concept.lower() not in definition.lower()[:50]:
            score += 1.0
        elif len(definition) > 100:  # Longer definitions with concept name are acceptable
            score += 0.5
        
        # Spiritual appropriateness (0-1 points)
        spiritual_indicators = [
            'practice', 'spiritual', 'consciousness', 'awareness', 'inner',
            'divine', 'sacred', 'meditation', 'wisdom', 'enlightenment'
        ]
        spiritual_count = sum(
            1 for indicator in spiritual_indicators 
            if indicator in definition.lower()
        )
        score += min(1.0, spiritual_count / 3)  # Max score if 3+ spiritual indicators
        
        return score / max_score

# Additional test utilities for knowledge cache
class TestKnowledgeCacheUtils:
    """Utility functions for knowledge cache testing"""
    
    @staticmethod
    def validate_search_results_structure(results: Dict[str, Any]) -> bool:
        """Validate search results structure"""
        required_fields = [
            'concepts', 'recommended_practices', 'related_concepts', 
            'query_hash', 'response_time', 'cache_hit'
        ]
        
        for field in required_fields:
            if field not in results:
                return False
        
        # Type validations
        if not isinstance(results['concepts'], list):
            return False
        if not isinstance(results['recommended_practices'], list):
            return False
        if not isinstance(results['related_concepts'], list):
            return False
        
        return True
    
    @staticmethod
    def calculate_concept_relevance(query: str, concepts: List[Dict]) -> float:
        """Calculate relevance of concepts to query"""
        if not concepts:
            return 0.0
        
        query_words = set(query.lower().split())
        relevant_concepts = 0
        
        for concept in concepts:
            concept_text = ""
            if isinstance(concept, dict):
                concept_text = f"{concept.get('name', '')} {concept.get('definition', '')}"
            else:
                concept_text = str(concept)
            
            concept_words = set(concept_text.lower().split())
            if query_words & concept_words:  # Any word overlap
                relevant_concepts += 1
        
        return relevant_concepts / len(concepts)
    
    @staticmethod
    def assess_cache_efficiency(cache_times: List[float], non_cache_times: List[float]) -> Dict[str, float]:
        """Assess cache efficiency metrics"""
        if not cache_times or not non_cache_times:
            return {'speedup': 0.0, 'efficiency': 0.0}
        
        avg_cache_time = sum(cache_times) / len(cache_times)
        avg_non_cache_time = sum(non_cache_times) / len(non_cache_times)
        
        speedup = avg_non_cache_time / avg_cache_time if avg_cache_time > 0 else 0
        efficiency = 1 - (avg_cache_time / avg_non_cache_time) if avg_non_cache_time > 0 else 0
        
        return {
            'speedup': speedup,
            'efficiency': max(0, efficiency),
            'avg_cache_time': avg_cache_time,
            'avg_non_cache_time': avg_non_cache_time
        }

# Export test utilities
__all__ = [
    'TestOptimizedKnowledgeCache',
    'TestKnowledgeCacheUtils'
]