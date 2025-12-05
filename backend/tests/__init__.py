"""
🧪 DharmaMind Comprehensive Testing Framework
===========================================

Enterprise-grade testing framework for the DharmaMind spiritual guidance platform:
- Unit tests for all core modules
- Integration tests for system workflows
- Performance benchmarks and load testing
- Spiritual guidance accuracy validation
- Security and authentication testing
- Cache performance validation
- End-to-end user experience testing

This framework ensures the highest quality and reliability for users seeking
spiritual guidance and enlightenment.
"""

import pytest
import asyncio
import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from unittest.mock import Mock, AsyncMock, patch
from contextlib import asynccontextmanager

# Add backend to path
backend_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(backend_path))

# Test framework imports
import fakeredis.aioredis
import pytest_asyncio

# Setup comprehensive logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Standardized test result structure"""
    test_name: str
    category: str
    status: str  # passed, failed, error, skipped
    duration: float
    message: Optional[str] = None
    details: Optional[Dict] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class TestSuiteReport:
    """Comprehensive test suite report"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_duration: float
    success_rate: float
    results: List[TestResult]
    environment: Dict[str, Any]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class DharmaMindTestFramework:
    """
    Comprehensive testing framework for DharmaMind platform
    """
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.test_environment = self._setup_test_environment()
        
        # Test data
        self.spiritual_test_data = self._load_spiritual_test_data()
        self.user_test_data = self._load_user_test_data()
        
        # Mock services
        self.mock_services = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            'emotional_analysis': 0.1,  # 100ms
            'knowledge_search': 0.1,    # 100ms
            'spiritual_guidance': 0.2,   # 200ms
            'authentication': 0.05,     # 50ms
            'cache_operations': 0.01,   # 10ms
            'api_response': 0.3         # 300ms
        }
    
    def _setup_test_environment(self) -> Dict[str, Any]:
        """Setup test environment configuration"""
        return {
            'testing': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 15,  # Use different DB for tests
            'cache_enabled': True,
            'authentication_enabled': True,
            'security_enabled': True,
            'performance_monitoring': True,
            'log_level': 'DEBUG',
            'test_data_path': str(Path(__file__).parent / 'fixtures'),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def _load_spiritual_test_data(self) -> Dict[str, Any]:
        """Load spiritual guidance test data"""
        return {
            'queries': [
                {
                    'text': "How can I find inner peace through meditation?",
                    'expected_emotions': ['seeking', 'peace'],
                    'expected_concepts': ['meditation', 'inner_peace', 'mindfulness'],
                    'expected_practices': ['meditation', 'breathing_exercises']
                },
                {
                    'text': "I am struggling with anger and need spiritual guidance",
                    'expected_emotions': ['anger', 'seeking', 'struggle'],
                    'expected_concepts': ['emotional_regulation', 'compassion', 'forgiveness'],
                    'expected_practices': ['loving_kindness', 'forgiveness_practice']
                },
                {
                    'text': "What is the nature of consciousness and enlightenment?",
                    'expected_emotions': ['wisdom', 'seeking', 'curiosity'],
                    'expected_concepts': ['consciousness', 'enlightenment', 'awareness'],
                    'expected_practices': ['self_inquiry', 'contemplation']
                },
                {
                    'text': "I feel grateful for my spiritual journey",
                    'expected_emotions': ['gratitude', 'joy', 'peace'],
                    'expected_concepts': ['spiritual_journey', 'gratitude', 'growth'],
                    'expected_practices': ['gratitude_meditation', 'reflection']
                },
                {
                    'text': "How do I overcome suffering and find liberation?",
                    'expected_emotions': ['suffering', 'seeking', 'hope'],
                    'expected_concepts': ['liberation', 'suffering', 'freedom'],
                    'expected_practices': ['mindfulness', 'detachment_practice']
                }
            ],
            'concepts': {
                'meditation': {
                    'definition': 'Practice of focused attention and awareness',
                    'related_concepts': ['mindfulness', 'concentration', 'awareness'],
                    'practices': ['breathing_meditation', 'walking_meditation']
                },
                'consciousness': {
                    'definition': 'The state of being aware and perceiving',
                    'related_concepts': ['awareness', 'mind', 'perception'],
                    'practices': ['self_inquiry', 'witness_meditation']
                },
                'compassion': {
                    'definition': 'Loving-kindness and empathy for all beings',
                    'related_concepts': ['love', 'kindness', 'empathy'],
                    'practices': ['loving_kindness', 'compassion_meditation']
                }
            },
            'emotional_patterns': {
                'seeking_guidance': ['seeking', 'confusion', 'hope'],
                'spiritual_growth': ['growth', 'progress', 'evolution'],
                'inner_peace': ['peace', 'tranquility', 'calm'],
                'wisdom_inquiry': ['wisdom', 'understanding', 'insight']
            }
        }
    
    def _load_user_test_data(self) -> Dict[str, Any]:
        """Load user test data"""
        return {
            'test_users': [
                {
                    'user_id': 'test_user_1',
                    'profile': {
                        'spiritual_tradition': 'vedantic',
                        'experience_level': 'beginner',
                        'cultural_context': 'indian_traditional',
                        'language': 'english'
                    }
                },
                {
                    'user_id': 'test_user_2',
                    'profile': {
                        'spiritual_tradition': 'buddhist',
                        'experience_level': 'intermediate',
                        'cultural_context': 'western_contemporary',
                        'language': 'english'
                    }
                },
                {
                    'user_id': 'test_user_3',
                    'profile': {
                        'spiritual_tradition': 'universal',
                        'experience_level': 'advanced',
                        'cultural_context': 'secular_spiritual',
                        'language': 'english'
                    }
                }
            ],
            'session_data': {
                'active_sessions': 10,
                'avg_session_duration': 1800,  # 30 minutes
                'interactions_per_session': 15
            }
        }
    
    @asynccontextmanager
    async def test_context(self, test_name: str, category: str = "general"):
        """Context manager for test execution with timing and error handling"""
        start_time = time.time()
        test_result = TestResult(
            test_name=test_name,
            category=category,
            status="running",
            duration=0.0
        )
        
        try:
            logger.info(f"🧪 Starting test: {test_name}")
            yield test_result
            
            test_result.status = "passed"
            test_result.message = "Test completed successfully"
            
        except AssertionError as e:
            test_result.status = "failed"
            test_result.message = f"Assertion failed: {str(e)}"
            logger.error(f"❌ Test failed: {test_name} - {e}")
            
        except Exception as e:
            test_result.status = "error"
            test_result.message = f"Test error: {str(e)}"
            logger.error(f"💥 Test error: {test_name} - {e}")
            
        finally:
            test_result.duration = time.time() - start_time
            self.test_results.append(test_result)
            
            status_icon = "✅" if test_result.status == "passed" else "❌"
            logger.info(f"{status_icon} Test completed: {test_name} ({test_result.duration:.3f}s)")
    
    async def create_test_redis(self):
        """Create fake Redis instance for testing"""
        return fakeredis.aioredis.FakeRedis()
    
    async def create_mock_cache_manager(self):
        """Create mock cache manager for testing"""
        fake_redis = await self.create_test_redis()
        
        # Import cache manager
        from app.services.cache_service import CacheService as AdvancedCacheManager
        return AdvancedCacheManager(redis_client=fake_redis)
    
    async def create_mock_knowledge_cache(self):
        """Create mock knowledge cache for testing"""
        cache_manager = await self.create_mock_cache_manager()
        
        from app.services.intelligent_cache import OptimizedKnowledgeCache
        knowledge_cache = OptimizedKnowledgeCache(cache_manager)
        await knowledge_cache.initialize()
        return knowledge_cache
    
    async def create_mock_emotional_cache(self):
        """Create mock emotional cache for testing"""
        cache_manager = await self.create_mock_cache_manager()
        
        from app.services.intelligent_cache import OptimizedEmotionalCache
        emotional_cache = OptimizedEmotionalCache(cache_manager)
        await emotional_cache.initialize()
        return emotional_cache
    
    def assert_performance_threshold(self, operation: str, duration: float):
        """Assert that operation meets performance threshold"""
        threshold = self.performance_thresholds.get(operation, 1.0)
        assert duration <= threshold, f"{operation} took {duration:.3f}s, exceeds threshold {threshold}s"
    
    def assert_spiritual_response_quality(self, response: Dict[str, Any], expected: Dict[str, Any]):
        """Assert that spiritual guidance response meets quality standards"""
        # Check required fields
        required_fields = ['emotional_analysis', 'knowledge_insights', 'integrated_guidance']
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        
        # Check emotional analysis quality
        emotional_analysis = response.get('emotional_analysis', {})
        assert 'emotional_states' in emotional_analysis, "Missing emotional states"
        assert 'confidence' in emotional_analysis, "Missing confidence score"
        assert emotional_analysis.get('confidence', 0) >= 0.5, "Low confidence in emotional analysis"
        
        # Check knowledge insights
        knowledge_insights = response.get('knowledge_insights', {})
        assert 'concepts' in knowledge_insights, "Missing spiritual concepts"
        assert 'recommended_practices' in knowledge_insights, "Missing recommended practices"
        
        # Check integrated guidance
        integrated_guidance = response.get('integrated_guidance', {})
        assert 'primary_message' in integrated_guidance, "Missing primary message"
        assert 'spiritual_practices' in integrated_guidance, "Missing spiritual practices"
    
    def assert_emotional_analysis_accuracy(self, analysis: Dict[str, Any], expected_emotions: List[str]):
        """Assert emotional analysis accuracy"""
        detected_emotions = analysis.get('emotional_states', [])
        confidence = analysis.get('confidence', 0)
        
        # Check confidence threshold
        assert confidence >= 0.6, f"Low emotional analysis confidence: {confidence}"
        
        # Check if at least one expected emotion is detected
        overlap = set(detected_emotions) & set(expected_emotions)
        assert len(overlap) > 0, f"No overlap between detected {detected_emotions} and expected {expected_emotions}"
    
    def assert_knowledge_retrieval_accuracy(self, results: Dict[str, Any], expected_concepts: List[str]):
        """Assert knowledge retrieval accuracy"""
        retrieved_concepts = results.get('concepts', [])
        recommended_practices = results.get('recommended_practices', [])
        
        # Check if concepts were retrieved
        assert len(retrieved_concepts) > 0, "No concepts retrieved"
        
        # Check concept relevance
        concept_names = [c.get('name', '') for c in retrieved_concepts if isinstance(c, dict)]
        overlap = set(concept_names) & set(expected_concepts)
        assert len(overlap) > 0, f"No relevant concepts found. Retrieved: {concept_names}, Expected: {expected_concepts}"
        
        # Check practices recommendation
        assert len(recommended_practices) > 0, "No practices recommended"
    
    async def validate_cache_performance(self, cache_manager):
        """Validate cache performance"""
        # Test cache operations
        start_time = time.time()
        
        # Set operation
        await cache_manager.set("test", "perf_test", {"data": "test"})
        set_time = time.time() - start_time
        
        # Get operation
        start_time = time.time()
        result = await cache_manager.get("test", "perf_test")
        get_time = time.time() - start_time
        
        # Validate performance
        self.assert_performance_threshold('cache_operations', set_time)
        self.assert_performance_threshold('cache_operations', get_time)
        
        # Validate result
        assert result is not None, "Cache get operation failed"
        assert result.get('data') == 'test', "Cache data integrity failed"
    
    def generate_test_report(self, suite_name: str = "DharmaMind Test Suite") -> TestSuiteReport:
        """Generate comprehensive test report"""
        if not self.test_results:
            return TestSuiteReport(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                error_tests=0,
                skipped_tests=0,
                total_duration=0.0,
                success_rate=0.0,
                results=[],
                environment=self.test_environment
            )
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "passed"])
        failed_tests = len([r for r in self.test_results if r.status == "failed"])
        error_tests = len([r for r in self.test_results if r.status == "error"])
        skipped_tests = len([r for r in self.test_results if r.status == "skipped"])
        
        total_duration = sum(r.duration for r in self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        return TestSuiteReport(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests,
            total_duration=total_duration,
            success_rate=success_rate,
            results=self.test_results,
            environment=self.test_environment
        )
    
    def export_test_report(self, report: TestSuiteReport, filename: Optional[str] = None) -> str:
        """Export test report to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/reports/test_report_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert report to dict
        report_dict = asdict(report)
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"📊 Test report exported to: {filename}")
        return filename
    
    def print_test_summary(self, report: TestSuiteReport):
        """Print formatted test summary"""
        print("\n🧪 DHARMAMIND TEST SUITE SUMMARY")
        print("=" * 50)
        print(f"Suite: {report.suite_name}")
        print(f"Total Tests: {report.total_tests}")
        print(f"✅ Passed: {report.passed_tests}")
        print(f"❌ Failed: {report.failed_tests}")
        print(f"💥 Errors: {report.error_tests}")
        print(f"⏭️ Skipped: {report.skipped_tests}")
        print(f"📊 Success Rate: {report.success_rate:.1%}")
        print(f"⏱️ Total Duration: {report.total_duration:.2f}s")
        print("=" * 50)
        
        # Print failed tests details
        failed_tests = [r for r in report.results if r.status in ["failed", "error"]]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test.test_name}: {test.message}")
        
        # Print performance summary
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        categories = {}
        for result in report.results:
            category = result.category
            if category not in categories:
                categories[category] = []
            categories[category].append(result.duration)
        
        for category, durations in categories.items():
            avg_duration = sum(durations) / len(durations)
            print(f"  • {category}: {avg_duration:.3f}s avg")

# Global test framework instance
test_framework = DharmaMindTestFramework()

# Pytest fixtures
@pytest.fixture
async def fake_redis():
    """Pytest fixture for fake Redis"""
    redis_client = fakeredis.aioredis.FakeRedis()
    yield redis_client
    await redis_client.close()

@pytest.fixture
async def test_cache_manager(fake_redis):
    """Pytest fixture for test cache manager"""
    from app.services.cache_service import CacheService as AdvancedCacheManager
    return AdvancedCacheManager(redis_client=fake_redis)

@pytest.fixture
async def test_knowledge_cache(test_cache_manager):
    """Pytest fixture for test knowledge cache"""
    from app.services.intelligent_cache import OptimizedKnowledgeCache
    cache = OptimizedKnowledgeCache(test_cache_manager)
    await cache.initialize()
    return cache

@pytest.fixture
async def test_emotional_cache(test_cache_manager):
    """Pytest fixture for test emotional cache"""
    from app.services.intelligent_cache import OptimizedEmotionalCache
    cache = OptimizedEmotionalCache(test_cache_manager)
    await cache.initialize()
    return cache

@pytest.fixture
def spiritual_test_data():
    """Pytest fixture for spiritual test data"""
    return test_framework.spiritual_test_data

@pytest.fixture
def user_test_data():
    """Pytest fixture for user test data"""
    return test_framework.user_test_data

# Export for use by test modules
__all__ = [
    'DharmaMindTestFramework',
    'TestResult',
    'TestSuiteReport',
    'test_framework',
    'fake_redis',
    'test_cache_manager',
    'test_knowledge_cache',
    'test_emotional_cache',
    'spiritual_test_data',
    'user_test_data'
]