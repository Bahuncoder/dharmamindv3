# 🧪 Comprehensive Testing Framework
# Advanced testing suite for DharmaMind production

import os
import pytest
import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
import redis
from fastapi.testclient import TestClient
from httpx import AsyncClient
import structlog

# Test Configuration
@dataclass
class TestConfig:
    """Testing configuration and thresholds"""
    
    # Performance thresholds
    MAX_RESPONSE_TIME: float = 2.0
    MAX_AI_RESPONSE_TIME: float = 30.0
    MIN_THROUGHPUT_RPS: int = 100
    
    # Load testing parameters
    CONCURRENT_USERS: int = 50
    TEST_DURATION_SECONDS: int = 300
    RAMP_UP_TIME: int = 60
    
    # Database testing
    DB_CONNECTION_TIMEOUT: int = 5
    MAX_DB_CONNECTIONS: int = 100
    
    # AI testing
    AI_QUALITY_THRESHOLD: float = 0.7
    MAX_TOKEN_LIMIT: int = 4000
    
    # Security testing
    MAX_FAILED_LOGIN_ATTEMPTS: int = 5
    JWT_EXPIRY_HOURS: int = 24
    
    # Test data
    TEST_USERS_COUNT: int = 100
    SPIRITUAL_PATHS = ['karma_yoga', 'bhakti_yoga', 'raja_yoga', 'jnana_yoga']

# ================================
# 🔧 TEST FIXTURES AND UTILITIES
# ================================
@pytest.fixture
async def test_client():
    """FastAPI test client fixture"""
    from backend.app.main import app
    
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

@pytest.fixture
async def redis_client():
    """Redis test client fixture"""
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_TEST_DB', 1)),  # Use test database
        decode_responses=True
    )
    
    # Clear test database
    await redis_client.flushdb()
    
    yield redis_client
    
    # Cleanup
    await redis_client.flushdb()
    await redis_client.close()

@pytest.fixture
def mock_ai_service():
    """Mock AI service for testing"""
    mock_service = Mock()
    mock_service.generate_response = AsyncMock(return_value={
        'response': 'Mock spiritual guidance response',
        'tokens_used': {'prompt': 100, 'completion': 200},
        'quality_score': 0.85,
        'spiritual_context': 'karma_yoga'
    })
    return mock_service

class TestDataGenerator:
    """Generate realistic test data"""
    
    @staticmethod
    def generate_user_data(count: int = 100) -> List[Dict[str, Any]]:
        """Generate test user data"""
        users = []
        
        for i in range(count):
            user = {
                'user_id': f'test_user_{i}',
                'email': f'test{i}@dharmamind.com',
                'spiritual_path': random.choice(TestConfig.SPIRITUAL_PATHS),
                'experience_level': random.choice(['beginner', 'intermediate', 'advanced']),
                'preferences': {
                    'language': random.choice(['english', 'sanskrit', 'hindi']),
                    'guidance_style': random.choice(['gentle', 'direct', 'detailed']),
                    'meditation_time': random.randint(5, 60)
                },
                'created_at': time.time() - random.randint(0, 86400 * 30)  # Last 30 days
            }
            users.append(user)
        
        return users
    
    @staticmethod
    def generate_spiritual_questions() -> List[str]:
        """Generate diverse spiritual questions for testing"""
        return [
            "How can I find inner peace in stressful times?",
            "What is the meaning of dharma in daily life?",
            "How do I practice meditation for beginners?",
            "What are the principles of karma yoga?",
            "How can I develop devotion in bhakti yoga?",
            "What is the path to self-realization?",
            "How do I overcome negative thoughts?",
            "What is the significance of Om meditation?",
            "How can I balance work and spiritual practice?",
            "What are the stages of consciousness?",
            "How do I understand the Bhagavad Gita teachings?",
            "What is the difference between dharma and adharma?",
            "How can I cultivate compassion?",
            "What is the role of a guru in spiritual growth?",
            "How do I practice mindfulness in daily activities?"
        ]

# ================================
# 🧪 UNIT TESTS
# ================================
class TestSpiritualCore:
    """Test spiritual guidance core functionality"""
    
    @pytest.mark.asyncio
    async def test_spiritual_response_generation(self, mock_ai_service):
        """Test spiritual response generation"""
        from backend.app.chakra_modules.dharma_engine import DharmaEngine
        
        engine = DharmaEngine()
        engine.ai_service = mock_ai_service
        
        question = "How can I find inner peace?"
        user_context = {
            'spiritual_path': 'karma_yoga',
            'experience_level': 'beginner'
        }
        
        response = await engine.generate_spiritual_guidance(question, user_context)
        
        assert response is not None
        assert 'response' in response
        assert 'spiritual_context' in response
        assert response['quality_score'] >= TestConfig.AI_QUALITY_THRESHOLD
        
        mock_ai_service.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_personalization_engine(self):
        """Test personalization system"""
        from backend.app.services.personalization_engine import PersonalizationEngine
        
        engine = PersonalizationEngine()
        
        user_profile = {
            'user_id': 'test_user',
            'spiritual_path': 'bhakti_yoga',
            'experience_level': 'intermediate',
            'preferences': {'guidance_style': 'gentle'}
        }
        
        raw_response = "This is a generic spiritual response."
        
        personalized = await engine.personalize_response(raw_response, user_profile)
        
        assert personalized != raw_response
        assert len(personalized) > 0
    
    @pytest.mark.asyncio
    async def test_practice_recommendations(self):
        """Test practice recommendation engine"""
        from backend.app.services.practice_recommendation_engine import PracticeRecommendationEngine
        
        engine = PracticeRecommendationEngine()
        
        user_profile = {
            'spiritual_path': 'raja_yoga',
            'experience_level': 'beginner',
            'available_time': 30,
            'meditation_experience': False
        }
        
        recommendations = await engine.get_recommendations(user_profile)
        
        assert len(recommendations) > 0
        assert all('practice_type' in rec for rec in recommendations)
        assert all('duration' in rec for rec in recommendations)
        assert all(rec['duration'] <= 30 for rec in recommendations)
    
    def test_consciousness_core(self):
        """Test consciousness processing"""
        from backend.app.chakra_modules.consciousness_core import ConsciousnessCore
        
        core = ConsciousnessCore()
        
        input_data = {
            'query': 'What is consciousness?',
            'user_state': 'seeking',
            'context': 'philosophical_inquiry'
        }
        
        processed = core.process_consciousness_query(input_data)
        
        assert 'consciousness_level' in processed
        assert 'awakeness_indicators' in processed
        assert 'guidance_direction' in processed

class TestAuthentication:
    """Test authentication and security"""
    
    @pytest.mark.asyncio
    async def test_user_registration(self, test_client: AsyncClient):
        """Test user registration"""
        user_data = {
            'email': 'test@dharmamind.com',
            'password': 'SecurePassword123!',
            'spiritual_path': 'karma_yoga'
        }
        
        response = await test_client.post('/auth/register', json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert 'access_token' in data
        assert 'user_id' in data
    
    @pytest.mark.asyncio
    async def test_user_login(self, test_client: AsyncClient):
        """Test user login"""
        # First register a user
        user_data = {
            'email': 'login_test@dharmamind.com',
            'password': 'SecurePassword123!',
            'spiritual_path': 'bhakti_yoga'
        }
        
        await test_client.post('/auth/register', json=user_data)
        
        # Then test login
        login_data = {
            'email': 'login_test@dharmamind.com',
            'password': 'SecurePassword123!'
        }
        
        response = await test_client.post('/auth/login', json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data
    
    @pytest.mark.asyncio
    async def test_failed_login_rate_limiting(self, test_client: AsyncClient):
        """Test rate limiting on failed logins"""
        login_data = {
            'email': 'nonexistent@dharmamind.com',
            'password': 'WrongPassword'
        }
        
        # Attempt multiple failed logins
        failed_attempts = 0
        for _ in range(TestConfig.MAX_FAILED_LOGIN_ATTEMPTS + 2):
            response = await test_client.post('/auth/login', json=login_data)
            if response.status_code == 429:  # Rate limited
                break
            failed_attempts += 1
        
        assert failed_attempts <= TestConfig.MAX_FAILED_LOGIN_ATTEMPTS
    
    @pytest.mark.asyncio
    async def test_jwt_token_validation(self, test_client: AsyncClient):
        """Test JWT token validation"""
        # Register and login to get token
        user_data = {
            'email': 'jwt_test@dharmamind.com',
            'password': 'SecurePassword123!',
            'spiritual_path': 'jnana_yoga'
        }
        
        await test_client.post('/auth/register', json=user_data)
        login_response = await test_client.post('/auth/login', json={
            'email': user_data['email'],
            'password': user_data['password']
        })
        
        token = login_response.json()['access_token']
        
        # Test protected endpoint with valid token
        headers = {'Authorization': f'Bearer {token}'}
        response = await test_client.get('/user/profile', headers=headers)
        
        assert response.status_code == 200
        
        # Test with invalid token
        invalid_headers = {'Authorization': 'Bearer invalid_token'}
        response = await test_client.get('/user/profile', headers=invalid_headers)
        
        assert response.status_code == 401

# ================================
# 🔗 INTEGRATION TESTS
# ================================
class TestAPIIntegration:
    """Test complete API workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_spiritual_guidance_flow(self, test_client: AsyncClient):
        """Test complete spiritual guidance workflow"""
        
        # 1. Register user
        user_data = {
            'email': 'integration_test@dharmamind.com',
            'password': 'SecurePassword123!',
            'spiritual_path': 'karma_yoga'
        }
        
        register_response = await test_client.post('/auth/register', json=user_data)
        assert register_response.status_code == 201
        
        token = register_response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        
        # 2. Ask spiritual question
        question_data = {
            'question': 'How can I practice karma yoga in my daily work?',
            'context': 'workplace_spirituality'
        }
        
        question_response = await test_client.post(
            '/chat/spiritual-guidance', 
            json=question_data, 
            headers=headers
        )
        
        assert question_response.status_code == 200
        guidance = question_response.json()
        
        assert 'response' in guidance
        assert 'spiritual_insights' in guidance
        assert len(guidance['response']) > 0
        
        # 3. Get practice recommendations
        recommendations_response = await test_client.get(
            '/practices/recommendations', 
            headers=headers
        )
        
        assert recommendations_response.status_code == 200
        recommendations = recommendations_response.json()
        
        assert 'practices' in recommendations
        assert len(recommendations['practices']) > 0
        
        # 4. Update user progress
        progress_data = {
            'practice_completed': 'daily_karma_reflection',
            'duration_minutes': 15,
            'quality_rating': 4,
            'insights': 'Felt more mindful at work today'
        }
        
        progress_response = await test_client.post(
            '/user/progress', 
            json=progress_data, 
            headers=headers
        )
        
        assert progress_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_ai_model_integration(self, test_client: AsyncClient):
        """Test AI model integration and failover"""
        
        # Setup authenticated user
        token = await self._get_test_token(test_client)
        headers = {'Authorization': f'Bearer {token}'}
        
        # Test different types of spiritual questions
        test_questions = [
            {'question': 'Explain the concept of dharma', 'expected_context': 'philosophy'},
            {'question': 'How do I meditate?', 'expected_context': 'practice'},
            {'question': 'What is moksha?', 'expected_context': 'liberation'},
        ]
        
        for test_case in test_questions:
            response = await test_client.post(
                '/chat/spiritual-guidance',
                json={'question': test_case['question']},
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify AI response quality
            assert len(data['response']) > 50  # Meaningful response length
            assert 'spiritual_insights' in data
            assert data.get('quality_score', 0) >= TestConfig.AI_QUALITY_THRESHOLD
    
    async def _get_test_token(self, test_client: AsyncClient) -> str:
        """Helper to get authentication token"""
        user_data = {
            'email': f'test_{int(time.time())}@dharmamind.com',
            'password': 'SecurePassword123!',
            'spiritual_path': 'karma_yoga'
        }
        
        response = await test_client.post('/auth/register', json=user_data)
        return response.json()['access_token']

class TestDatabaseIntegration:
    """Test database operations and consistency"""
    
    @pytest.mark.asyncio
    async def test_user_data_persistence(self, test_client: AsyncClient):
        """Test user data persistence across sessions"""
        
        # Create user
        user_data = {
            'email': 'persistence_test@dharmamind.com',
            'password': 'SecurePassword123!',
            'spiritual_path': 'raja_yoga'
        }
        
        register_response = await test_client.post('/auth/register', json=user_data)
        user_id = register_response.json()['user_id']
        
        # Login and get profile
        login_response = await test_client.post('/auth/login', json={
            'email': user_data['email'],
            'password': user_data['password']
        })
        
        token = login_response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        
        profile_response = await test_client.get('/user/profile', headers=headers)
        profile = profile_response.json()
        
        assert profile['user_id'] == user_id
        assert profile['spiritual_path'] == 'raja_yoga'
        
        # Update profile
        update_data = {
            'experience_level': 'intermediate',
            'meditation_minutes_daily': 30
        }
        
        update_response = await test_client.put(
            '/user/profile', 
            json=update_data, 
            headers=headers
        )
        assert update_response.status_code == 200
        
        # Verify persistence
        updated_profile_response = await test_client.get('/user/profile', headers=headers)
        updated_profile = updated_profile_response.json()
        
        assert updated_profile['experience_level'] == 'intermediate'
        assert updated_profile['meditation_minutes_daily'] == 30
    
    @pytest.mark.asyncio
    async def test_chat_history_storage(self, test_client: AsyncClient):
        """Test chat history storage and retrieval"""
        
        token = await TestAPIIntegration()._get_test_token(test_client)
        headers = {'Authorization': f'Bearer {token}'}
        
        # Send multiple messages
        questions = [
            'What is karma?',
            'How do I practice compassion?',
            'Explain the concept of ahimsa'
        ]
        
        for question in questions:
            await test_client.post(
                '/chat/spiritual-guidance',
                json={'question': question},
                headers=headers
            )
        
        # Retrieve chat history
        history_response = await test_client.get('/chat/history', headers=headers)
        assert history_response.status_code == 200
        
        history = history_response.json()
        assert len(history['conversations']) >= len(questions)
        
        # Verify questions are stored
        stored_questions = [conv['user_message'] for conv in history['conversations']]
        for question in questions:
            assert question in stored_questions

# ================================
# ⚡ PERFORMANCE TESTS
# ================================
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_response_time_benchmarks(self, test_client: AsyncClient):
        """Test API response time benchmarks"""
        
        token = await TestAPIIntegration()._get_test_token(test_client)
        headers = {'Authorization': f'Bearer {token}'}
        
        endpoints_to_test = [
            ('GET', '/user/profile'),
            ('GET', '/practices/recommendations'),
            ('POST', '/chat/spiritual-guidance', {'question': 'What is dharma?'}),
        ]
        
        for method, endpoint, *data in endpoints_to_test:
            start_time = time.time()
            
            if method == 'GET':
                response = await test_client.get(endpoint, headers=headers)
            else:
                response = await test_client.post(endpoint, json=data[0], headers=headers)
            
            duration = time.time() - start_time
            
            assert response.status_code in [200, 201]
            assert duration < TestConfig.MAX_RESPONSE_TIME, f"{endpoint} took {duration}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_users(self, test_client: AsyncClient):
        """Test concurrent user handling"""
        
        async def simulate_user_session(user_id: int):
            """Simulate a single user session"""
            try:
                # Register user
                user_data = {
                    'email': f'concurrent_user_{user_id}@dharmamind.com',
                    'password': 'SecurePassword123!',
                    'spiritual_path': random.choice(TestConfig.SPIRITUAL_PATHS)
                }
                
                register_response = await test_client.post('/auth/register', json=user_data)
                if register_response.status_code != 201:
                    return False
                
                token = register_response.json()['access_token']
                headers = {'Authorization': f'Bearer {token}'}
                
                # Ask questions
                questions = TestDataGenerator.generate_spiritual_questions()
                for question in random.sample(questions, 3):  # Ask 3 random questions
                    response = await test_client.post(
                        '/chat/spiritual-guidance',
                        json={'question': question},
                        headers=headers
                    )
                    
                    if response.status_code != 200:
                        return False
                
                return True
                
            except Exception as e:
                print(f"User {user_id} session failed: {e}")
                return False
        
        # Run concurrent user sessions
        tasks = [simulate_user_session(i) for i in range(TestConfig.CONCURRENT_USERS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate success rate
        successful_sessions = sum(1 for result in results if result is True)
        success_rate = successful_sessions / len(results)
        
        assert success_rate >= 0.95, f"Success rate {success_rate} below threshold"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory efficiency"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate heavy usage
        test_data = []
        for i in range(1000):
            test_data.append({
                'user_id': f'user_{i}',
                'data': 'x' * 1000,  # 1KB per entry
                'timestamp': time.time()
            })
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Cleanup
        del test_data
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should be released after cleanup
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50, f"Memory leak detected: {memory_increase}MB"

# ================================
# 🛡️ SECURITY TESTS
# ================================
class TestSecurity:
    """Security testing suite"""
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, test_client: AsyncClient):
        """Test SQL injection attack protection"""
        
        # Attempt SQL injection in registration
        malicious_data = {
            'email': "'; DROP TABLE users; --",
            'password': 'password',
            'spiritual_path': 'karma_yoga'
        }
        
        response = await test_client.post('/auth/register', json=malicious_data)
        
        # Should either reject with 400 or sanitize the input
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_xss_protection(self, test_client: AsyncClient):
        """Test XSS attack protection"""
        
        token = await TestAPIIntegration()._get_test_token(test_client)
        headers = {'Authorization': f'Bearer {token}'}
        
        # Attempt XSS in spiritual question
        malicious_question = {
            'question': '<script>alert("XSS")</script>What is dharma?'
        }
        
        response = await test_client.post(
            '/chat/spiritual-guidance',
            json=malicious_question,
            headers=headers
        )
        
        if response.status_code == 200:
            # Response should not contain unescaped script tags
            response_data = response.json()
            assert '<script>' not in response_data.get('response', '')
    
    @pytest.mark.asyncio
    async def test_authentication_bypass(self, test_client: AsyncClient):
        """Test authentication bypass attempts"""
        
        # Try to access protected endpoints without token
        protected_endpoints = [
            '/user/profile',
            '/chat/spiritual-guidance',
            '/practices/recommendations'
        ]
        
        for endpoint in protected_endpoints:
            response = await test_client.get(endpoint)
            assert response.status_code == 401, f"{endpoint} allowed unauthenticated access"
        
        # Try with invalid token
        invalid_headers = {'Authorization': 'Bearer invalid_token_12345'}
        for endpoint in protected_endpoints:
            response = await test_client.get(endpoint, headers=invalid_headers)
            assert response.status_code == 401, f"{endpoint} accepted invalid token"
    
    @pytest.mark.asyncio
    async def test_data_encryption(self):
        """Test sensitive data encryption"""
        from backend.app.services.security_framework import EncryptionService
        
        encryption_service = EncryptionService()
        
        sensitive_data = "User's spiritual journey details"
        
        # Test encryption
        encrypted = encryption_service.encrypt_data(sensitive_data)
        assert encrypted != sensitive_data
        assert len(encrypted) > 0
        
        # Test decryption
        decrypted = encryption_service.decrypt_data(encrypted)
        assert decrypted == sensitive_data
        
        # Test password hashing
        password = "SecurePassword123!"
        hashed = encryption_service.hash_password(password)
        
        assert hashed != password
        assert encryption_service.verify_password(password, hashed) is True
        assert encryption_service.verify_password("wrong_password", hashed) is False

# ================================
# 📊 MONITORING TESTS
# ================================
class TestMonitoring:
    """Test monitoring and observability"""
    
    def test_metrics_collection(self):
        """Test Prometheus metrics collection"""
        from backend.app.monitoring.monitoring_system import PrometheusMetrics
        
        metrics = PrometheusMetrics()
        
        # Test HTTP metrics
        metrics.http_requests_total.labels(
            method='POST', 
            endpoint='/chat/spiritual-guidance', 
            status_code=200
        ).inc()
        
        # Test AI metrics
        metrics.ai_requests_total.labels(model='gpt-4', status='success').inc()
        metrics.ai_response_time.labels(model='gpt-4').observe(2.5)
        
        # Verify metrics are recorded (basic check)
        assert metrics.http_requests_total
        assert metrics.ai_requests_total
    
    @pytest.mark.asyncio
    async def test_health_checks(self, redis_client):
        """Test system health monitoring"""
        from backend.app.monitoring.monitoring_system import HealthMonitor, PrometheusMetrics
        
        metrics = PrometheusMetrics()
        health_monitor = HealthMonitor(redis_client, metrics)
        
        # Register test health checks
        async def mock_database_check():
            return {'status': 'healthy', 'details': {'connection': 'ok'}}
        
        await health_monitor.register_health_check('test_db', mock_database_check)
        
        # Run health checks
        results = await health_monitor.run_health_checks()
        
        assert 'overall_status' in results
        assert 'checks' in results
        assert 'test_db' in results['checks']
    
    @pytest.mark.asyncio
    async def test_logging_integration(self):
        """Test structured logging"""
        from backend.app.monitoring.monitoring_system import StructuredLogger
        
        logger = StructuredLogger("test_service")
        
        # Test different log levels
        await logger.log_ai_interaction(
            model='test-model',
            prompt_tokens=100,
            completion_tokens=200,
            duration=1.5,
            quality_score=0.85,
            user_id='test_user',
            spiritual_path='karma_yoga'
        )
        
        await logger.log_error(
            error_type='test_error',
            error_message='Test error message',
            component='test_component',
            user_id='test_user'
        )
        
        # Basic verification that logging doesn't crash
        assert logger.logger is not None

# ================================
# 🎯 TEST RUNNERS AND UTILITIES
# ================================
class TestRunner:
    """Test execution and reporting utilities"""
    
    @staticmethod
    async def run_comprehensive_test_suite():
        """Run complete test suite with reporting"""
        
        print("🧪 Starting DharmaMind Comprehensive Test Suite")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Unit Tests", [
                TestSpiritualCore,
                TestAuthentication
            ]),
            ("Integration Tests", [
                TestAPIIntegration,
                TestDatabaseIntegration
            ]),
            ("Performance Tests", [
                TestPerformance
            ]),
            ("Security Tests", [
                TestSecurity
            ]),
            ("Monitoring Tests", [
                TestMonitoring
            ])
        ]
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category_name, test_classes in test_categories:
            print(f"\n📋 Running {category_name}")
            print("-" * 40)
            
            for test_class in test_classes:
                class_name = test_class.__name__
                print(f"  🔍 {class_name}")
                
                # Run pytest programmatically for this test class
                result = pytest.main([
                    f"-v",
                    f"--tb=short",
                    f"-k", class_name
                ])
                
                # Simple result tracking (pytest returns 0 for success)
                if result == 0:
                    print(f"    ✅ {class_name} - PASSED")
                    passed_tests += 1
                else:
                    print(f"    ❌ {class_name} - FAILED")
                    failed_tests += 1
                
                total_tests += 1
        
        # Test summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("-" * 20)
        print(f"Total Test Categories: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! DharmaMind is ready for beta launch!")
        else:
            print(f"\n⚠️  {failed_tests} test categories failed. Review and fix issues.")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests/total_tests)*100
        }

# ================================
# 🚀 TEST EXECUTION
# ================================
if __name__ == "__main__":
    # Run comprehensive test suite
    asyncio.run(TestRunner.run_comprehensive_test_suite())
