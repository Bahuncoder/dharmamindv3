"""
🚀 Load Testing Suite
====================

Performance and load testing for DharmaMind backend:
- API endpoint load testing
- Database performance under load
- Concurrent user simulation
- Memory and resource monitoring
- Spiritual AI performance testing
"""

import pytest
import asyncio
import aiohttp
import time
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Load testing configuration
LOAD_TEST_CONFIG = {
    'light_load': {'users': 10, 'requests_per_user': 5, 'duration': 10},
    'medium_load': {'users': 50, 'requests_per_user': 10, 'duration': 30},
    'heavy_load': {'users': 100, 'requests_per_user': 20, 'duration': 60},
    'stress_test': {'users': 200, 'requests_per_user': 50, 'duration': 120}
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'max_response_time': 2.0,  # seconds
    'p95_response_time': 1.0,  # seconds
    'min_success_rate': 0.95,  # 95%
    'max_memory_usage': 500,   # MB
    'max_cpu_usage': 80        # percentage
}


@pytest.mark.load
@pytest.mark.slow
class TestAPILoadTesting:
    """Test API endpoints under various load conditions."""
    
    @pytest.fixture
    async def load_test_session(self):
        """Create HTTP session for load testing."""
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        ) as session:
            yield session
    
    async def make_request(self, session: aiohttp.ClientSession, url: str, 
                          method: str = 'GET', headers: Dict = None, 
                          json_data: Dict = None) -> Dict[str, Any]:
        """Make a single HTTP request and record metrics."""
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                async with session.get(url, headers=headers) as response:
                    content = await response.text()
                    end_time = time.time()
                    
                    return {
                        'status_code': response.status,
                        'response_time': end_time - start_time,
                        'success': response.status < 400,
                        'content_length': len(content),
                        'timestamp': datetime.utcnow()
                    }
            
            elif method.upper() == 'POST':
                async with session.post(url, headers=headers, json=json_data) as response:
                    content = await response.text()
                    end_time = time.time()
                    
                    return {
                        'status_code': response.status,
                        'response_time': end_time - start_time,
                        'success': response.status < 400,
                        'content_length': len(content),
                        'timestamp': datetime.utcnow()
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                'status_code': 0,
                'response_time': end_time - start_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }
    
    async def simulate_user_session(self, session: aiohttp.ClientSession, 
                                   user_id: int, config: Dict) -> List[Dict]:
        """Simulate a complete user session."""
        base_url = "http://localhost:8000/api"  # Adjust for your setup
        results = []
        
        # User authentication
        auth_data = {
            "email": f"loadtest_user_{user_id}@example.com",
            "password": "test_password"
        }
        
        auth_result = await self.make_request(
            session, f"{base_url}/auth/login", 
            method='POST', json_data=auth_data
        )
        results.append({**auth_result, 'endpoint': 'auth/login', 'user_id': user_id})
        
        # If auth failed, skip rest of session
        if not auth_result['success']:
            return results
        
        # Extract token (simplified)
        headers = {'Authorization': f'Bearer mock_token_{user_id}'}
        
        # Chat session creation
        chat_data = {
            "title": f"Load Test Session {user_id}",
            "spiritual_focus": "mindfulness"
        }
        
        chat_result = await self.make_request(
            session, f"{base_url}/chat/sessions", 
            method='POST', headers=headers, json_data=chat_data
        )
        results.append({**chat_result, 'endpoint': 'chat/sessions', 'user_id': user_id})
        
        # Multiple chat messages
        for msg_num in range(config.get('requests_per_user', 5)):
            message_data = {
                "content": f"Load test message {msg_num} from user {user_id}",
                "session_id": f"session_{user_id}"
            }
            
            msg_result = await self.make_request(
                session, f"{base_url}/chat/messages",
                method='POST', headers=headers, json_data=message_data
            )
            results.append({**msg_result, 'endpoint': 'chat/messages', 'user_id': user_id})
            
            # Small delay between messages
            await asyncio.sleep(0.1)
        
        # Get user profile
        profile_result = await self.make_request(
            session, f"{base_url}/users/profile", headers=headers
        )
        results.append({**profile_result, 'endpoint': 'users/profile', 'user_id': user_id})
        
        return results
    
    @pytest.mark.parametrize("load_level", ["light_load", "medium_load"])
    async def test_api_load_performance(self, load_test_session, load_level):
        """Test API performance under different load levels."""
        config = LOAD_TEST_CONFIG[load_level]
        
        print(f"\n🚀 Running {load_level} test:")
        print(f"   Users: {config['users']}")
        print(f"   Requests per user: {config['requests_per_user']}")
        print(f"   Duration: {config['duration']}s")
        
        # Record system metrics before test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start load test
        start_time = time.time()
        
        # Create user session tasks
        tasks = []
        for user_id in range(config['users']):
            task = self.simulate_user_session(
                load_test_session, user_id, config
            )
            tasks.append(task)
        
        # Execute load test
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Flatten results
        flat_results = []
        for user_results in all_results:
            if isinstance(user_results, list):
                flat_results.extend(user_results)
        
        # Calculate metrics
        metrics = self.calculate_load_test_metrics(flat_results, total_duration)
        
        # Record system metrics after test
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Print results
        self.print_load_test_results(metrics, memory_increase)
        
        # Assert performance requirements
        assert metrics['success_rate'] >= PERFORMANCE_THRESHOLDS['min_success_rate'], \
            f"Success rate {metrics['success_rate']:.2%} below threshold"
        
        assert metrics['p95_response_time'] <= PERFORMANCE_THRESHOLDS['p95_response_time'], \
            f"P95 response time {metrics['p95_response_time']:.3f}s above threshold"
        
        assert metrics['max_response_time'] <= PERFORMANCE_THRESHOLDS['max_response_time'], \
            f"Max response time {metrics['max_response_time']:.3f}s above threshold"
    
    def calculate_load_test_metrics(self, results: List[Dict], 
                                   total_duration: float) -> Dict[str, Any]:
        """Calculate comprehensive load test metrics.""" 
        successful_requests = [r for r in results if r.get('success', False)]
        failed_requests = [r for r in results if not r.get('success', False)]
        
        response_times = [r['response_time'] for r in results if 'response_time' in r]
        successful_response_times = [r['response_time'] for r in successful_requests]
        
        return {
            'total_requests': len(results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(results) if results else 0,
            
            # Response time metrics
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'p95_response_time': self.percentile(response_times, 95) if response_times else 0,
            'p99_response_time': self.percentile(response_times, 99) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            
            # Throughput metrics
            'requests_per_second': len(results) / total_duration if total_duration > 0 else 0,
            'successful_rps': len(successful_requests) / total_duration if total_duration > 0 else 0,
            
            # Duration
            'total_duration': total_duration,
            
            # Error breakdown
            'error_types': self.categorize_errors(failed_requests),
            
            # Endpoint performance
            'endpoint_metrics': self.calculate_endpoint_metrics(results)
        }
    
    def percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return (sorted_data[lower_index] * (1 - weight) + 
                   sorted_data[upper_index] * weight)
    
    def categorize_errors(self, failed_requests: List[Dict]) -> Dict[str, int]:
        """Categorize types of errors."""
        error_types = {}
        
        for request in failed_requests:
            if 'error' in request:
                error_type = type(request.get('error', 'Unknown')).__name__
            else:
                status = request.get('status_code', 0)
                if status >= 500:
                    error_type = 'Server Error (5xx)'
                elif status >= 400:
                    error_type = 'Client Error (4xx)'
                else:
                    error_type = 'Unknown Error'
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return error_types
    
    def calculate_endpoint_metrics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate per-endpoint performance metrics."""
        endpoint_data = {}
        
        for result in results:
            endpoint = result.get('endpoint', 'unknown')
            
            if endpoint not in endpoint_data:
                endpoint_data[endpoint] = []
            
            endpoint_data[endpoint].append(result)
        
        endpoint_metrics = {}
        for endpoint, requests in endpoint_data.items():
            successful = [r for r in requests if r.get('success', False)]
            response_times = [r['response_time'] for r in requests if 'response_time' in r]
            
            endpoint_metrics[endpoint] = {
                'total_requests': len(requests),
                'successful_requests': len(successful),
                'success_rate': len(successful) / len(requests) if requests else 0,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'p95_response_time': self.percentile(response_times, 95) if response_times else 0
            }
        
        return endpoint_metrics
    
    def print_load_test_results(self, metrics: Dict[str, Any], memory_increase: float):
        """Print comprehensive load test results."""
        print(f"\n📊 Load Test Results:")
        print(f"   Total Requests: {metrics['total_requests']}")
        print(f"   Successful: {metrics['successful_requests']} ({metrics['success_rate']:.2%})")
        print(f"   Failed: {metrics['failed_requests']}")
        print(f"   Duration: {metrics['total_duration']:.2f}s")
        print(f"   Requests/sec: {metrics['requests_per_second']:.2f}")
        
        print(f"\n⏱️  Response Time Metrics:")
        print(f"   Average: {metrics['avg_response_time']:.3f}s")
        print(f"   Median: {metrics['median_response_time']:.3f}s")
        print(f"   P95: {metrics['p95_response_time']:.3f}s")
        print(f"   P99: {metrics['p99_response_time']:.3f}s")
        print(f"   Max: {metrics['max_response_time']:.3f}s")
        
        print(f"\n🔧 System Impact:")
        print(f"   Memory Increase: {memory_increase:.1f} MB")
        
        if metrics['error_types']:
            print(f"\n❌ Error Breakdown:")
            for error_type, count in metrics['error_types'].items():
                print(f"   {error_type}: {count}")
        
        print(f"\n🎯 Endpoint Performance:")
        for endpoint, endpoint_metrics in metrics['endpoint_metrics'].items():
            print(f"   {endpoint}:")
            print(f"     Requests: {endpoint_metrics['total_requests']}")
            print(f"     Success Rate: {endpoint_metrics['success_rate']:.2%}")
            print(f"     Avg Response: {endpoint_metrics['avg_response_time']:.3f}s")


@pytest.mark.load
class TestDatabaseLoadTesting:
    """Test database performance under load."""
    
    async def test_concurrent_database_operations(self, db_session, benchmark_timer):
        """Test database performance with concurrent operations."""
        timer = benchmark_timer()
        
        async def create_user_with_chat_data(user_id: int):
            """Create user with associated chat data."""
            from app.models.user import User
            from app.models.chat import ChatSession, ChatMessage
            
            # Create user
            user = User(
                email=f"concurrent_user_{user_id}@example.com",
                username=f"concurrent_user_{user_id}",
                hashed_password="test_password",
                full_name=f"Concurrent User {user_id}"
            )
            db_session.add(user)
            
            # Create chat session
            session = ChatSession(
                user_id=user.id,
                title=f"Concurrent Session {user_id}",
                spiritual_focus="mindfulness"
            )
            db_session.add(session)
            
            # Create messages
            messages = []
            for i in range(5):
                message = ChatMessage(
                    session_id=session.id,
                    user_id=user.id,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Concurrent message {i} from user {user_id}"
                )
                messages.append(message)
            
            db_session.add_all(messages)
            return user_id
        
        # Execute concurrent database operations
        timer.start()
        
        tasks = [create_user_with_chat_data(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        await db_session.commit()
        execution_time = timer.stop()
        
        # Verify results
        successful_operations = [r for r in results if not isinstance(r, Exception)]
        
        print(f"\n💾 Database Load Test Results:")
        print(f"   Concurrent Operations: 20")
        print(f"   Successful: {len(successful_operations)}")
        print(f"   Execution Time: {execution_time:.3f}s")
        print(f"   Operations/sec: {len(successful_operations) / execution_time:.2f}")
        
        # Performance assertions
        assert len(successful_operations) == 20
        assert execution_time < 5.0  # Should complete within 5 seconds
    
    async def test_database_query_performance_under_load(self, db_session, test_user):
        """Test query performance with large datasets."""
        # Create substantial test data
        from app.models.chat import ChatSession, ChatMessage
        
        # Create many chat sessions
        sessions = []
        for i in range(50):
            session = ChatSession(
                user_id=test_user.id,
                title=f"Performance Session {i}",
                spiritual_focus=f"focus_{i % 5}",
                is_active=True
            )
            db_session.add(session)
            sessions.append(session)
        
        await db_session.commit()
        
        # Add messages to sessions
        all_messages = []
        for session in sessions:
            await db_session.refresh(session)
            for j in range(20):  # 20 messages per session = 1000 total
                message = ChatMessage(
                    session_id=session.id,
                    user_id=test_user.id,
                    role="user" if j % 2 == 0 else "assistant",
                    content=f"Performance message {j} in session {session.id}",
                    metadata={"performance_test": True, "index": j}
                )
                all_messages.append(message)
        
        db_session.add_all(all_messages)
        await db_session.commit()
        
        # Test various query patterns under load
        async def query_user_sessions():
            from sqlalchemy import select, func
            result = await db_session.execute(
                select(ChatSession)
                .where(ChatSession.user_id == test_user.id)
                .order_by(ChatSession.created_at.desc())
                .limit(20)
            )
            return result.scalars().all()
        
        async def query_session_statistics():
            from sqlalchemy import select, func
            result = await db_session.execute(
                select(
                    ChatSession.spiritual_focus,
                    func.count(ChatMessage.id).label('message_count'),
                    func.avg(func.length(ChatMessage.content)).label('avg_length')
                )
                .join(ChatMessage, ChatSession.id == ChatMessage.session_id)
                .where(ChatSession.user_id == test_user.id)
                .group_by(ChatSession.spiritual_focus)
            )
            return result.all()
        
        async def query_recent_messages():
            from sqlalchemy import select
            result = await db_session.execute(
                select(ChatMessage)
                .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                .where(ChatSession.user_id == test_user.id)
                .order_by(ChatMessage.created_at.desc())
                .limit(100)
            )
            return result.scalars().all()
        
        # Execute concurrent queries
        start_time = time.time()
        
        query_tasks = []
        for _ in range(10):  # 10 concurrent query sets
            query_tasks.extend([
                query_user_sessions(),
                query_session_statistics(),
                query_recent_messages()
            ])
        
        query_results = await asyncio.gather(*query_tasks)
        end_time = time.time()
        
        query_execution_time = end_time - start_time
        
        print(f"\n🔍 Database Query Load Test:")
        print(f"   Total Queries: {len(query_tasks)}")
        print(f"   Execution Time: {query_execution_time:.3f}s")
        print(f"   Queries/sec: {len(query_tasks) / query_execution_time:.2f}")
        print(f"   Data Scale: 50 sessions, 1000 messages")
        
        # Performance assertions
        assert query_execution_time < 10.0  # Should complete within 10 seconds
        assert all(result is not None for result in query_results)


@pytest.mark.load
class TestSpiritualAILoadTesting:
    """Test Spiritual AI components under load."""
    
    @pytest.fixture
    def mock_ai_components(self):
        """Mock AI components for load testing."""
        from unittest.mock import AsyncMock, MagicMock
        
        mock_orchestrator = AsyncMock()
        mock_orchestrator.process_spiritual_query.return_value = {
            "response": "Mindfulness brings peace to the present moment.",
            "spiritual_insights": ["awareness", "presence", "compassion"],
            "dharma_teaching": "The path to enlightenment begins with awareness.",
            "meditation_guidance": "Focus on your breath and observe without judgment.",
            "confidence_score": 0.95,
            "processing_time": 0.1
        }
        
        mock_consciousness = MagicMock()
        mock_consciousness.current_state = "contemplative"
        mock_consciousness.awareness_level = 0.8
        
        mock_dharma_engine = AsyncMock()
        mock_dharma_engine.generate_teaching.return_value = {
            "teaching": "Compassion is the foundation of wisdom.",
            "context": "buddhist_philosophy",
            "practice_suggestions": ["loving-kindness meditation", "daily reflection"],
            "confidence": 0.92
        }
        
        return {
            'orchestrator': mock_orchestrator,
            'consciousness': mock_consciousness,
            'dharma_engine': mock_dharma_engine
        }
    
    async def test_spiritual_ai_concurrent_processing(self, mock_ai_components, benchmark_timer):
        """Test AI components handling concurrent spiritual queries."""
        timer = benchmark_timer()
        
        # Simulate various types of spiritual queries
        spiritual_queries = [
            "How can I find inner peace?",
            "What is the meaning of suffering?",
            "How should I practice meditation?",
            "What is compassion?",
            "How can I overcome anger?",
            "What is the path to enlightenment?",
            "How can mindfulness help me?",
            "What is the nature of consciousness?",
            "How do I cultivate wisdom?",
            "What is the purpose of life?"
        ]
        
        async def process_spiritual_query(query: str, query_id: int):
            """Process a single spiritual query."""
            return await mock_ai_components['orchestrator'].process_spiritual_query(
                query=query,
                user_context={"id": query_id, "level": "intermediate"},
                session_context={"spiritual_focus": "mindfulness"}
            )
        
        # Execute concurrent spiritual processing
        timer.start()
        
        tasks = []
        for i in range(50):  # 50 concurrent queries
            query = spiritual_queries[i % len(spiritual_queries)]
            task = process_spiritual_query(query, i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        execution_time = timer.stop()
        
        # Analyze results
        successful_responses = [r for r in results if r and r.get('confidence_score', 0) > 0.8]
        avg_processing_time = sum(r.get('processing_time', 0) for r in results) / len(results)
        
        print(f"\n🧘 Spiritual AI Load Test Results:")
        print(f"   Concurrent Queries: 50")
        print(f"   Successful Responses: {len(successful_responses)}")
        print(f"   Success Rate: {len(successful_responses) / len(results):.2%}")
        print(f"   Total Execution Time: {execution_time:.3f}s")
        print(f"   Average Processing Time: {avg_processing_time:.3f}s")
        print(f"   Queries/sec: {len(results) / execution_time:.2f}")
        
        # Performance assertions
        assert len(successful_responses) >= 45  # 90% success rate
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert avg_processing_time < 0.2  # Average processing under 200ms
    
    async def test_consciousness_core_load(self, mock_ai_components, benchmark_timer):
        """Test Consciousness Core under sustained load."""
        timer = benchmark_timer()
        consciousness = mock_ai_components['consciousness']
        
        # Simulate continuous consciousness state updates
        timer.start()
        
        state_updates = []
        for i in range(1000):  # 1000 state updates
            # Simulate consciousness state evolution
            new_state = {
                'awareness_level': min(1.0, consciousness.awareness_level + (i * 0.0001)),
                'emotional_state': 'peaceful' if i % 3 == 0 else 'focused',
                'spiritual_insight': f'insight_{i}',
                'update_id': i
            }
            state_updates.append(new_state)
        
        execution_time = timer.stop()
        
        print(f"\n🌟 Consciousness Core Load Test:")
        print(f"   State Updates: 1000")
        print(f"   Execution Time: {execution_time:.3f}s")
        print(f"   Updates/sec: {1000 / execution_time:.2f}")
        
        # Performance assertions
        assert execution_time < 1.0  # Should complete within 1 second
        assert len(state_updates) == 1000


@pytest.mark.load
class TestSystemResourceMonitoring:
    """Test system resource usage during load testing."""
    
    def test_memory_usage_under_load(self):
        """Monitor memory usage patterns during load testing."""
        import gc
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Record initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_data_structures = []
        
        for i in range(100):
            # Create large data structure
            data = {
                'id': i,
                'content': f'Large content string {i} ' * 1000,
                'metadata': {'index': i, 'timestamp': datetime.utcnow()},
                'nested_data': [{'item': j} for j in range(100)]
            }
            large_data_structures.append(data)
        
        # Record peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up and record final memory
        large_data_structures.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_recovered = peak_memory - final_memory
        
        # Get tracemalloc statistics
        current, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n💾 Memory Usage Analysis:")
        print(f"   Initial Memory: {initial_memory:.1f} MB")
        print(f"   Peak Memory: {peak_memory:.1f} MB")
        print(f"   Memory Increase: {memory_increase:.1f} MB")
        print(f"   Memory Recovered: {memory_recovered:.1f} MB")
        print(f"   Final Memory: {final_memory:.1f} MB")
        print(f"   Traced Peak: {peak_trace / 1024 / 1024:.1f} MB")
        
        # Memory usage assertions
        assert memory_increase < PERFORMANCE_THRESHOLDS['max_memory_usage']
        assert memory_recovered > memory_increase * 0.8  # Should recover 80%+
    
    def test_cpu_usage_monitoring(self):
        """Monitor CPU usage during intensive operations."""
        import threading
        import time
        
        cpu_samples = []
        monitoring = True
        
        def monitor_cpu():
            """Monitor CPU usage in background thread."""
            while monitoring:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                time.sleep(0.1)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Simulate CPU-intensive operations
        start_time = time.time()
        
        # CPU-intensive computation
        for i in range(100000):
            # Mathematical operations
            result = sum(j**2 for j in range(100))
            # String operations  
            text_processing = f"Processing item {i}" * 10
            # List operations
            data_processing = [x for x in range(50) if x % 2 == 0]
        
        end_time = time.time()
        
        # Stop monitoring
        monitoring = False
        monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            min_cpu = min(cpu_samples)
        else:
            avg_cpu = max_cpu = min_cpu = 0
        
        execution_time = end_time - start_time
        
        print(f"\n🔧 CPU Usage Analysis:")
        print(f"   Execution Time: {execution_time:.3f}s")
        print(f"   Average CPU: {avg_cpu:.1f}%")
        print(f"   Max CPU: {max_cpu:.1f}%")
        print(f"   Min CPU: {min_cpu:.1f}%")
        print(f"   CPU Samples: {len(cpu_samples)}")
        
        # CPU usage assertions
        assert max_cpu <= PERFORMANCE_THRESHOLDS['max_cpu_usage']
        assert execution_time < 10.0  # Should complete within 10 seconds


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "load"
    ])
