"""
⚙️ Test Configuration for DharmaMind Platform  
===========================================

Comprehensive test configuration and settings for the DharmaMind spiritual guidance platform:
- Test environment configuration
- Performance thresholds and benchmarks
- Cultural sensitivity settings
- Mock service configurations
- Test data management
- Reporting and analytics setup

These configurations ensure thorough testing while maintaining
the spiritual integrity and cultural sensitivity of the platform.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Test Environment Configuration
@dataclass
class TestEnvironmentConfig:
    """Configuration for test environment setup"""
    
    # Database and Cache Configuration
    use_fake_redis: bool = True
    redis_test_db: int = 15  # Use separate DB for testing
    cache_ttl_seconds: int = 300  # 5 minutes for testing
    
    # AI Service Configuration
    use_mock_llm: bool = True
    use_mock_embeddings: bool = True
    mock_response_delay: float = 0.1  # Simulate API delay
    
    # Performance Test Configuration
    performance_test_timeout: int = 300  # 5 minutes
    benchmark_iterations: int = 10
    concurrent_test_max_users: int = 100
    load_test_duration: int = 60  # seconds
    
    # Cultural Sensitivity Configuration
    enable_cultural_validation: bool = True
    strict_cultural_checking: bool = False  # Set True for production
    supported_languages: List[str] = field(default_factory=lambda: [
        'en', 'hi', 'sa', 'zh', 'ar', 'es', 'fr', 'de', 'ja'
    ])
    
    # Logging Configuration
    log_level: str = 'INFO'
    enable_test_logging: bool = True
    log_file_path: str = 'tests/logs/test_results.log'
    
    # Test Data Configuration
    test_data_path: str = 'tests/fixtures'
    generate_synthetic_data: bool = True
    synthetic_data_size: int = 1000
    
    # Reporting Configuration
    generate_html_report: bool = True
    report_output_path: str = 'tests/reports'
    include_performance_graphs: bool = True
    
    # Safety and Ethics Configuration
    enable_content_safety_checks: bool = True
    spiritual_appropriateness_threshold: float = 0.8
    cultural_sensitivity_threshold: float = 0.9

@dataclass 
class PerformanceThresholds:
    """Performance thresholds for different test categories"""
    
    # Response Time Thresholds (seconds)
    emotional_analysis_max_time: float = 0.05
    knowledge_search_max_time: float = 0.05
    complete_guidance_max_time: float = 0.2
    cache_operation_max_time: float = 0.01
    
    # Throughput Thresholds
    queries_per_second_min: int = 50
    concurrent_users_min: int = 25
    
    # Quality Thresholds
    emotional_accuracy_min: float = 0.8
    knowledge_relevance_min: float = 0.85
    response_coherence_min: float = 0.8
    cultural_sensitivity_min: float = 0.9
    
    # Resource Usage Thresholds
    memory_usage_max_mb: int = 512
    cpu_usage_max_percent: float = 70
    cache_hit_ratio_min: float = 0.8
    
    # Percentile Thresholds
    p95_response_time_max: float = 0.3
    p99_response_time_max: float = 0.5

# Global Test Configuration
TEST_CONFIG = TestEnvironmentConfig()
PERFORMANCE_THRESHOLDS = PerformanceThresholds()

# Test Environment Setup Functions

def setup_test_environment() -> Dict[str, Any]:
    """Set up the test environment with all necessary configurations"""
    
    # Create test directories
    test_dirs = [
        'tests/logs',
        'tests/reports', 
        'tests/baselines',
        'tests/temp',
        'tests/artifacts'
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    if TEST_CONFIG.enable_test_logging:
        logging.basicConfig(
            level=getattr(logging, TEST_CONFIG.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(TEST_CONFIG.log_file_path),
                logging.StreamHandler()
            ]
        )
    
    # Environment variables for testing
    test_env = {
        'DHARMA_MIND_TEST_MODE': 'true',
        'DHARMA_MIND_LOG_LEVEL': TEST_CONFIG.log_level,
        'DHARMA_MIND_USE_MOCK_SERVICES': str(TEST_CONFIG.use_mock_llm),
        'DHARMA_MIND_REDIS_DB': str(TEST_CONFIG.redis_test_db),
        'DHARMA_MIND_CACHE_TTL': str(TEST_CONFIG.cache_ttl_seconds)
    }
    
    # Set environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    return {
        'config': TEST_CONFIG,
        'thresholds': PERFORMANCE_THRESHOLDS,
        'environment_variables': test_env
    }

def teardown_test_environment():
    """Clean up test environment"""
    
    # Clean temporary files
    temp_dir = Path('tests/temp')
    if temp_dir.exists():
        for file in temp_dir.glob('*'):
            file.unlink()
    
    # Reset environment variables
    test_env_vars = [
        'DHARMA_MIND_TEST_MODE',
        'DHARMA_MIND_LOG_LEVEL', 
        'DHARMA_MIND_USE_MOCK_SERVICES',
        'DHARMA_MIND_REDIS_DB',
        'DHARMA_MIND_CACHE_TTL'
    ]
    
    for var in test_env_vars:
        os.environ.pop(var, None)

def validate_test_environment() -> Dict[str, bool]:
    """Validate that test environment is properly configured"""
    
    validations = {}
    
    # Check required directories
    required_dirs = ['tests', 'tests/logs', 'tests/reports']
    for dir_path in required_dirs:
        validations[f'directory_{dir_path}'] = Path(dir_path).exists()
    
    # Check mock services availability
    validations['mock_llm_configured'] = TEST_CONFIG.use_mock_llm
    validations['mock_embeddings_configured'] = TEST_CONFIG.use_mock_embeddings
    
    # Check cultural configuration
    validations['cultural_validation_enabled'] = TEST_CONFIG.enable_cultural_validation
    validations['supported_languages_configured'] = len(TEST_CONFIG.supported_languages) > 0
    
    # Check performance configuration
    validations['performance_thresholds_set'] = PERFORMANCE_THRESHOLDS.emotional_analysis_max_time > 0
    
    return validations

# Export configuration objects
__all__ = [
    'TEST_CONFIG',
    'PERFORMANCE_THRESHOLDS',
    'TestEnvironmentConfig',
    'PerformanceThresholds',
    'setup_test_environment',
    'teardown_test_environment',
    'validate_test_environment'
]