#!/usr/bin/env python3
"""
🔍 Test Validation and Execution Script for DharmaMind Platform
============================================================

Comprehensive test validation and execution script for the DharmaMind spiritual guidance platform:
- Pre-test environment validation
- Test dependency checking
- Test execution with proper error handling
- Post-test cleanup and reporting
- Continuous integration support
- Test result validation

This script ensures all tests run in a clean, validated environment
while maintaining the spiritual integrity of the platform.
"""

import os
import sys
import asyncio
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test components
from tests.test_config import setup_test_environment, teardown_test_environment, validate_test_environment
from tests.run_tests import DharmaMindTestRunner

class TestValidator:
    """Validates test environment and dependencies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        self.setup_completed = False
    
    def validate_python_environment(self) -> Dict[str, bool]:
        """Validate Python environment and required packages"""
        validations = {}
        
        # Check Python version
        python_version = sys.version_info
        validations['python_version_ok'] = python_version >= (3, 8)
        
        # Check required packages
        required_packages = [
            'pytest',
            'pytest-asyncio', 
            'fakeredis',
            'asyncio',
            'dataclasses',
            'typing',
            'pathlib',
            'json',
            'logging'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                validations[f'package_{package}'] = True
            except ImportError:
                validations[f'package_{package}'] = False
                self.logger.warning(f"Missing package: {package}")
        
        return validations
    
    def validate_file_structure(self) -> Dict[str, bool]:
        """Validate required test file structure"""
        validations = {}
        
        # Required directories
        required_dirs = [
            'tests',
            'tests/unit',
            'tests/integration', 
            'tests/performance',
            'tests/fixtures'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            validations[f'dir_{dir_path}'] = path.exists() and path.is_dir()
        
        # Required files
        required_files = [
            'tests/__init__.py',
            'tests/unit/test_emotional_intelligence.py',
            'tests/unit/test_knowledge_cache.py',
            'tests/integration/test_system_integration.py',
            'tests/performance/test_benchmarks.py',
            'tests/fixtures/test_fixtures.py',
            'tests/test_config.py',
            'tests/run_tests.py'
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            validations[f'file_{file_path}'] = path.exists() and path.is_file()
        
        return validations
    
    def validate_test_imports(self) -> Dict[str, bool]:
        """Validate that all test modules can be imported"""
        validations = {}
        
        test_modules = [
            'tests',
            'tests.unit.test_emotional_intelligence',
            'tests.unit.test_knowledge_cache',
            'tests.integration.test_system_integration',
            'tests.performance.test_benchmarks',
            'tests.fixtures.test_fixtures',
            'tests.test_config',
            'tests.run_tests'
        ]
        
        for module in test_modules:
            try:
                __import__(module)
                validations[f'import_{module}'] = True
            except Exception as e:
                validations[f'import_{module}'] = False
                self.logger.error(f"Failed to import {module}: {e}")
        
        return validations
    
    def validate_test_dependencies(self) -> Dict[str, bool]:
        """Validate test dependencies and mock services"""
        validations = {}
        
        try:
            # Test FakeRedis
            import fakeredis.aioredis
            fake_redis = fakeredis.aioredis.FakeRedis()
            validations['fakeredis_available'] = True
        except Exception as e:
            validations['fakeredis_available'] = False
            self.logger.error(f"FakeRedis validation failed: {e}")
        
        try:
            # Test async capabilities
            async def test_async():
                return True
            
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(test_async())
            loop.close()
            validations['asyncio_working'] = result
        except Exception as e:
            validations['asyncio_working'] = False
            self.logger.error(f"Asyncio validation failed: {e}")
        
        return validations
    
    def run_full_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Run complete validation suite"""
        print("🔍 Validating DharmaMind test environment...")
        
        all_validations = {}
        
        # Python environment
        print("  🐍 Checking Python environment...")
        python_validations = self.validate_python_environment()
        all_validations['python_environment'] = python_validations
        
        # File structure
        print("  📁 Checking file structure...")
        file_validations = self.validate_file_structure()
        all_validations['file_structure'] = file_validations
        
        # Test imports
        print("  📦 Checking test imports...")
        import_validations = self.validate_test_imports()
        all_validations['test_imports'] = import_validations
        
        # Dependencies
        print("  🔧 Checking test dependencies...")
        dependency_validations = self.validate_test_dependencies()
        all_validations['test_dependencies'] = dependency_validations
        
        # Test environment setup
        print("  ⚙️  Checking test environment...")
        try:
            env_config = setup_test_environment()
            env_validations = validate_test_environment()
            all_validations['test_environment'] = env_validations
            teardown_test_environment()
        except Exception as e:
            all_validations['test_environment'] = {'setup_failed': True, 'error': str(e)}
        
        # Determine overall status
        all_passed = True
        failed_validations = []
        
        for category, validations in all_validations.items():
            for validation_name, passed in validations.items():
                if not passed:
                    all_passed = False
                    failed_validations.append(f"{category}.{validation_name}")
        
        # Summary
        if all_passed:
            print("  ✅ All validations passed!")
        else:
            print(f"  ❌ {len(failed_validations)} validations failed:")
            for failed in failed_validations[:5]:  # Show first 5
                print(f"     - {failed}")
            if len(failed_validations) > 5:
                print(f"     ... and {len(failed_validations) - 5} more")
        
        return all_passed, {
            'overall_status': 'PASSED' if all_passed else 'FAILED',
            'total_validations': sum(len(v) for v in all_validations.values()),
            'failed_validations': failed_validations,
            'detailed_results': all_validations
        }

class TestExecutor:
    """Executes tests with proper error handling and reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_results = {}
    
    async def execute_test_suite(self, test_categories: Optional[List[str]] = None, smoke_only: bool = False) -> Dict[str, Any]:
        """Execute the complete test suite"""
        
        print(f"🚀 Executing DharmaMind test suite...")
        
        start_time = time.time()
        
        try:
            # Create test runner
            runner = DharmaMindTestRunner()
            
            # Execute tests
            if smoke_only:
                print("🔥 Running smoke tests only...")
                results = await runner.run_quick_smoke_test()
            else:
                print(f"📊 Running test categories: {test_categories or 'default'}")
                results = await runner.run_all_tests(test_categories)
            
            execution_time = time.time() - start_time
            
            # Add execution metadata
            results['execution_metadata'] = {
                'execution_time': execution_time,
                'test_categories': test_categories,
                'smoke_only': smoke_only,
                'executor_version': '1.0.0'
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return {
                'status': 'EXECUTION_FAILED',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def run_pytest_directly(self, test_path: str = "tests/", extra_args: List[str] = None) -> Dict[str, Any]:
        """Run pytest directly for comparison"""
        
        print(f"🧪 Running pytest directly on {test_path}...")
        
        cmd = ['python', '-m', 'pytest', test_path, '-v']
        if extra_args:
            cmd.extend(extra_args)
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = time.time() - start_time
            
            return {
                'status': 'COMPLETED',
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'TIMEOUT',
                'error': 'Pytest execution timed out after 300 seconds'
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e)
            }

def main():
    """Main execution function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='DharmaMind Test Validation and Execution')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation only, skip test execution')
    
    parser.add_argument('--smoke-only', action='store_true',
                       help='Run smoke tests only')
    
    parser.add_argument('--categories', nargs='+',
                       choices=['unit', 'integration', 'performance', 'cultural', 'security', 'e2e'],
                       help='Test categories to run')
    
    parser.add_argument('--pytest-direct', action='store_true',
                       help='Also run pytest directly for comparison')
    
    parser.add_argument('--output-dir', default='tests/reports',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation phase
    print("🕉️  DharmaMind Platform Test Validation & Execution")
    print("=" * 60)
    
    validator = TestValidator()
    validation_passed, validation_results = validator.run_full_validation()
    
    # Save validation results
    validation_file = output_dir / 'validation_results.json'
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"📁 Validation results saved to: {validation_file}")
    
    if not validation_passed:
        print("❌ Validation failed. Cannot proceed with test execution.")
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Validation complete. Skipping test execution.")
        sys.exit(0)
    
    # Test execution phase
    executor = TestExecutor()
    
    async def run_tests():
        # Run our test suite
        test_results = await executor.execute_test_suite(
            test_categories=args.categories,
            smoke_only=args.smoke_only
        )
        
        # Save test results
        results_file = output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"📁 Test results saved to: {results_file}")
        
        # Optionally run pytest directly
        if args.pytest_direct:
            pytest_results = executor.run_pytest_directly()
            pytest_file = output_dir / 'pytest_results.json'
            with open(pytest_file, 'w') as f:
                json.dump(pytest_results, f, indent=2)
            print(f"📁 Pytest results saved to: {pytest_file}")
        
        # Determine exit code
        if isinstance(test_results, dict):
            status = test_results.get('test_run_summary', {}).get('overall_status', 'UNKNOWN')
            if status == 'PASSED':
                print("🎉 All tests passed successfully!")
                sys.exit(0)
            else:
                print(f"❌ Tests failed with status: {status}")
                sys.exit(1)
        else:
            print("⚠️ Unexpected test results format")
            sys.exit(1)
    
    # Run the tests
    asyncio.run(run_tests())

if __name__ == '__main__':
    main()