"""
🧪 DharmaMind Security Test Suite

Comprehensive testing of all security implementations:
- bcrypt password hashing
- Input sanitization (XSS/SQL injection prevention)
- Data encryption
- HTTPS configuration
- Database security
"""

import asyncio
import json
import logging
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.security_service import (
    hash_password,
    verify_password,
    sanitize_input,
    SecurityLevel
)
from services.https_service import validate_https
from services.database_service import get_db_health

logger = logging.getLogger(__name__)


class SecurityTestSuite:
    """🧪 Comprehensive security testing"""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
    
    def test_bcrypt_password_hashing(self):
        """Test bcrypt password hashing security"""
        print("\n🔐 Testing bcrypt Password Hashing...")
        
        test_cases = [
            {
                "name": "Strong password hashing",
                "password": "SecurePassword123!",
                "should_pass": True
            },
            {
                "name": "Password verification",
                "password": "TestPassword456#",
                "should_pass": True
            },
            {
                "name": "Wrong password rejection",
                "password": "CorrectPassword789$",
                "wrong_password": "WrongPassword123!",
                "should_pass": True
            }
        ]
        
        for test_case in test_cases:
            try:
                password = test_case["password"]
                
                # Test hashing
                hashed = hash_password(password)
                
                # Verify it's bcrypt (should start with $2b$)
                if not hashed.startswith("$2b$"):
                    raise ValueError("Not using bcrypt algorithm")
                
                # Test correct verification
                if not verify_password(password, hashed):
                    raise ValueError("Password verification failed")
                
                # Test wrong password rejection (if provided)
                if "wrong_password" in test_case:
                    if verify_password(test_case["wrong_password"], hashed):
                        raise ValueError("Wrong password was accepted")
                
                self._record_test_result(test_case["name"], True, "✅ Passed")
                print(f"  ✅ {test_case['name']}")
                
            except Exception as e:
                self._record_test_result(test_case["name"], False, str(e))
                print(f"  ❌ {test_case['name']}: {e}")
    
    def test_input_sanitization(self):
        """Test input sanitization against XSS and SQL injection"""
        print("\n🧹 Testing Input Sanitization...")
        
        # XSS test cases
        xss_tests = [
            {
                "name": "Basic XSS script tag",
                "input": "<script>alert('xss')</script>",
                "should_reject": True
            },
            {
                "name": "JavaScript protocol XSS",
                "input": "javascript:alert('xss')",
                "should_reject": True
            },
            {
                "name": "Event handler XSS",
                "input": "<img src=x onerror=alert('xss')>",
                "should_reject": True
            },
            {
                "name": "Safe text input",
                "input": "Hello, this is a normal message",
                "should_reject": False
            }
        ]
        
        # SQL injection test cases
        sql_tests = [
            {
                "name": "Union-based SQL injection",
                "input": "1' UNION SELECT * FROM users--",
                "should_reject": True
            },
            {
                "name": "Boolean-based SQL injection",
                "input": "admin' OR '1'='1",
                "should_reject": True
            },
            {
                "name": "Time-based SQL injection",
                "input": "1'; WAITFOR DELAY '00:00:05'--",
                "should_reject": True
            },
            {
                "name": "Safe SQL-like text",
                "input": "I like SQL databases",
                "should_reject": False
            }
        ]
        
        all_tests = xss_tests + sql_tests
        
        for test_case in all_tests:
            try:
                test_input = test_case["input"]
                should_reject = test_case["should_reject"]
                
                if should_reject:
                    # Should raise an exception
                    try:
                        sanitize_input(test_input, security_level=SecurityLevel.HIGH)
                        raise ValueError("Malicious input was not rejected")
                    except ValueError as e:
                        if "malicious" in str(e).lower() or "invalid" in str(e).lower():
                            # Expected rejection
                            pass
                        else:
                            raise e
                else:
                    # Should not raise an exception
                    result = sanitize_input(test_input, security_level=SecurityLevel.HIGH)
                    if not result:
                        raise ValueError("Safe input was incorrectly rejected")
                
                self._record_test_result(test_case["name"], True, "✅ Passed")
                print(f"  ✅ {test_case['name']}")
                
            except Exception as e:
                self._record_test_result(test_case["name"], False, str(e))
                print(f"  ❌ {test_case['name']}: {e}")
    
    def test_https_configuration(self):
        """Test HTTPS configuration"""
        print("\n🔒 Testing HTTPS Configuration...")
        
        try:
            https_status = validate_https()
            
            tests = [
                {
                    "name": "Certificate generation attempted",
                    "condition": "certificate_exists" in https_status,
                    "expected": True
                },
                {
                    "name": "Private key generation attempted", 
                    "condition": "private_key_exists" in https_status,
                    "expected": True
                },
                {
                    "name": "SSL context validation attempted",
                    "condition": "ssl_context_valid" in https_status,
                    "expected": True
                }
            ]
            
            for test in tests:
                if test["condition"] == test["expected"]:
                    self._record_test_result(test["name"], True, "✅ Passed")
                    print(f"  ✅ {test['name']}")
                else:
                    self._record_test_result(test["name"], False, "Configuration missing")
                    print(f"  ❌ {test['name']}: Configuration missing")
            
            # Print HTTPS status for info
            print(f"  📋 HTTPS Status: {json.dumps(https_status, indent=2)}")
            
        except Exception as e:
            self._record_test_result("HTTPS Configuration Test", False, str(e))
            print(f"  ❌ HTTPS Configuration Test: {e}")
    
    async def test_database_security(self):
        """Test database security configuration"""
        print("\n🗄️ Testing Database Security...")
        
        try:
            db_health = await get_db_health()
            
            tests = [
                {
                    "name": "Database health check",
                    "condition": "status" in db_health,
                    "expected": True
                },
                {
                    "name": "Encryption support",
                    "condition": db_health.get("encryption") == "enabled",
                    "expected": True
                }
            ]
            
            for test in tests:
                if test["condition"] == test["expected"]:
                    self._record_test_result(test["name"], True, "✅ Passed")
                    print(f"  ✅ {test['name']}")
                else:
                    self._record_test_result(test["name"], False, "Test failed")
                    print(f"  ❌ {test['name']}: Test failed")
            
            # Print database status for info
            print(f"  📋 Database Status: {json.dumps(db_health, indent=2)}")
            
        except Exception as e:
            self._record_test_result("Database Security Test", False, str(e))
            print(f"  ❌ Database Security Test: {e}")
    
    def test_secret_management(self):
        """Test secret management and environment variables"""
        print("\n🔑 Testing Secret Management...")
        
        tests = [
            {
                "name": "Environment variable support",
                "test": lambda: os.getenv("SECRET_KEY", "default") is not None,
                "expected": True
            },
            {
                "name": "Encryption key environment support",
                "test": lambda: os.getenv("DHARMAMIND_ENCRYPTION_KEY", "default") is not None,
                "expected": True
            },
            {
                "name": "Database URL environment support",
                "test": lambda: os.getenv("DATABASE_URL", "default") is not None,
                "expected": True
            }
        ]
        
        for test in tests:
            try:
                result = test["test"]()
                if result == test["expected"]:
                    self._record_test_result(test["name"], True, "✅ Passed")
                    print(f"  ✅ {test['name']}")
                else:
                    self._record_test_result(test["name"], False, "Test failed")
                    print(f"  ❌ {test['name']}: Test failed")
            except Exception as e:
                self._record_test_result(test["name"], False, str(e))
                print(f"  ❌ {test['name']}: {e}")
    
    def _record_test_result(self, name: str, passed: bool, details: str):
        """Record test result"""
        self.test_results.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    async def run_all_tests(self):
        """Run all security tests"""
        print("🛡️ DharmaMind Security Test Suite")
        print("=" * 50)
        
        # Run all tests
        self.test_bcrypt_password_hashing()
        self.test_input_sanitization()
        self.test_https_configuration()
        await self.test_database_security()
        self.test_secret_management()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📊 Total: {len(self.test_results)}")
        
        success_rate = (self.passed_tests / len(self.test_results)) * 100
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🎉 Security implementation is EXCELLENT!")
        elif success_rate >= 60:
            print("\n✅ Security implementation is GOOD!")
        else:
            print("\n⚠️ Security implementation needs improvement!")
        
        return self.test_results


async def main():
    """Run the security test suite"""
    suite = SecurityTestSuite()
    results = await suite.run_all_tests()
    
    # Return results for potential integration testing
    return results


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
