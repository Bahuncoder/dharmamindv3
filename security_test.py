#!/usr/bin/env python3
"""
Security Fix Verification Script
Tests the implemented security improvements
"""

import sys
import os
sys.path.append('/media/rupert/New Volume/new complete apps/backend')

from app.routes.auth import hash_password, verify_password, create_access_token
from app.routes.admin_auth import create_admin_token, pwd_context
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta

def test_password_security():
    """Test secure password hashing"""
    print("🔐 Testing Password Security...")
    
    # Test password hashing
    password = "TestPassword123!"
    hashed = hash_password(password)
    
    print(f"   ✅ Password hashed: {hashed[:50]}...")
    print(f"   ✅ Hash length: {len(hashed)} characters")
    
    # Test password verification
    assert verify_password(password, hashed), "Password verification failed!"
    print("   ✅ Password verification successful")
    
    # Test wrong password
    assert not verify_password("WrongPassword", hashed), "Wrong password should fail!"
    print("   ✅ Wrong password correctly rejected")
    
    # Test that different passwords create different hashes
    hash1 = hash_password(password)
    hash2 = hash_password(password)
    assert hash1 != hash2, "Same password should create different hashes (salt test)!"
    print("   ✅ Salt verification: Different hashes for same password")

def test_jwt_security():
    """Test JWT token security"""
    print("\n🔑 Testing JWT Security...")
    
    # Test token creation
    token_data = {"sub": "test@dharmamind.com", "email": "test@dharmamind.com", "plan": "free"}
    token = create_access_token(data=token_data)
    
    print(f"   ✅ JWT token created: {token[:50]}...")
    
    # Test token decoding
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    
    assert decoded["sub"] == "test@dharmamind.com", "Token decoding failed!"
    print("   ✅ JWT token decoding successful")
    
    # Test token expiration
    assert "exp" in decoded, "Token should have expiration!"
    exp_time = datetime.fromtimestamp(decoded["exp"])
    assert exp_time > datetime.now(), "Token should not be expired!"
    print("   ✅ Token expiration properly set")

def test_admin_security():
    """Test admin authentication security"""
    print("\n👑 Testing Admin Security...")
    
    # Test admin token creation
    admin_token = create_admin_token("admin@dharmamind.com")
    print(f"   ✅ Admin token created: {admin_token[:50]}...")
    
    # Test admin password hashing
    admin_password = "SecureAdminPassword2025!"
    admin_hash = pwd_context.hash(admin_password)
    
    assert pwd_context.verify(admin_password, admin_hash), "Admin password verification failed!"
    print("   ✅ Admin password hashing verified")

def test_security_improvements():
    """Test overall security improvements"""
    print("\n🛡️ Testing Security Improvements...")
    
    # Test that we're using bcrypt (should contain $2b$)
    test_hash = hash_password("test")
    assert test_hash.startswith("$2b$"), "Should be using bcrypt hashing!"
    print("   ✅ bcrypt hashing confirmed")
    
    # Test JWT contains proper claims
    token_data = {"sub": "test", "role": "user"}
    token = create_access_token(data=token_data)
    
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    
    assert "sub" in decoded, "Token should contain subject!"
    assert "exp" in decoded, "Token should contain expiration!"
    print("   ✅ JWT claims properly structured")

def run_security_tests():
    """Run all security tests"""
    print("🚀 Running Security Fix Verification Tests")
    print("=" * 50)
    
    try:
        test_password_security()
        test_jwt_security()
        test_admin_security()
        test_security_improvements()
        
        print("\n" + "=" * 50)
        print("✅ ALL SECURITY TESTS PASSED!")
        print("🔐 Password security: IMPROVED")
        print("🔑 JWT authentication: SECURED")
        print("👑 Admin authentication: HARDENED")
        print("🛡️ Overall security: ENHANCED")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ SECURITY TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)
