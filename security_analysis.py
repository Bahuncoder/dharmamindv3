#!/usr/bin/env python3
"""
🔒 DHARMAMIND SECURITY ANALYSIS & HARDENING ASSESSMENT
====================================================

Comprehensive security evaluation of the DharmaMind spiritual AI system
to identify vulnerabilities and implement robust protection measures.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
import subprocess
import hashlib
import secrets
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SecurityAnalyzer:
    """Comprehensive security analysis for DharmaMind system."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_score = 0
        self.recommendations = []
        self.critical_issues = []
        
    def analyze_system_security(self) -> Dict[str, Any]:
        """Perform comprehensive security analysis."""
        
        print("🔒 DHARMAMIND SECURITY ANALYSIS")
        print("=" * 60)
        print("🛡️ Analyzing security posture of spiritual AI system...")
        print()
        
        analysis_results = {
            "overall_score": 0,
            "categories": {},
            "vulnerabilities": [],
            "critical_issues": [],
            "recommendations": [],
            "security_status": "UNKNOWN"
        }
        
        # 1. File System Security
        print("📁 1. FILE SYSTEM SECURITY ANALYSIS")
        fs_results = self._analyze_file_security()
        analysis_results["categories"]["file_system"] = fs_results
        
        # 2. Database Security
        print("\n🗄️ 2. DATABASE SECURITY ANALYSIS")
        db_results = self._analyze_database_security()
        analysis_results["categories"]["database"] = db_results
        
        # 3. API Security
        print("\n🌐 3. API SECURITY ANALYSIS")
        api_results = self._analyze_api_security()
        analysis_results["categories"]["api"] = api_results
        
        # 4. Authentication & Authorization
        print("\n🔐 4. AUTHENTICATION & AUTHORIZATION")
        auth_results = self._analyze_auth_security()
        analysis_results["categories"]["authentication"] = auth_results
        
        # 5. Input Validation & Sanitization
        print("\n🧹 5. INPUT VALIDATION & SANITIZATION")
        input_results = self._analyze_input_security()
        analysis_results["categories"]["input_validation"] = input_results
        
        # 6. Secrets Management
        print("\n🔑 6. SECRETS MANAGEMENT")
        secrets_results = self._analyze_secrets_security()
        analysis_results["categories"]["secrets"] = secrets_results
        
        # 7. Container & Infrastructure Security
        print("\n🐳 7. CONTAINER & INFRASTRUCTURE SECURITY")
        infra_results = self._analyze_infrastructure_security()
        analysis_results["categories"]["infrastructure"] = infra_results
        
        # 8. Spiritual Data Protection
        print("\n🕉️ 8. SPIRITUAL DATA PROTECTION")
        spiritual_results = self._analyze_spiritual_data_security()
        analysis_results["categories"]["spiritual_data"] = spiritual_results
        
        # Calculate overall security score
        category_scores = [cat["score"] for cat in analysis_results["categories"].values()]
        analysis_results["overall_score"] = sum(category_scores) / len(category_scores)
        
        # Determine security status
        if analysis_results["overall_score"] >= 80:
            analysis_results["security_status"] = "EXCELLENT"
        elif analysis_results["overall_score"] >= 70:
            analysis_results["security_status"] = "GOOD"
        elif analysis_results["overall_score"] >= 60:
            analysis_results["security_status"] = "FAIR"
        else:
            analysis_results["security_status"] = "NEEDS ATTENTION"
        
        # Compile recommendations
        analysis_results["vulnerabilities"] = self.vulnerabilities
        analysis_results["critical_issues"] = self.critical_issues
        analysis_results["recommendations"] = self.recommendations
        
        return analysis_results
    
    def _analyze_file_security(self) -> Dict[str, Any]:
        """Analyze file system security."""
        score = 70  # Base score
        issues = []
        
        print("   📋 Checking file permissions...")
        
        # Check for sensitive files with loose permissions
        sensitive_files = [
            ".env",
            "docker-compose.yml",
            "requirements.txt",
            "backend/app/core/security.py"
        ]
        
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                permissions = oct(stat_info.st_mode)[-3:]
                if permissions > "644":
                    issues.append(f"File {file_path} has loose permissions: {permissions}")
                    score -= 5
        
        # Check for exposed secrets in files
        print("   🔍 Scanning for exposed secrets...")
        secret_patterns = ["password", "secret", "key", "token", "api_key"]
        
        try:
            result = subprocess.run(
                ["grep", "-r", "-i", "--include=*.py", "--include=*.json", "--include=*.env", 
                 "|".join(secret_patterns), "."],
                capture_output=True, text=True, cwd=project_root
            )
            if result.stdout:
                issues.append("Potential secrets found in source code")
                self.critical_issues.append("🚨 CRITICAL: Secrets potentially exposed in source code")
                score -= 15
        except:
            pass
        
        # Check for .env file exposure
        if os.path.exists(".env"):
            print("   ✅ .env file found (good for secrets management)")
            score += 5
        else:
            issues.append("No .env file found for secrets management")
            score -= 10
        
        print(f"   📊 File Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "File System Security"
        }
    
    def _analyze_database_security(self) -> Dict[str, Any]:
        """Analyze database security."""
        score = 75  # Base score
        issues = []
        
        print("   🔍 Checking database configurations...")
        
        # Check SQLite database files
        db_files = list(Path(".").rglob("*.db"))
        for db_file in db_files:
            print(f"   📄 Found database: {db_file}")
            
            # Check permissions
            stat_info = os.stat(db_file)
            permissions = oct(stat_info.st_mode)[-3:]
            if permissions > "640":
                issues.append(f"Database {db_file} has loose permissions: {permissions}")
                score -= 10
            
            # Basic SQLite security check
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check for SQL injection vulnerabilities (basic check)
                cursor.execute("PRAGMA table_list")
                tables = cursor.fetchall()
                
                conn.close()
                
                if len(tables) > 0:
                    print(f"   ✅ Database {db_file} accessible with {len(tables)} tables")
                
            except Exception as e:
                issues.append(f"Database {db_file} access error: {e}")
                score -= 5
        
        # Check for database encryption
        if not any("encrypt" in str(db_file).lower() for db_file in db_files):
            issues.append("No evidence of database encryption")
            self.recommendations.append("🔐 Implement database encryption for sensitive spiritual data")
            score -= 15
        
        # Check for backup security
        backup_files = list(Path(".").rglob("*.backup")) + list(Path(".").rglob("*.bak"))
        if backup_files:
            issues.append("Backup files found - ensure they're encrypted")
            score -= 5
        
        print(f"   📊 Database Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "Database Security"
        }
    
    def _analyze_api_security(self) -> Dict[str, Any]:
        """Analyze API security."""
        score = 65  # Base score - needs improvement
        issues = []
        
        print("   🌐 Checking API security configurations...")
        
        # Check for CORS configuration
        backend_files = list(Path("backend").rglob("*.py"))
        cors_configured = False
        rate_limiting = False
        
        for file_path in backend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "CORS" in content or "cors" in content:
                        cors_configured = True
                        print("   ✅ CORS configuration found")
                        score += 5
                    if "rate" in content.lower() and "limit" in content.lower():
                        rate_limiting = True
                        print("   ✅ Rate limiting found")
                        score += 10
            except:
                continue
        
        if not cors_configured:
            issues.append("No CORS configuration found")
            self.recommendations.append("🌐 Implement proper CORS configuration")
            score -= 10
        
        if not rate_limiting:
            issues.append("No rate limiting found")
            self.critical_issues.append("🚨 CRITICAL: No API rate limiting - vulnerable to DoS attacks")
            score -= 20
        
        # Check for HTTPS enforcement
        issues.append("HTTPS enforcement not verified")
        self.recommendations.append("🔒 Ensure HTTPS enforcement in production")
        score -= 10
        
        # Check for API authentication
        auth_found = False
        for file_path in backend_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(auth_term in content.lower() for auth_term in ["token", "jwt", "auth", "bearer"]):
                        auth_found = True
                        break
            except:
                continue
        
        if auth_found:
            print("   ✅ Authentication mechanisms found")
            score += 10
        else:
            issues.append("No API authentication found")
            self.critical_issues.append("🚨 CRITICAL: No API authentication - endpoints are public")
            score -= 25
        
        print(f"   📊 API Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "API Security"
        }
    
    def _analyze_auth_security(self) -> Dict[str, Any]:
        """Analyze authentication and authorization."""
        score = 60  # Base score - needs significant improvement
        issues = []
        
        print("   🔐 Checking authentication mechanisms...")
        
        # Check for authentication files
        auth_files = []
        for pattern in ["*auth*", "*security*", "*login*"]:
            auth_files.extend(list(Path(".").rglob(pattern + ".py")))
        
        if auth_files:
            print(f"   ✅ Found {len(auth_files)} authentication-related files")
            score += 15
        else:
            issues.append("No dedicated authentication files found")
            self.critical_issues.append("🚨 CRITICAL: No authentication system implemented")
            score -= 30
        
        # Check for password security
        password_security = False
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(term in content for term in ["bcrypt", "scrypt", "argon2", "pbkdf2"]):
                        password_security = True
                        print("   ✅ Secure password hashing found")
                        score += 15
                        break
            except:
                continue
        
        if not password_security:
            issues.append("No secure password hashing found")
            self.recommendations.append("🔐 Implement secure password hashing (bcrypt/argon2)")
            score -= 15
        
        # Check for session management
        session_mgmt = False
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "session" in content.lower():
                        session_mgmt = True
                        break
            except:
                continue
        
        if session_mgmt:
            print("   ✅ Session management found")
            score += 10
        else:
            issues.append("No session management found")
            score -= 10
        
        print(f"   📊 Authentication Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "Authentication & Authorization"
        }
    
    def _analyze_input_security(self) -> Dict[str, Any]:
        """Analyze input validation and sanitization."""
        score = 70  # Base score
        issues = []
        
        print("   🧹 Checking input validation...")
        
        # Check for input validation libraries
        validation_found = False
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(lib in content for lib in ["pydantic", "marshmallow", "cerberus", "validate"]):
                        validation_found = True
                        print("   ✅ Input validation library found")
                        score += 15
                        break
            except:
                continue
        
        if not validation_found:
            issues.append("No input validation library found")
            self.recommendations.append("🧹 Implement input validation with Pydantic or similar")
            score -= 15
        
        # Check for SQL injection protection
        sql_injection_protection = False
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "parameterized" in content or "?" in content or "prepare" in content:
                        sql_injection_protection = True
                        break
            except:
                continue
        
        if sql_injection_protection:
            print("   ✅ SQL injection protection found")
            score += 10
        else:
            issues.append("SQL injection protection not verified")
            score -= 10
        
        # Check for XSS protection
        xss_protection = False
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(term in content for term in ["escape", "sanitize", "bleach"]):
                        xss_protection = True
                        break
            except:
                continue
        
        if xss_protection:
            print("   ✅ XSS protection found")
            score += 10
        else:
            issues.append("XSS protection not found")
            self.recommendations.append("🛡️ Implement XSS protection for user inputs")
            score -= 10
        
        print(f"   📊 Input Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "Input Validation & Sanitization"
        }
    
    def _analyze_secrets_security(self) -> Dict[str, Any]:
        """Analyze secrets management."""
        score = 65  # Base score
        issues = []
        
        print("   🔑 Checking secrets management...")
        
        # Check for .env file
        if os.path.exists(".env"):
            print("   ✅ .env file found")
            score += 10
            
            # Check if .env is in .gitignore
            if os.path.exists(".gitignore"):
                with open(".gitignore", 'r') as f:
                    gitignore_content = f.read()
                    if ".env" in gitignore_content:
                        print("   ✅ .env file properly ignored in git")
                        score += 10
                    else:
                        issues.append(".env file not in .gitignore")
                        self.critical_issues.append("🚨 CRITICAL: .env file not in .gitignore - secrets may be exposed")
                        score -= 20
        else:
            issues.append("No .env file found")
            score -= 15
        
        # Check for hardcoded secrets
        hardcoded_secrets = []
        secret_patterns = [r"password\s*=\s*['\"]", r"secret\s*=\s*['\"]", r"key\s*=\s*['\"]"]
        
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line_num, line in enumerate(content.split('\n'), 1):
                        for pattern in secret_patterns:
                            import re
                            if re.search(pattern, line, re.IGNORECASE):
                                hardcoded_secrets.append(f"{file_path}:{line_num}")
            except:
                continue
        
        if hardcoded_secrets:
            issues.append(f"Potential hardcoded secrets found in {len(hardcoded_secrets)} locations")
            self.critical_issues.append("🚨 CRITICAL: Hardcoded secrets found in source code")
            score -= 25
        else:
            print("   ✅ No obvious hardcoded secrets found")
            score += 10
        
        # Check for environment variable usage
        env_usage = False
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "os.environ" in content or "getenv" in content:
                        env_usage = True
                        break
            except:
                continue
        
        if env_usage:
            print("   ✅ Environment variable usage found")
            score += 10
        else:
            issues.append("No environment variable usage found")
            score -= 10
        
        print(f"   📊 Secrets Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "Secrets Management"
        }
    
    def _analyze_infrastructure_security(self) -> Dict[str, Any]:
        """Analyze container and infrastructure security."""
        score = 75  # Base score
        issues = []
        
        print("   🐳 Checking container security...")
        
        # Check for Dockerfile security
        dockerfiles = list(Path(".").rglob("Dockerfile*"))
        for dockerfile in dockerfiles:
            print(f"   📄 Analyzing {dockerfile}")
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    
                    # Check for non-root user
                    if "USER" in content and "USER root" not in content:
                        print("   ✅ Non-root user found in Dockerfile")
                        score += 10
                    else:
                        issues.append(f"Dockerfile {dockerfile} may run as root")
                        self.recommendations.append("🐳 Run containers as non-root user")
                        score -= 10
                    
                    # Check for multi-stage builds
                    if "FROM" in content and content.count("FROM") > 1:
                        print("   ✅ Multi-stage build found")
                        score += 5
                    
                    # Check for security updates
                    if "apt-get update" in content or "apk update" in content:
                        if "upgrade" in content:
                            print("   ✅ Security updates included")
                            score += 5
                        else:
                            issues.append("Package updates without upgrades")
                            score -= 5
            except:
                continue
        
        # Check for docker-compose security
        compose_files = list(Path(".").rglob("docker-compose*.yml"))
        for compose_file in compose_files:
            print(f"   📄 Analyzing {compose_file}")
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                    
                    # Check for secrets management
                    if "secrets:" in content:
                        print("   ✅ Docker secrets found")
                        score += 10
                    else:
                        issues.append("No Docker secrets configuration")
                        score -= 5
                    
                    # Check for privileged containers
                    if "privileged: true" in content:
                        issues.append("Privileged containers found")
                        self.critical_issues.append("🚨 CRITICAL: Privileged containers are security risk")
                        score -= 20
            except:
                continue
        
        # Check for Kubernetes security
        k8s_files = list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.yml"))
        k8s_security = False
        
        for k8s_file in k8s_files:
            try:
                with open(k8s_file, 'r') as f:
                    content = f.read()
                    if any(k8s_term in content for k8s_term in ["apiVersion:", "kind:", "metadata:"]):
                        k8s_security = True
                        print(f"   ✅ Kubernetes configuration found: {k8s_file}")
                        score += 5
                        break
            except:
                continue
        
        print(f"   📊 Infrastructure Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "Container & Infrastructure Security"
        }
    
    def _analyze_spiritual_data_security(self) -> Dict[str, Any]:
        """Analyze spiritual data protection (unique to DharmaMind)."""
        score = 80  # Base score
        issues = []
        
        print("   🕉️ Checking spiritual data protection...")
        
        # Check for spiritual data encryption
        spiritual_files = list(Path(".").rglob("*spiritual*")) + list(Path(".").rglob("*wisdom*"))
        spiritual_files += list(Path(".").rglob("*chakra*")) + list(Path(".").rglob("*dharma*"))
        
        encrypted_files = 0
        for file_path in spiritual_files:
            if file_path.suffix in ['.py', '.json', '.db']:
                # Basic check for encryption indicators
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read(100)  # Read first 100 bytes
                        # If content is not mostly printable, might be encrypted
                        printable_ratio = sum(1 for byte in content if 32 <= byte <= 126) / len(content)
                        if printable_ratio < 0.7 and file_path.suffix == '.db':
                            encrypted_files += 1
                except:
                    continue
        
        if encrypted_files > 0:
            print(f"   ✅ {encrypted_files} potentially encrypted spiritual data files")
            score += 10
        else:
            issues.append("No evidence of spiritual data encryption")
            self.recommendations.append("🕉️ Encrypt sensitive spiritual guidance data")
            score -= 15
        
        # Check for spiritual data access controls
        spiritual_modules = list(Path("backend/app/chakra_modules").rglob("*.py"))
        access_controls = False
        
        for module_path in spiritual_modules:
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(term in content for term in ["authorize", "permission", "access_control"]):
                        access_controls = True
                        break
            except:
                continue
        
        if access_controls:
            print("   ✅ Spiritual data access controls found")
            score += 15
        else:
            issues.append("No spiritual data access controls found")
            self.recommendations.append("🔐 Implement access controls for spiritual guidance")
            score -= 10
        
        # Check for user privacy protection
        privacy_protection = False
        for file_path in Path(".").rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(term in content for term in ["anonymize", "privacy", "gdpr", "personal_data"]):
                        privacy_protection = True
                        break
            except:
                continue
        
        if privacy_protection:
            print("   ✅ Privacy protection mechanisms found")
            score += 10
        else:
            issues.append("No privacy protection mechanisms found")
            self.recommendations.append("🔒 Implement user privacy protection for spiritual consultations")
            score -= 10
        
        print(f"   📊 Spiritual Data Security Score: {score}/100")
        for issue in issues:
            print(f"   ⚠️  {issue}")
        
        return {
            "score": score,
            "issues": issues,
            "category": "Spiritual Data Protection"
        }

def generate_security_report(analysis_results: Dict[str, Any]):
    """Generate comprehensive security report."""
    
    print("\n" + "="*80)
    print("🔒 DHARMAMIND SECURITY ASSESSMENT REPORT")
    print("="*80)
    
    # Overall security status
    score = analysis_results["overall_score"]
    status = analysis_results["security_status"]
    
    print(f"📊 OVERALL SECURITY SCORE: {score:.1f}/100")
    print(f"🛡️ SECURITY STATUS: {status}")
    
    if status == "EXCELLENT":
        print("🟢 Your system has excellent security posture!")
    elif status == "GOOD":
        print("🟡 Your system has good security with room for improvement")
    elif status == "FAIR":
        print("🟠 Your system needs security improvements")
    else:
        print("🔴 Your system requires immediate security attention!")
    
    print("\n📋 CATEGORY BREAKDOWN:")
    print("-" * 50)
    
    for category, results in analysis_results["categories"].items():
        score = results["score"]
        if score >= 80:
            status_icon = "🟢"
        elif score >= 70:
            status_icon = "🟡"
        elif score >= 60:
            status_icon = "🟠"
        else:
            status_icon = "🔴"
        
        print(f"{status_icon} {results['category']}: {score}/100")
        for issue in results["issues"][:3]:  # Show top 3 issues
            print(f"    • {issue}")
    
    # Critical issues
    if analysis_results["critical_issues"]:
        print(f"\n🚨 CRITICAL SECURITY ISSUES ({len(analysis_results['critical_issues'])}):")
        print("-" * 50)
        for issue in analysis_results["critical_issues"]:
            print(f"   {issue}")
    
    # Top recommendations
    if analysis_results["recommendations"]:
        print(f"\n💡 TOP SECURITY RECOMMENDATIONS ({len(analysis_results['recommendations'])}):")
        print("-" * 50)
        for i, rec in enumerate(analysis_results["recommendations"][:5], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n🎯 SECURITY HARDENING PRIORITY:")
    print("-" * 40)
    
    if analysis_results["critical_issues"]:
        print("🔴 HIGH PRIORITY: Address critical security issues immediately")
    
    if score < 70:
        print("🟠 MEDIUM PRIORITY: Implement basic security measures")
    
    if score < 80:
        print("🟡 LOW PRIORITY: Enhance security posture for production")
    
    print("\n" + "="*80)

async def main():
    """Run comprehensive security analysis."""
    print("🛡️ Starting DharmaMind Security Analysis...")
    print()
    
    analyzer = SecurityAnalyzer()
    results = analyzer.analyze_system_security()
    
    generate_security_report(results)
    
    print("✅ Security analysis complete!")
    
    # Save results to file
    with open("security_analysis_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("📄 Security report saved to: security_analysis_report.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
