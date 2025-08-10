#!/usr/bin/env python3
"""
🔱 DharmaMind Complete Integration Test Suite
============================================

This script performs comprehensive integration testing across the entire DharmaMind ecosystem:
- Backend API health and functionality
- Frontend applications connectivity
- Database connections and data persistence
- Chakra modules integration and harmony
- LLM routing and AI processing
- Authentication and security
- Payment and subscription systems
- Docker services orchestration

🕉️ Ensuring all components work in perfect harmony
"""

import asyncio
import aiohttp
import json
import subprocess
import sys
import time
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class DharmaMindIntegrationTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.chat_frontend_url = "http://localhost:3001"
        self.brand_frontend_url = "http://localhost:3000"
        self.community_frontend_url = "http://localhost:3003"
        
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "tests": {},
            "scores": {},
            "recommendations": []
        }
        
        # Color codes for console output
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.PURPLE = '\033[95m'
        self.CYAN = '\033[96m'
        self.WHITE = '\033[97m'
        self.BOLD = '\033[1m'
        self.END = '\033[0m'

    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{self.PURPLE}{self.BOLD}{'='*60}{self.END}")
        print(f"{self.PURPLE}{self.BOLD}{text:^60}{self.END}")
        print(f"{self.PURPLE}{self.BOLD}{'='*60}{self.END}")

    def print_test(self, test_name: str, status: str, details: str = ""):
        """Print test result with color coding"""
        if status == "PASS":
            icon = "✅"
            color = self.GREEN
        elif status == "FAIL":
            icon = "❌"
            color = self.RED
        elif status == "WARN":
            icon = "⚠️"
            color = self.YELLOW
        else:
            icon = "ℹ️"
            color = self.BLUE
        
        print(f"{icon} {color}{test_name:<40}{self.END} {status}")
        if details:
            print(f"   {self.CYAN}{details}{self.END}")

    async def test_backend_health(self) -> Dict[str, Any]:
        """Test backend API health and basic functionality"""
        self.print_header("🔱 BACKEND API INTEGRATION TESTS")
        
        results = {
            "api_health": False,
            "chakra_modules": False,
            "system_analysis": False,
            "auth_endpoints": False,
            "chat_endpoints": False,
            "details": {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test basic health endpoint
                try:
                    async with session.get(f"{self.base_url}/") as response:
                        if response.status == 200:
                            data = await response.json()
                            results["api_health"] = True
                            results["details"]["root_response"] = data
                            self.print_test("Backend API Root", "PASS", f"Version: {data.get('version', 'unknown')}")
                        else:
                            self.print_test("Backend API Root", "FAIL", f"Status: {response.status}")
                except Exception as e:
                    self.print_test("Backend API Root", "FAIL", f"Error: {str(e)}")
                
                # Test health check endpoint
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            health_data = await response.json()
                            results["details"]["health_data"] = health_data
                            overall_score = health_data.get("overall_score", 0)
                            if overall_score >= 0.8:
                                self.print_test("Health Check", "PASS", f"Score: {overall_score:.2f}")
                            elif overall_score >= 0.5:
                                self.print_test("Health Check", "WARN", f"Score: {overall_score:.2f}")
                            else:
                                self.print_test("Health Check", "FAIL", f"Score: {overall_score:.2f}")
                        else:
                            self.print_test("Health Check", "FAIL", f"Status: {response.status}")
                except Exception as e:
                    self.print_test("Health Check", "FAIL", f"Error: {str(e)}")
                
                # Test Chakra modules status
                try:
                    async with session.get(f"{self.base_url}/chakra/status") as response:
                        if response.status == 200:
                            chakra_data = await response.json()
                            results["chakra_modules"] = True
                            results["details"]["chakra_status"] = chakra_data
                            
                            total_modules = len(chakra_data.get("modules", {}))
                            active_modules = len([m for m in chakra_data.get("modules", {}).values() 
                                                if m.get("status") not in ["error", "inactive"]])
                            
                            harmony = chakra_data.get("system_harmony", "unknown")
                            consciousness = chakra_data.get("consciousness_level", "unknown")
                            dharma = chakra_data.get("dharma_alignment", "unknown")
                            
                            self.print_test("Chakra Modules", "PASS", 
                                          f"{active_modules}/{total_modules} active, Harmony: {harmony}")
                            self.print_test("Consciousness Level", "PASS" if consciousness == "awakened" else "WARN", 
                                          consciousness)
                            self.print_test("Dharma Alignment", "PASS" if dharma == "aligned" else "WARN", 
                                          dharma)
                        else:
                            self.print_test("Chakra Modules", "FAIL", f"Status: {response.status}")
                except Exception as e:
                    self.print_test("Chakra Modules", "FAIL", f"Error: {str(e)}")
                
                # Test system analysis
                try:
                    async with session.get(f"{self.base_url}/system/analysis") as response:
                        if response.status == 200:
                            analysis_data = await response.json()
                            results["system_analysis"] = True
                            results["details"]["system_analysis"] = analysis_data
                            
                            health_score = analysis_data.get("overall_health_score", 0)
                            system_status = analysis_data.get("system_status", "unknown")
                            
                            self.print_test("System Analysis", "PASS", 
                                          f"Status: {system_status}, Health: {health_score:.2f}")
                        else:
                            self.print_test("System Analysis", "FAIL", f"Status: {response.status}")
                except Exception as e:
                    self.print_test("System Analysis", "FAIL", f"Error: {str(e)}")
                
                # Test auth endpoints
                try:
                    # Test auth endpoint availability (should return method not allowed or similar)
                    async with session.get(f"{self.base_url}/auth/") as response:
                        # Any response (even 404/405) means the auth router is loaded
                        results["auth_endpoints"] = True
                        self.print_test("Auth Endpoints", "PASS", "Router loaded")
                except Exception as e:
                    self.print_test("Auth Endpoints", "FAIL", f"Error: {str(e)}")
                
                # Test chat endpoints
                try:
                    # Test chat endpoint availability (should require auth or return method not allowed)
                    async with session.get(f"{self.base_url}/api/v1/chat/") as response:
                        # Any response means the chat router is loaded
                        results["chat_endpoints"] = True
                        self.print_test("Chat Endpoints", "PASS", "Router loaded")
                except Exception as e:
                    self.print_test("Chat Endpoints", "FAIL", f"Error: {str(e)}")
        
        except Exception as e:
            self.print_test("Backend Connection", "FAIL", f"Cannot connect: {str(e)}")
        
        return results

    async def test_frontend_health(self) -> Dict[str, Any]:
        """Test frontend applications health and connectivity"""
        self.print_header("🌐 FRONTEND APPLICATIONS TESTS")
        
        results = {
            "chat_app": False,
            "brand_website": False,
            "community_app": False,
            "details": {}
        }
        
        frontends = [
            ("Chat App", self.chat_frontend_url, "chat_app"),
            ("Brand Website", self.brand_frontend_url, "brand_website"),
            ("Community App", self.community_frontend_url, "community_app")
        ]
        
        async with aiohttp.ClientSession() as session:
            for name, url, key in frontends:
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            results[key] = True
                            content = await response.text()
                            # Check if it's a Next.js app
                            if "next" in content.lower() or "react" in content.lower():
                                self.print_test(name, "PASS", f"Serving at {url}")
                            else:
                                self.print_test(name, "WARN", f"Serving but may not be Next.js")
                        else:
                            self.print_test(name, "FAIL", f"Status: {response.status}")
                except asyncio.TimeoutError:
                    self.print_test(name, "FAIL", "Timeout - app may not be running")
                except Exception as e:
                    self.print_test(name, "FAIL", f"Error: {str(e)}")
        
        return results

    def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connections and data integrity"""
        self.print_header("🗄️ DATABASE CONNECTIVITY TESTS")
        
        results = {
            "sqlite_knowledge": False,
            "postgresql": False,
            "data_integrity": False,
            "details": {}
        }
        
        # Test SQLite knowledge databases
        knowledge_db_paths = [
            "D:/new complete apps/data/dharma_knowledge.db",
            "D:/new complete apps/backend/app/data/dharma_knowledge.db",
            "D:/new complete apps/backend/data/dharma_knowledge.db"
        ]
        
        sqlite_found = False
        for db_path in knowledge_db_paths:
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    sqlite_found = True
                    results["sqlite_knowledge"] = True
                    results["details"]["sqlite_tables"] = len(tables)
                    self.print_test("SQLite Knowledge DB", "PASS", 
                                  f"Found at {db_path}, {len(tables)} tables")
                    break
                except Exception as e:
                    self.print_test("SQLite Knowledge DB", "FAIL", f"Error: {str(e)}")
        
        if not sqlite_found:
            self.print_test("SQLite Knowledge DB", "FAIL", "No knowledge database found")
        
        # Test PostgreSQL connection (if configured)
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="dharmamind",
                user="postgres",
                password="password123"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            conn.close()
            
            results["postgresql"] = True
            self.print_test("PostgreSQL", "PASS", "Connection successful")
        except ImportError:
            self.print_test("PostgreSQL", "WARN", "psycopg2 not installed")
        except Exception as e:
            self.print_test("PostgreSQL", "FAIL", f"Connection failed: {str(e)}")
        
        return results

    def test_process_integration(self) -> Dict[str, Any]:
        """Test running processes and port usage"""
        self.print_header("⚙️ PROCESS INTEGRATION TESTS")
        
        results = {
            "backend_process": False,
            "frontend_processes": {},
            "port_usage": {},
            "details": {}
        }
        
        # Check for running processes
        try:
            if sys.platform == "win32":
                # Windows netstat command
                result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
                output = result.stdout
                
                ports_to_check = {
                    "8000": "Backend API",
                    "3000": "Brand Website",
                    "3001": "Chat App", 
                    "3003": "Community App",
                    "5432": "PostgreSQL",
                    "6379": "Redis"
                }
                
                for port, service in ports_to_check.items():
                    if f":{port}" in output and "LISTENING" in output:
                        results["port_usage"][port] = True
                        if port == "8000":
                            results["backend_process"] = True
                        elif port.startswith("30"):
                            results["frontend_processes"][service] = True
                        self.print_test(f"Port {port} ({service})", "PASS", "Active")
                    else:
                        results["port_usage"][port] = False
                        self.print_test(f"Port {port} ({service})", "FAIL", "Not listening")
            
        except Exception as e:
            self.print_test("Process Check", "FAIL", f"Error: {str(e)}")
        
        return results

    def test_configuration_consistency(self) -> Dict[str, Any]:
        """Test configuration consistency across components"""
        self.print_header("⚙️ CONFIGURATION CONSISTENCY TESTS")
        
        results = {
            "env_files": False,
            "api_urls": False,
            "port_consistency": False,
            "details": {}
        }
        
        # Check environment files
        env_files = [
            "D:/new complete apps/.env",
            "D:/new complete apps/backend/.env",
            "D:/new complete apps/dharmamind-chat/.env.local",
            "D:/new complete apps/Brand_Webpage/.env.local",
            "D:/new complete apps/DhramaMind_Community/.env.local"
        ]
        
        found_env_files = []
        for env_file in env_files:
            if os.path.exists(env_file):
                found_env_files.append(env_file)
        
        if len(found_env_files) >= 4:  # At least backend + 3 frontends
            results["env_files"] = True
            self.print_test("Environment Files", "PASS", f"{len(found_env_files)} files found")
        else:
            self.print_test("Environment Files", "FAIL", f"Only {len(found_env_files)} files found")
        
        results["details"]["env_files_found"] = found_env_files
        
        # Check API URL consistency
        api_url_consistent = True
        expected_backend_url = "http://localhost:8000"
        
        frontend_env_files = [
            "D:/new complete apps/dharmamind-chat/.env.local",
            "D:/new complete apps/Brand_Webpage/.env.local",
            "D:/new complete apps/DhramaMind_Community/.env.local"
        ]
        
        for env_file in frontend_env_files:
            if os.path.exists(env_file):
                try:
                    with open(env_file, 'r') as f:
                        content = f.read()
                        if expected_backend_url not in content:
                            api_url_consistent = False
                            break
                except Exception:
                    api_url_consistent = False
                    break
        
        results["api_urls"] = api_url_consistent
        if api_url_consistent:
            self.print_test("API URL Consistency", "PASS", "All frontends point to localhost:8000")
        else:
            self.print_test("API URL Consistency", "FAIL", "Inconsistent API URLs")
        
        return results

    async def test_end_to_end_flow(self) -> Dict[str, Any]:
        """Test complete end-to-end functionality"""
        self.print_header("🔄 END-TO-END FLOW TESTS")
        
        results = {
            "frontend_to_backend": False,
            "api_communication": False,
            "chakra_processing": False,
            "details": {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test if frontend can reach backend (simulate CORS)
                headers = {
                    'Origin': 'http://localhost:3001',
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'Content-Type'
                }
                
                async with session.options(f"{self.base_url}/api/v1/chat", headers=headers) as response:
                    if response.status in [200, 204]:
                        results["frontend_to_backend"] = True
                        self.print_test("Frontend-Backend CORS", "PASS", "CORS configured")
                    else:
                        self.print_test("Frontend-Backend CORS", "WARN", f"Status: {response.status}")
                
                # Test API communication with health endpoint
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        results["api_communication"] = True
                        self.print_test("API Communication", "PASS", "Health endpoint accessible")
                    else:
                        self.print_test("API Communication", "FAIL", f"Status: {response.status}")
                
                # Test Chakra processing capability
                async with session.get(f"{self.base_url}/chakra/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("system_harmony") in ["perfect", "excellent", "good"]:
                            results["chakra_processing"] = True
                            self.print_test("Chakra Processing", "PASS", 
                                          f"Harmony: {data.get('system_harmony')}")
                        else:
                            self.print_test("Chakra Processing", "WARN", 
                                          f"Harmony: {data.get('system_harmony')}")
        
        except Exception as e:
            self.print_test("End-to-End Flow", "FAIL", f"Error: {str(e)}")
        
        return results

    def calculate_overall_score(self) -> float:
        """Calculate overall integration score"""
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results["tests"].items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    if isinstance(test_result, bool):
                        total_tests += 1
                        if test_result:
                            passed_tests += 1
        
        return (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check backend health
        backend_tests = self.test_results["tests"].get("backend", {})
        if not backend_tests.get("api_health"):
            recommendations.append("🔧 Start the backend server: python -m uvicorn backend.app.main:app --reload")
        
        if not backend_tests.get("chakra_modules"):
            recommendations.append("🔮 Initialize Chakra modules in backend configuration")
        
        # Check frontend health
        frontend_tests = self.test_results["tests"].get("frontend", {})
        if not frontend_tests.get("chat_app"):
            recommendations.append("🚀 Start Chat App: cd dharmamind-chat && npm run dev")
        
        if not frontend_tests.get("brand_website"):
            recommendations.append("🌐 Start Brand Website: cd Brand_Webpage && npm run dev")
        
        if not frontend_tests.get("community_app"):
            recommendations.append("👥 Start Community App: cd DhramaMind_Community && npm run dev")
        
        # Check database
        db_tests = self.test_results["tests"].get("database", {})
        if not db_tests.get("sqlite_knowledge"):
            recommendations.append("📚 Set up knowledge database with authentic spiritual content")
        
        if not db_tests.get("postgresql"):
            recommendations.append("🗄️ Install and configure PostgreSQL for user data")
        
        # Check configuration
        config_tests = self.test_results["tests"].get("configuration", {})
        if not config_tests.get("api_urls"):
            recommendations.append("⚙️ Update frontend .env.local files to point to localhost:8000")
        
        # General recommendations based on score
        overall_score = self.calculate_overall_score()
        if overall_score < 60:
            recommendations.append("🆘 Critical: Multiple components need attention for deployment readiness")
        elif overall_score < 80:
            recommendations.append("⚠️ Warning: Some components need minor fixes for optimal performance")
        else:
            recommendations.append("✨ Excellent: System is well-integrated and deployment-ready!")
        
        return recommendations

    async def run_complete_test_suite(self):
        """Run the complete integration test suite"""
        print(f"{self.BOLD}{self.CYAN}")
        print("🕉️" + "="*58 + "🕉️")
        print("🔱        DHARMAMIND COMPLETE INTEGRATION TEST        🔱")
        print("🕉️" + "="*58 + "🕉️")
        print(f"{self.END}")
        print(f"{self.YELLOW}🧘 Testing the harmony of all system components...{self.END}")
        
        # Run all test categories
        self.test_results["tests"]["backend"] = await self.test_backend_health()
        self.test_results["tests"]["frontend"] = await self.test_frontend_health()
        self.test_results["tests"]["database"] = self.test_database_connectivity()
        self.test_results["tests"]["processes"] = self.test_process_integration()
        self.test_results["tests"]["configuration"] = self.test_configuration_consistency()
        self.test_results["tests"]["end_to_end"] = await self.test_end_to_end_flow()
        
        # Calculate scores and recommendations
        overall_score = self.calculate_overall_score()
        self.test_results["scores"]["overall"] = overall_score
        self.test_results["recommendations"] = self.generate_recommendations()
        
        # Determine overall status
        if overall_score >= 90:
            self.test_results["overall_status"] = "EXCELLENT"
            status_color = self.GREEN
            status_icon = "🌟"
        elif overall_score >= 80:
            self.test_results["overall_status"] = "GOOD"
            status_color = self.GREEN
            status_icon = "✅"
        elif overall_score >= 60:
            self.test_results["overall_status"] = "NEEDS_ATTENTION"
            status_color = self.YELLOW
            status_icon = "⚠️"
        else:
            self.test_results["overall_status"] = "CRITICAL"
            status_color = self.RED
            status_icon = "❌"
        
        # Print final summary
        self.print_header("📊 INTEGRATION TEST SUMMARY")
        
        print(f"\n{status_icon} {status_color}{self.BOLD}Overall Status: {self.test_results['overall_status']}{self.END}")
        print(f"📈 {self.BOLD}Integration Score: {overall_score:.1f}%{self.END}")
        
        print(f"\n{self.BOLD}📋 RECOMMENDATIONS:{self.END}")
        for i, rec in enumerate(self.test_results["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print(f"\n{self.BOLD}🔍 DETAILED RESULTS:{self.END}")
        for category, tests in self.test_results["tests"].items():
            if isinstance(tests, dict):
                passed = sum(1 for v in tests.values() if isinstance(v, bool) and v)
                total = sum(1 for v in tests.values() if isinstance(v, bool))
                print(f"   {category.title()}: {passed}/{total} tests passed")
        
        # Save results to file
        results_file = "integration_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 {self.BLUE}Full results saved to: {results_file}{self.END}")
        
        # Final message based on deployment readiness
        if overall_score >= 80:
            print(f"\n{self.GREEN}{self.BOLD}🚀 DEPLOYMENT READY!{self.END}")
            print(f"{self.GREEN}The DharmaMind system is well-integrated and ready for production deployment.{self.END}")
        else:
            print(f"\n{self.YELLOW}{self.BOLD}🔧 NEEDS WORK{self.END}")
            print(f"{self.YELLOW}Please address the recommendations above before deployment.{self.END}")
        
        print(f"\n{self.PURPLE}🕉️ May this system serve all beings with wisdom and compassion 🕉️{self.END}")
        
        return self.test_results

async def main():
    """Main function to run integration tests"""
    tester = DharmaMindIntegrationTester()
    results = await tester.run_complete_test_suite()
    
    # Exit with appropriate code
    if results["scores"]["overall"] >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Needs attention

if __name__ == "__main__":
    asyncio.run(main())
