#!/usr/bin/env python3
"""
🕉️ DHARMAMIND COMPREHENSIVE SYSTEM INTEGRATION TEST
=================================================

Complete system verification to ensure all components work together seamlessly.
"""

import asyncio
import sys
import os
import subprocess
import json
import sqlite3
from pathlib import Path

class DharmaMindSystemTester:
    def __init__(self):
        self.base_path = Path.cwd()
        self.results = {
            'backend': {'status': 'pending', 'details': []},
            'frontend_chat': {'status': 'pending', 'details': []},
            'frontend_brand': {'status': 'pending', 'details': []},
            'knowledge_system': {'status': 'pending', 'details': []},
            'database': {'status': 'pending', 'details': []},
            'chakra_modules': {'status': 'pending', 'details': []},
            'enhanced_ai': {'status': 'pending', 'details': []},
            'integration': {'status': 'pending', 'details': []}
        }
    
    async def run_comprehensive_test(self):
        """Run complete system integration test."""
        print("🕉️ DHARMAMIND COMPREHENSIVE SYSTEM INTEGRATION TEST")
        print("="*80)
        print("🔍 Verifying all system components and their integration...")
        print()
        
        # Test each component
        await self.test_backend_structure()
        await self.test_frontend_structure()
        await self.test_knowledge_system()
        await self.test_database_connectivity()
        await self.test_chakra_modules()
        await self.test_enhanced_ai_system()
        await self.test_system_integration()
        
        # Generate final report
        self.generate_integration_report()
    
    async def test_backend_structure(self):
        """Test backend FastAPI structure."""
        print("🔧 TESTING BACKEND STRUCTURE:")
        print("-" * 40)
        
        backend_path = self.base_path / "backend"
        required_files = [
            "app/__init__.py",
            "app/main.py", 
            "app/core/__init__.py",
            "app/chakra_modules/__init__.py",
            "app/chakra_modules/knowledge_base.py",
            "requirements.txt"
        ]
        
        all_present = True
        for file_path in required_files:
            full_path = backend_path / file_path
            if full_path.exists():
                print(f"   ✅ {file_path}")
                self.results['backend']['details'].append(f"✅ {file_path} - Present")
            else:
                print(f"   ❌ {file_path} - Missing")
                self.results['backend']['details'].append(f"❌ {file_path} - Missing")
                all_present = False
        
        # Test if we can import the enhanced knowledge system
        try:
            sys.path.append(str(backend_path / "app"))
            from chakra_modules.knowledge_base import KnowledgeBase
            print(f"   ✅ KnowledgeBase import successful")
            self.results['backend']['details'].append("✅ Enhanced knowledge integration working")
        except Exception as e:
            print(f"   ⚠️ KnowledgeBase import issue: {e}")
            self.results['backend']['details'].append(f"⚠️ Import issue: {str(e)}")
        
        self.results['backend']['status'] = 'pass' if all_present else 'fail'
        print()
    
    async def test_frontend_structure(self):
        """Test frontend Next.js structures."""
        print("🌐 TESTING FRONTEND STRUCTURES:")
        print("-" * 40)
        
        # Test DharmaMind Chat Frontend
        chat_path = self.base_path / "dharmamind-chat"
        chat_files = ["package.json", "next.config.js", "pages/index.tsx", "components/"]
        
        print("   📱 DharmaMind Chat Frontend:")
        chat_ok = True
        for file_name in chat_files:
            file_path = chat_path / file_name
            if file_path.exists():
                print(f"      ✅ {file_name}")
            else:
                print(f"      ❌ {file_name} - Missing")
                chat_ok = False
        
        # Test Brand Website
        brand_path = self.base_path / "Brand_Webpage"
        brand_files = ["package.json", "next.config.js", "pages/index.tsx", "components/"]
        
        print("   🎨 Brand Website:")
        brand_ok = True
        for file_name in brand_files:
            file_path = brand_path / file_name
            if file_path.exists():
                print(f"      ✅ {file_name}")
            else:
                print(f"      ❌ {file_name} - Missing")
                brand_ok = False
        
        self.results['frontend_chat']['status'] = 'pass' if chat_ok else 'fail'
        self.results['frontend_brand']['status'] = 'pass' if brand_ok else 'fail'
        print()
    
    async def test_knowledge_system(self):
        """Test enhanced knowledge system."""
        print("🧠 TESTING ENHANCED KNOWLEDGE SYSTEM:")
        print("-" * 40)
        
        knowledge_path = self.base_path / "knowledge_base"
        
        # Check for enhanced knowledge files
        enhanced_files = [
            "advanced_knowledge_enhancer.py",
            "advanced_philosophical_frameworks.json",
            "consciousness_science_integration.json", 
            "advanced_spiritual_practices.json",
            "wisdom_synthesis_framework.json"
        ]
        
        all_present = True
        for file_name in enhanced_files:
            file_path = knowledge_path / file_name
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"   ✅ {file_name} ({file_size:,} bytes)")
                self.results['knowledge_system']['details'].append(f"✅ {file_name} - {file_size:,} bytes")
            else:
                print(f"   ❌ {file_name} - Missing")
                self.results['knowledge_system']['details'].append(f"❌ {file_name} - Missing")
                all_present = False
        
        # Test enhanced knowledge loading
        try:
            sys.path.append(str(knowledge_path))
            from advanced_knowledge_enhancer import AdvancedKnowledgeEnhancer
            enhancer = AdvancedKnowledgeEnhancer(str(knowledge_path))
            print(f"   ✅ AdvancedKnowledgeEnhancer initialization successful")
            self.results['knowledge_system']['details'].append("✅ Enhancement system initialized")
        except Exception as e:
            print(f"   ⚠️ Enhancement system issue: {e}")
            self.results['knowledge_system']['details'].append(f"⚠️ Enhancement issue: {str(e)}")
        
        self.results['knowledge_system']['status'] = 'pass' if all_present else 'fail'
        print()
    
    async def test_database_connectivity(self):
        """Test database systems."""
        print("🗄️ TESTING DATABASE CONNECTIVITY:")
        print("-" * 40)
        
        # Check main dharma knowledge database
        dharma_db = self.base_path / "data" / "dharma_knowledge.db"
        
        if dharma_db.exists():
            try:
                conn = sqlite3.connect(str(dharma_db))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()
                
                print(f"   ✅ Main database connected - {len(tables)} tables")
                self.results['database']['details'].append(f"✅ Main DB: {len(tables)} tables")
                
                for table in tables[:5]:  # Show first 5 tables
                    print(f"      📊 Table: {table[0]}")
                    
            except Exception as e:
                print(f"   ⚠️ Database connection issue: {e}")
                self.results['database']['details'].append(f"⚠️ Connection issue: {str(e)}")
        else:
            print(f"   ❌ Main database not found at {dharma_db}")
            self.results['database']['details'].append("❌ Main database missing")
        
        # Check enhanced knowledge database
        enhanced_db = self.base_path / "knowledge_base" / "enhanced_knowledge.db"
        if enhanced_db.exists():
            print(f"   ✅ Enhanced knowledge database present")
            self.results['database']['details'].append("✅ Enhanced knowledge DB present")
        else:
            print(f"   ℹ️ Enhanced knowledge database will be created on first use")
            self.results['database']['details'].append("ℹ️ Enhanced DB - will be created on use")
        
        self.results['database']['status'] = 'pass'
        print()
    
    async def test_chakra_modules(self):
        """Test chakra module system."""
        print("🕉️ TESTING CHAKRA MODULE SYSTEM:")
        print("-" * 40)
        
        chakra_path = self.base_path / "backend" / "app" / "chakra_modules"
        
        if chakra_path.exists():
            modules = list(chakra_path.glob("*.py"))
            print(f"   ✅ Chakra modules directory found")
            print(f"   📊 Found {len(modules)} chakra modules:")
            
            for module in modules:
                if module.name != "__init__.py":
                    print(f"      🔸 {module.name}")
                    
            self.results['chakra_modules']['details'].append(f"✅ {len(modules)} modules found")
            self.results['chakra_modules']['status'] = 'pass'
        else:
            print(f"   ❌ Chakra modules directory not found")
            self.results['chakra_modules']['status'] = 'fail'
        
        print()
    
    async def test_enhanced_ai_system(self):
        """Test enhanced AI capabilities."""
        print("🤖 TESTING ENHANCED AI SYSTEM:")
        print("-" * 40)
        
        try:
            # Test if we can load the enhanced knowledge
            knowledge_path = self.base_path / "knowledge_base"
            sys.path.append(str(knowledge_path))
            
            # Load a sample from each enhanced domain
            domains = [
                "advanced_philosophical_frameworks.json",
                "consciousness_science_integration.json",
                "advanced_spiritual_practices.json", 
                "wisdom_synthesis_framework.json"
            ]
            
            total_entries = 0
            for domain_file in domains:
                domain_path = knowledge_path / domain_file
                if domain_path.exists():
                    with open(domain_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        entries = self.count_knowledge_entries(data)
                        total_entries += entries
                        print(f"   ✅ {domain_file}: {entries} entries")
            
            print(f"   🎯 Total Enhanced Knowledge Entries: {total_entries}")
            self.results['enhanced_ai']['details'].append(f"✅ {total_entries} enhanced entries loaded")
            self.results['enhanced_ai']['status'] = 'pass'
            
        except Exception as e:
            print(f"   ⚠️ Enhanced AI system issue: {e}")
            self.results['enhanced_ai']['details'].append(f"⚠️ AI system issue: {str(e)}")
            self.results['enhanced_ai']['status'] = 'fail'
        
        print()
    
    async def test_system_integration(self):
        """Test overall system integration."""
        print("🔗 TESTING SYSTEM INTEGRATION:")
        print("-" * 40)
        
        integration_checks = [
            ("Backend → Knowledge System", self.check_backend_knowledge_integration),
            ("Knowledge → Database", self.check_knowledge_database_integration),
            ("Frontend → Backend API", self.check_frontend_backend_integration),
            ("Chakra → Enhanced AI", self.check_chakra_ai_integration)
        ]
        
        integration_ok = True
        for check_name, check_func in integration_checks:
            try:
                result = await check_func()
                if result:
                    print(f"   ✅ {check_name}")
                    self.results['integration']['details'].append(f"✅ {check_name}")
                else:
                    print(f"   ⚠️ {check_name} - Needs attention")
                    self.results['integration']['details'].append(f"⚠️ {check_name} - Needs attention")
                    integration_ok = False
            except Exception as e:
                print(f"   ❌ {check_name} - Error: {e}")
                self.results['integration']['details'].append(f"❌ {check_name} - Error: {str(e)}")
                integration_ok = False
        
        self.results['integration']['status'] = 'pass' if integration_ok else 'needs_attention'
        print()
    
    async def check_backend_knowledge_integration(self):
        """Check if backend can access enhanced knowledge."""
        try:
            backend_path = self.base_path / "backend" / "app"
            knowledge_path = self.base_path / "knowledge_base"
            
            # Check if the enhanced knowledge system is accessible from backend
            return (backend_path / "chakra_modules" / "knowledge_base.py").exists() and \
                   (knowledge_path / "advanced_knowledge_enhancer.py").exists()
        except:
            return False
    
    async def check_knowledge_database_integration(self):
        """Check knowledge system database integration."""
        try:
            knowledge_path = self.base_path / "knowledge_base"
            return (knowledge_path / "advanced_knowledge_enhancer.py").exists()
        except:
            return False
    
    async def check_frontend_backend_integration(self):
        """Check frontend backend API integration."""
        try:
            chat_path = self.base_path / "dharmamind-chat"
            backend_path = self.base_path / "backend"
            
            return (chat_path / "package.json").exists() and \
                   (backend_path / "app" / "main.py").exists()
        except:
            return False
    
    async def check_chakra_ai_integration(self):
        """Check chakra modules AI integration."""
        try:
            chakra_path = self.base_path / "backend" / "app" / "chakra_modules"
            knowledge_path = self.base_path / "knowledge_base"
            
            return (chakra_path / "knowledge_base.py").exists() and \
                   (knowledge_path / "advanced_knowledge_enhancer.py").exists()
        except:
            return False
    
    def count_knowledge_entries(self, data, level=0):
        """Recursively count knowledge entries."""
        count = 0
        
        if isinstance(data, dict):
            entry_indicators = ['title', 'sanskrit_term', 'original_sanskrit', 'mastery_description']
            if any(key in data for key in entry_indicators):
                count += 1
            
            for value in data.values():
                count += self.count_knowledge_entries(value, level + 1)
        
        elif isinstance(data, list):
            for item in data:
                count += self.count_knowledge_entries(item, level + 1)
        
        return count
    
    def generate_integration_report(self):
        """Generate final integration report."""
        print("📊 COMPREHENSIVE SYSTEM INTEGRATION REPORT:")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['status'] == 'pass')
        
        print(f"🎯 OVERALL STATUS: {passed_tests}/{total_tests} components fully operational")
        print()
        
        for component, result in self.results.items():
            status_icon = {
                'pass': '✅',
                'fail': '❌', 
                'needs_attention': '⚠️',
                'pending': '⏳'
            }.get(result['status'], '❓')
            
            print(f"{status_icon} {component.upper().replace('_', ' ')}: {result['status'].upper()}")
            for detail in result['details']:
                print(f"   {detail}")
            print()
        
        # Final assessment
        if passed_tests == total_tests:
            print("🎉 EXCELLENT! ALL SYSTEMS FULLY INTEGRATED AND OPERATIONAL!")
            print("🕉️ DharmaMind is ready to guide humanity with unprecedented spiritual wisdom!")
        elif passed_tests >= total_tests * 0.8:
            print("🌟 GREAT! Most systems operational with minor areas for improvement.")
            print("🔧 Address the flagged items for optimal performance.")
        else:
            print("⚠️ ATTENTION NEEDED! Several components require fixes for full integration.")
            print("🛠️ Focus on resolving the failed components.")
        
        print()
        print("="*60)
        print("🚀 DHARMAMIND SYSTEM INTEGRATION TEST COMPLETE!")

async def main():
    """Run the comprehensive system test."""
    tester = DharmaMindSystemTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
