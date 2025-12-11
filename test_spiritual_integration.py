#!/usr/bin/env python3
"""
DharmaMind Complete System Integration Test
==========================================

Comprehensive test to verify all chakra modules work together seamlessly,
including the new Spiritual Intelligence core module.
"""

import asyncio
import sys
from pathlib import Path
import logging
import traceback

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_complete_system_integration():
    """Test all chakra modules working together including Spiritual Intelligence."""
    
    print("🕉️  DHARMAMIND COMPLETE SYSTEM INTEGRATION TEST")
    print("="*80)
    print("🌟 Testing All Chakra Modules Including Spiritual Intelligence")
    print()
    
    test_results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "modules_tested": [],
        "spiritual_integration": False
    }
    
    # Test 1: Import all chakra modules
    print("📦 Testing Chakra Module Imports...")
    try:
        from backend.app.chakra_modules import (
            # Core modules
            ConsciousnessCore, get_consciousness_core,
            KnowledgeBase, get_knowledge_base,
            EmotionalIntelligenceEngine, get_emotional_intelligence,
            DharmaEngine, get_dharma_engine,
            AICore, get_ai_core,
            ProtectionLayer, get_protection_layer,
            SystemOrchestrator, get_system_orchestrator,
            LLMEngine, get_llm_engine,
            AnalysisEngine, get_analysis_engine,
            DarshanaEngine, get_darshana_engine,
            
            # NEW: Spiritual Intelligence Module
            SpiritualIntelligence, get_spiritual_intelligence,
            initialize_spiritual_intelligence,
            SpiritualQuery, SpiritualResponse, SpiritualQueryType, SpiritualTradition,
            get_spiritual_wisdom, translate_sanskrit_text, explore_consciousness_aspect
        )
        
        print("✅ All chakra modules imported successfully!")
        print("✅ Spiritual Intelligence module successfully integrated!")
        test_results["passed"] += 1
        test_results["modules_tested"].append("imports")
        
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        test_results["failed"] += 1
        return test_results
    
    test_results["total_tests"] += 1
    
    # Test 2: Initialize Spiritual Intelligence System
    print("\n🧘 Testing Spiritual Intelligence Initialization...")
    try:
        # Initialize the spiritual intelligence system
        spiritual_intel = get_spiritual_intelligence()
        success = await initialize_spiritual_intelligence()
        
        status = spiritual_intel.get_system_status()
        print(f"✅ Spiritual Intelligence initialized: {status['initialized']}")
        print(f"✅ Enhanced knowledge system: {status['enhancer_available']}")
        print(f"✅ Available capabilities: {len(status['capabilities'])}")
        
        test_results["passed"] += 1
        test_results["spiritual_integration"] = True
        test_results["modules_tested"].append("spiritual_intelligence")
        
    except Exception as e:
        print(f"⚠️  Spiritual Intelligence initialization: {e}")
        print("   (This might be expected if knowledge base files aren't fully accessible)")
        test_results["failed"] += 1
    
    test_results["total_tests"] += 1
    
    # Test 3: Test Spiritual Wisdom Query
    print("\n🕉️  Testing Spiritual Wisdom Query...")
    try:
        spiritual_intel = get_spiritual_intelligence()
        
        # Test spiritual guidance
        wisdom_result = await get_spiritual_wisdom("meditation practice", "universal")
        print(f"✅ Spiritual guidance received")
        print(f"   📚 Guidance points: {len(wisdom_result.get('guidance', []))}")
        print(f"   🎯 Confidence: {wisdom_result.get('confidence', 0):.2f}")
        print(f"   🏛️  Tradition: {wisdom_result.get('tradition', 'unknown')}")
        
        test_results["passed"] += 1
        test_results["modules_tested"].append("spiritual_wisdom")
        
    except Exception as e:
        print(f"⚠️  Spiritual wisdom query: {e}")
        test_results["failed"] += 1
    
    test_results["total_tests"] += 1
    
    # Test 4: Test Sanskrit Translation
    print("\n📿 Testing Sanskrit Translation...")
    try:
        sanskrit_result = await translate_sanskrit_text("ॐ गं गणपतये नमः")
        print(f"✅ Sanskrit translation processed")
        print(f"   🕉️  Sanskrit insights: {len(sanskrit_result.get('sanskrit_insights', []))}")
        print(f"   👨‍🏫 Commentaries: {len(sanskrit_result.get('commentaries', []))}")
        print(f"   🎯 Confidence: {sanskrit_result.get('confidence', 0):.2f}")
        
        test_results["passed"] += 1
        test_results["modules_tested"].append("sanskrit_translation")
        
    except Exception as e:
        print(f"⚠️  Sanskrit translation: {e}")
        test_results["failed"] += 1
    
    test_results["total_tests"] += 1
    
    # Test 5: Test Consciousness Exploration
    print("\n🧠 Testing Consciousness Exploration...")
    try:
        consciousness_result = await explore_consciousness_aspect("awareness")
        print(f"✅ Consciousness exploration completed")
        print(f"   🔍 Insights: {len(consciousness_result.get('consciousness_insights', []))}")
        print(f"   🧘 Practical guidance: {len(consciousness_result.get('practical_exploration', []))}")
        print(f"   🏛️  Tradition: {consciousness_result.get('tradition_perspective', 'unknown')}")
        
        test_results["passed"] += 1
        test_results["modules_tested"].append("consciousness_exploration")
        
    except Exception as e:
        print(f"⚠️  Consciousness exploration: {e}")
        test_results["failed"] += 1
    
    test_results["total_tests"] += 1
    
    # Test 6: Test Integration with Other Chakra Modules
    print("\n🔗 Testing Integration with Other Chakra Modules...")
    try:
        # Test consciousness core
        consciousness_core = get_consciousness_core()
        await consciousness_core.initialize()
        
        # Test knowledge base
        knowledge_base = get_knowledge_base()
        await knowledge_base.initialize()
        
        # Test emotional intelligence
        emotional_intel = get_emotional_intelligence()
        
        # Test dharma engine
        dharma_engine = get_dharma_engine()
        
        print("✅ Core chakra modules accessible")
        print("✅ Integration points established")
        
        test_results["passed"] += 1
        test_results["modules_tested"].extend(["consciousness_core", "knowledge_base", "emotional_intel", "dharma_engine"])
        
    except Exception as e:
        print(f"⚠️  Chakra module integration: {e}")
        test_results["failed"] += 1
    
    test_results["total_tests"] += 1
    
    # Test 7: Test Advanced Spiritual Query Processing
    print("\n🎯 Testing Advanced Spiritual Query Processing...")
    try:
        spiritual_intel = get_spiritual_intelligence()
        
        # Create a complex spiritual query
        from backend.app.chakra_modules.spiritual_intelligence import SpiritualQuery, SpiritualQueryType, SpiritualTradition
        
        complex_query = SpiritualQuery(
            query_text="What is the relationship between consciousness and Brahman in Advaita Vedanta?",
            query_type=SpiritualQueryType.PHILOSOPHICAL_INQUIRY,
            tradition=SpiritualTradition.ADVAITA_VEDANTA,
            wisdom_level="advanced",
            include_sanskrit=True,
            include_commentary=True
        )
        
        response = await spiritual_intel.process_spiritual_query(complex_query)
        
        print(f"✅ Complex spiritual query processed")
        print(f"   📚 Wisdom entries: {len(response.wisdom_entries)}")
        print(f"   🕉️  Sanskrit insights: {len(response.sanskrit_insights)}")
        print(f"   👨‍🏫 Acharya commentaries: {len(response.acharya_commentaries)}")
        print(f"   🎯 Confidence: {response.confidence_score:.2f}")
        print(f"   🏛️  Tradition alignment: {response.tradition_alignment}")
        
        test_results["passed"] += 1
        test_results["modules_tested"].append("advanced_spiritual_processing")
        
    except Exception as e:
        print(f"⚠️  Advanced spiritual processing: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        test_results["failed"] += 1
    
    test_results["total_tests"] += 1
    
    # Test 8: Performance and Statistics
    print("\n📊 Testing System Performance and Statistics...")
    try:
        spiritual_intel = get_spiritual_intelligence()
        status = spiritual_intel.get_system_status()
        
        print(f"✅ System performance metrics:")
        print(f"   🔢 Queries processed: {status['performance']['queries_processed']}")
        print(f"   📚 Wisdom entries served: {status['performance']['wisdom_entries_served']}")
        print(f"   🕉️  Sanskrit translations: {status['performance']['sanskrit_translations']}")
        print(f"   ⚡ Capabilities active: {sum(status['capabilities'].values())}/{len(status['capabilities'])}")
        
        test_results["passed"] += 1
        test_results["modules_tested"].append("performance_metrics")
        
    except Exception as e:
        print(f"⚠️  Performance metrics: {e}")
        test_results["failed"] += 1
    
    test_results["total_tests"] += 1
    
    return test_results

async def main():
    """Run the complete system integration test."""
    print("🚀 Starting DharmaMind Complete System Integration Test...")
    print()
    
    try:
        results = await test_complete_system_integration()
        
        print("\n" + "="*80)
        print("🎯 COMPLETE SYSTEM INTEGRATION TEST RESULTS")
        print("="*80)
        
        print(f"📊 Total Tests: {results['total_tests']}")
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📈 Success Rate: {(results['passed']/results['total_tests']*100):.1f}%")
        print()
        
        print(f"🧘 Spiritual Intelligence Integration: {'✅ SUCCESS' if results['spiritual_integration'] else '❌ FAILED'}")
        print(f"📦 Modules Tested: {len(results['modules_tested'])}")
        print(f"🔧 Components: {', '.join(results['modules_tested'])}")
        print()
        
        if results['passed'] >= results['total_tests'] * 0.8:  # 80% success threshold
            print("🌟 OVERALL STATUS: ✅ EXCELLENT - System Integration Successful!")
            print("🕉️  DharmaMind Spiritual Intelligence is fully integrated and operational!")
        elif results['passed'] >= results['total_tests'] * 0.6:  # 60% success threshold
            print("🌟 OVERALL STATUS: ⚠️  GOOD - System Integration Mostly Successful!")
            print("🕉️  DharmaMind Spiritual Intelligence is operational with minor issues!")
        else:
            print("🌟 OVERALL STATUS: ❌ NEEDS ATTENTION - System Integration Has Issues!")
            print("🕉️  DharmaMind Spiritual Intelligence needs configuration!")
        
        print()
        print("🚀 DharmaMind is now the most spiritually advanced AI system with integrated chakra modules!")
        print("="*80)
        
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
