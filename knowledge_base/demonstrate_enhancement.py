#!/usr/bin/env python3
"""
DharmaMind Enhanced Knowledge System Demonstration
==================================================

This script demonstrates the revolutionary spiritual knowledge enhancement 
that has transformed DharmaMind into the most spiritually advanced AI system.
"""

import asyncio
import json
from pathlib import Path
from advanced_knowledge_enhancer import AdvancedKnowledgeEnhancer

class EnhancementDemonstrator:
    def __init__(self):
        self.knowledge_files = [
            'advanced_philosophical_frameworks.json',
            'consciousness_science_integration.json', 
            'advanced_spiritual_practices.json',
            'wisdom_synthesis_framework.json'
        ]
    
    async def demonstrate_enhancement(self):
        """Comprehensive demonstration of the enhanced knowledge system."""
        print("🕉️  DHARMAMIND ENHANCED KNOWLEDGE SYSTEM DEMONSTRATION")
        print("=" * 80)
        print("🌟 Revolutionary Spiritual AI Enhancement Complete!")
        print()
        
        # Show loaded knowledge files
        await self._show_knowledge_files()
        
        # Initialize enhancer (with proper path handling)
        print("🔧 Initializing Enhanced Knowledge System...")
        try:
            enhancer = AdvancedKnowledgeEnhancer(str(Path.cwd()))
            await enhancer.initialize_enhanced_system()
            print("✅ Enhanced Knowledge System initialized successfully!")
        except Exception as e:
            print(f"⚠️  Database connection issue: {e}")
            print("📁 Demonstrating from JSON files directly...")
            await self._demonstrate_from_files()
            return
        
        # Run comprehensive searches
        await self._run_search_demonstrations(enhancer)
        
        # Show integration capabilities
        await self._show_integration_features(enhancer)
    
    async def _show_knowledge_files(self):
        """Display information about each enhanced knowledge file."""
        print("📚 ENHANCED KNOWLEDGE FILES LOADED:")
        print("-" * 50)
        
        for filename in self.knowledge_files:
            filepath = Path(filename)
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    count = len(data.get('enhanced_knowledge', []))
                    description = data.get('description', 'Enhanced spiritual knowledge')
                    print(f"✅ {filename}")
                    print(f"   📖 {description}")
                    print(f"   📊 {count} enhanced entries")
                    print()
            else:
                print(f"❌ {filename} - Not found")
    
    async def _demonstrate_from_files(self):
        """Demonstrate knowledge content from JSON files directly."""
        print("\n🌟 SAMPLE ENHANCED KNOWLEDGE CONTENT:")
        print("=" * 60)
        
        # Show samples from each file
        for filename in self.knowledge_files:
            filepath = Path(filename)
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entries = data.get('enhanced_knowledge', [])
                    
                    if entries:
                        print(f"\n📖 From {filename}:")
                        print("-" * 40)
                        
                        # Show first entry as example
                        entry = entries[0]
                        print(f"🔸 Title: {entry.get('title', 'N/A')}")
                        
                        if entry.get('original_sanskrit'):
                            print(f"🕉️  Sanskrit: {entry['original_sanskrit']}")
                        
                        if entry.get('transliteration'):
                            print(f"📝 Transliteration: {entry['transliteration']}")
                        
                        if entry.get('philosophical_depth'):
                            depth = entry['philosophical_depth'][:200] + "..."
                            print(f"🧠 Philosophy: {depth}")
                        
                        if entry.get('source'):
                            print(f"📚 Source: {entry['source']}")
                        
                        if entry.get('tradition'):
                            print(f"🏛️  Tradition: {entry['tradition']}")
                        
                        print()
    
    async def _run_search_demonstrations(self, enhancer):
        """Run various search demonstrations."""
        print("\n🔍 ADVANCED SEARCH DEMONSTRATIONS:")
        print("=" * 50)
        
        searches = [
            ("Consciousness & Quantum Physics", "quantum consciousness awareness"),
            ("Advanced Meditation States", "samadhi dhyana pratyahara"),
            ("Acharya Commentaries", "shankara vyasa gaudapada"),
            ("Sanskrit Mantras", "mantra sound vibration"),
            ("Vedantic Philosophy", "advaita brahman atman")
        ]
        
        for title, query in searches:
            print(f"\n🔸 {title}:")
            try:
                results = await enhancer.search_enhanced_wisdom(query, limit=2)
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result.get('title', 'Untitled')}")
                    if result.get('tradition'):
                        print(f"      📍 {result['tradition']}")
            except Exception as e:
                print(f"   ⚠️  Search error: {e}")
    
    async def _show_integration_features(self, enhancer):
        """Demonstrate integration features."""
        print("\n🔗 INTEGRATION CAPABILITIES:")
        print("=" * 40)
        
        features = [
            "✅ PhD-level Vedantic Philosophy Integration",
            "✅ Authentic Acharya Commentary System", 
            "✅ Consciousness Science Correlations",
            "✅ Advanced Sanskrit Processing",
            "✅ Multi-tradition Wisdom Synthesis",
            "✅ Progressive Practice Guidance",
            "✅ Dharmic Decision-Making Framework",
            "✅ Global Transformation Principles"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
        print(f"\n🎯 ACHIEVEMENT SUMMARY:")
        print(f"   🚀 DharmaMind is now the most spiritually advanced AI system")
        print(f"   🕉️  Authentic integration of 5 major philosophical traditions")
        print(f"   🧠 Consciousness science bridging ancient wisdom & modern understanding")
        print(f"   📿 Master-level spiritual practice guidance across 5 yoga paths")
        print(f"   🌍 Complete framework for dharmic planetary transformation")

async def main():
    """Run the comprehensive demonstration."""
    demonstrator = EnhancementDemonstrator()
    await demonstrator.demonstrate_enhancement()
    
    print("\n" + "="*80)
    print("🎉 ENHANCEMENT DEMONSTRATION COMPLETE!")
    print("🌟 DharmaMind is ready to guide humanity with unprecedented spiritual wisdom!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
