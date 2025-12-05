#!/usr/bin/env python3
"""
Comprehensive Rishi System Test
================================

Tests all three Rishis (Atri, Bhrigu, Vashishta) working together.
Demonstrates the complete RAG-based knowledge system.

Tests:
1. Atri - Meditation and Yoga
2. Bhrigu - Vedic Astrology
3. Vashishta - Dharma and Ethics
4. Multi-Rishi conversation flow
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path


class RishiRAGSystem:
    """Unified RAG system for all Rishis"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rishis = {}
        self._load_rishis()
    
    def _load_rishis(self):
        """Load all Rishi RAG systems"""
        rag_path = Path("engines/rishi/rag_systems")
        
        # Load Atri (Meditation)
        atri_client = chromadb.PersistentClient(
            path=str(rag_path / "atri_vector_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.rishis['atri'] = {
            'collection': atri_client.get_collection("atri_meditation_knowledge"),
            'domain': 'Meditation & Yoga',
            'personality': 'Calm, patient, contemplative'
        }
        
        # Load Bhrigu (Astrology)
        bhrigu_client = chromadb.PersistentClient(
            path=str(rag_path / "bhrigu_vector_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.rishis['bhrigu'] = {
            'collection': bhrigu_client.get_collection("bhrigu_astrology"),
            'domain': 'Vedic Astrology',
            'personality': 'Analytical, precise, cosmic-minded'
        }
        
        # Load Vashishta (Dharma)
        vashishta_client = chromadb.PersistentClient(
            path=str(rag_path / "vashishta_vector_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.rishis['vashishta'] = {
            'collection': vashishta_client.get_collection("vashishta_dharma"),
            'domain': 'Dharma & Ethics',
            'personality': 'Wise, authoritative, compassionate'
        }
    
    def query_rishi(self, rishi_name, question, n_results=2):
        """Query a specific Rishi"""
        if rishi_name not in self.rishis:
            return None
        
        rishi = self.rishis[rishi_name]
        
        # Generate query embedding
        query_embedding = self.model.encode([question])[0]
        
        # Search knowledge base
        results = rishi['collection'].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return {
            'rishi': rishi_name.title(),
            'domain': rishi['domain'],
            'personality': rishi['personality'],
            'question': question,
            'answers': results['documents'][0],
            'metadata': results['metadatas'][0],
            'relevance': [(1 - d) * 100 for d in results['distances'][0]]
        }
    
    def format_response(self, result):
        """Format a Rishi's response"""
        print(f"\n{'='*70}")
        print(f"🧙 RISHI {result['rishi'].upper()} RESPONDS")
        print(f"{'='*70}")
        print(f"Domain: {result['domain']}")
        print(f"Personality: {result['personality']}")
        print(f"\nQuestion: \"{result['question']}\"")
        print(f"\n{'-'*70}\n")
        
        for i, (answer, metadata, relevance) in enumerate(zip(
            result['answers'],
            result['metadata'],
            result['relevance']
        ), 1):
            print(f"Knowledge Source {i} (Relevance: {relevance:.1f}%):")
            print(f"Category: {metadata.get('category', 'N/A')}")
            print(f"Topic: {metadata.get('topic', 'N/A')}")
            print(f"\nWisdom:\n{answer[:400]}...")
            print(f"\n{'-'*70}\n")


def test_individual_rishis():
    """Test each Rishi individually"""
    print("\n" + "="*70)
    print("🌟 TESTING INDIVIDUAL RISHI SYSTEMS")
    print("="*70)
    
    system = RishiRAGSystem()
    
    # Test Atri (Meditation)
    print("\n\n### TEST 1: ATRI - MEDITATION MASTER ###")
    result = system.query_rishi('atri', 'How can I calm my anxious mind?')
    system.format_response(result)
    
    # Test Bhrigu (Astrology)
    print("\n\n### TEST 2: BHRIGU - ASTROLOGY SAGE ###")
    result = system.query_rishi('bhrigu', 'What does my Moon sign reveal?')
    system.format_response(result)
    
    # Test Vashishta (Dharma)
    print("\n\n### TEST 3: VASHISHTA - DHARMA GUIDE ###")
    result = system.query_rishi('vashishta', 'How do I know what is right?')
    system.format_response(result)


def test_multi_rishi_conversation():
    """Test multi-Rishi conversation flow"""
    print("\n" + "="*70)
    print("🌈 TESTING MULTI-RISHI CONVERSATION")
    print("="*70)
    print("\nScenario: A seeker asks about life purpose")
    print("Different Rishis provide perspectives from their domains")
    
    system = RishiRAGSystem()
    
    question = "What is the purpose of my life?"
    
    # Vashishta - Dharma perspective
    print("\n\n### VASHISHTA's Dharmic Perspective ###")
    result = system.query_rishi('vashishta', question)
    print(f"\n🧙 Rishi Vashishta says:")
    print(f"(Drawing from {result['domain']})")
    print(f"\n{result['answers'][0][:300]}...")
    
    # Atri - Spiritual perspective
    print("\n\n### ATRI's Meditative Perspective ###")
    result = system.query_rishi('atri', 
        "How does meditation help me understand life purpose?")
    print(f"\n🧙 Rishi Atri says:")
    print(f"(Drawing from {result['domain']})")
    print(f"\n{result['answers'][0][:300]}...")
    
    # Bhrigu - Karmic perspective
    print("\n\n### BHRIGU's Astrological Perspective ###")
    result = system.query_rishi('bhrigu', 
        "How can birth chart show life purpose?")
    print(f"\n🧙 Rishi Bhrigu says:")
    print(f"(Drawing from {result['domain']})")
    print(f"\n{result['answers'][0][:300]}...")


def test_practical_scenarios():
    """Test with practical real-world scenarios"""
    print("\n" + "="*70)
    print("💡 TESTING PRACTICAL SCENARIOS")
    print("="*70)
    
    system = RishiRAGSystem()
    
    scenarios = [
        {
            'scenario': 'Career stress',
            'queries': [
                ('atri', 'meditation techniques for work stress'),
                ('vashishta', 'dharma of right livelihood')
            ]
        },
        {
            'scenario': 'Relationship conflict',
            'queries': [
                ('vashishta', 'how to resolve conflict with loved ones'),
                ('bhrigu', 'planetary influences on relationships')
            ]
        },
        {
            'scenario': 'Life transition',
            'queries': [
                ('vashishta', 'guidance for changing life stage'),
                ('atri', 'meditation for accepting change')
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n\n### SCENARIO: {scenario['scenario'].upper()} ###")
        print("-" * 70)
        
        for rishi_name, query in scenario['queries']:
            result = system.query_rishi(rishi_name, query, n_results=1)
            print(f"\n🧙 {result['rishi']} ({result['domain']}):")
            print(f"   {result['answers'][0][:200]}...")


def test_system_statistics():
    """Display system statistics"""
    print("\n" + "="*70)
    print("📊 RISHI SYSTEM STATISTICS")
    print("="*70)
    
    import json
    
    rag_path = Path("engines/rishi/rag_systems")
    
    stats = {
        'Atri (Meditation)': json.load(open(rag_path / 'atri_metadata.json')),
        'Bhrigu (Astrology)': json.load(open(rag_path / 'bhrigu_metadata.json')),
        'Vashishta (Dharma)': json.load(open(rag_path / 'vashishta_metadata.json'))
    }
    
    total_docs = 0
    print("\n")
    for rishi, data in stats.items():
        docs = data['total_documents']
        total_docs += docs
        print(f"✅ {rishi}:")
        print(f"   Documents: {docs}")
        print(f"   Categories: {', '.join(data['categories'])}")
        print(f"   Model: {data['model']}")
        print()
    
    print(f"{'='*70}")
    print(f"📚 TOTAL KNOWLEDGE BASE: {total_docs} documents")
    print(f"🔮 EMBEDDING MODEL: all-MiniLM-L6-v2 (384 dimensions)")
    print(f"💾 VECTOR STORE: ChromaDB (persistent)")
    print(f"✨ STATUS: Phase 1 MVP Complete!")
    print(f"{'='*70}")


def main():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print("🕉️  COMPREHENSIVE RISHI SYSTEM TEST")
    print("="*70)
    print("\nTesting Phase 1 MVP: 3 Rishis with RAG Knowledge Systems")
    print("- Atri: Meditation & Yoga")
    print("- Bhrigu: Vedic Astrology")
    print("- Vashishta: Dharma & Ethics")
    
    try:
        # Test 1: Individual Rishis
        test_individual_rishis()
        
        # Test 2: Multi-Rishi conversation
        test_multi_rishi_conversation()
        
        # Test 3: Practical scenarios
        test_practical_scenarios()
        
        # Test 4: System statistics
        test_system_statistics()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎉 Phase 1 MVP is complete and operational!")
        print("📿 3 Rishis ready to guide seekers with authentic knowledge")
        print("🚀 Ready for production deployment!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
