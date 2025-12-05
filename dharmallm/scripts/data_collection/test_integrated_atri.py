#!/usr/bin/env python3
"""
Integrated Atri System - Personality + RAG Knowledge
====================================================

Combines Atri's authentic personality with RAG-retrieved meditation knowledge.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from engines.rishi.authentic_rishi_engine import AuthenticRishiEngine
from scripts.data_collection.create_atri_rag import AtriKnowledgeRAG


class IntegratedAtriEngine:
    """Atri with personality AND knowledge"""
    
    def __init__(self):
        # Load personality engine
        self.personality_engine = AuthenticRishiEngine()
        
        # Load RAG system
        try:
            self.rag_system = AtriKnowledgeRAG()
            self.rag_available = True
            print("✅ Atri RAG system loaded")
        except Exception as e:
            print(f"⚠️  RAG system not available: {e}")
            self.rag_available = False
    
    def get_guidance(self, query: str, context: dict = None):
        """Get Atri's guidance with knowledge + personality"""
        
        # Step 1: Retrieve relevant knowledge
        retrieved_knowledge = []
        if self.rag_available:
            rag_results = self.rag_system.query(query, n_results=3)
            retrieved_knowledge = rag_results['results']
        
        # Step 2: Build knowledge context
        knowledge_context = self._build_knowledge_context(retrieved_knowledge)
        
        # Step 3: Get personality-based response
        personality_response = self.personality_engine.get_authentic_response(
            rishi_name="atri",
            query=query,
            context=context or {}
        )
        
        # Step 4: Enhance response with retrieved knowledge
        enhanced_response = self._enhance_with_knowledge(
            personality_response,
            knowledge_context,
            retrieved_knowledge
        )
        
        return enhanced_response
    
    def _build_knowledge_context(self, retrieved_knowledge):
        """Build context string from retrieved knowledge"""
        if not retrieved_knowledge:
            return ""
        
        context = "\n\n=== RETRIEVED KNOWLEDGE ===\n"
        for i, result in enumerate(retrieved_knowledge, 1):
            source = result['metadata'].get('source', 'Unknown')
            text = result['text'][:300]  # First 300 chars
            context += f"\n{i}. From {source}:\n{text}...\n"
        
        return context
    
    def _enhance_with_knowledge(self, personality_response, 
                                knowledge_context, retrieved_knowledge):
        """Enhance personality response with actual knowledge"""
        
        # Extract primary wisdom
        primary_wisdom = personality_response['guidance']['primary_wisdom']
        
        # Add scriptural references
        scriptures = []
        for result in retrieved_knowledge:
            metadata = result['metadata']
            if metadata.get('type') in ['sutra', 'scripture']:
                scriptures.append({
                    'source': metadata.get('source', 'Unknown'),
                    'text': result['text'][:200],
                    'reference': metadata.get('sutra_number', 
                                             metadata.get('section', ''))
                })
        
        # Enhance guidance
        enhanced_wisdom = primary_wisdom
        
        if scriptures:
            enhanced_wisdom += "\n\n*speaks with the authority of scripture*\n\n"
            for scripture in scriptures[:2]:  # Top 2 most relevant
                enhanced_wisdom += f"As it is written: {scripture['text']}...\n\n"
        
        # Update response
        personality_response['guidance']['primary_wisdom'] = enhanced_wisdom
        personality_response['guidance']['scriptural_references'] = scriptures
        personality_response['rag_enhanced'] = True
        personality_response['knowledge_sources'] = len(retrieved_knowledge)
        
        return personality_response


def main():
    """Test integrated system"""
    print("\n" + "="*60)
    print("🧘 TESTING INTEGRATED ATRI SYSTEM")
    print("Personality + RAG Knowledge")
    print("="*60 + "\n")
    
    # Initialize
    atri = IntegratedAtriEngine()
    
    # Test questions
    test_questions = [
        "What is yoga according to Patanjali?",
        "How should I begin meditation practice?",
        "What is samadhi?",
        "Teach me about the eight limbs of yoga"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ {question}")
        print("="*60)
        
        # Get guidance
        response = atri.get_guidance(question)
        
        # Display
        print(f"\n{response['greeting']}\n")
        print(f"📖 Wisdom:\n{response['guidance']['primary_wisdom']}\n")
        
        if response.get('rag_enhanced'):
            print(f"✅ Enhanced with {response['knowledge_sources']} "
                  f"scriptural sources")
        
        print(f"\n🎯 Practical Steps:")
        for step in response['practical_steps'][:3]:
            print(f"  • {step}")
        
        print(f"\n🔮 Sanskrit Teaching: "
              f"{response['guidance']['sanskrit_teaching']}")
        print()


if __name__ == "__main__":
    main()
