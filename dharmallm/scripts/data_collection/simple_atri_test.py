#!/usr/bin/env python3
"""
Simple Atri RAG Test - Direct Integration
==========================================

Test Atri's RAG system without the complex personality engine.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.create_atri_rag import AtriKnowledgeRAG


class SimpleAtriGuide:
    """Simple Atri guide with RAG knowledge"""
    
    def __init__(self):
        self.rag = AtriKnowledgeRAG()
        self.name = "Rishi Atri"
        
    def get_guidance(self, query: str) -> dict:
        """Get Atri's guidance on meditation"""
        
        # Step 1: Retrieve relevant knowledge
        rag_results = self.rag.query(query, n_results=3)
        
        # Step 2: Build response with Atri's personality
        response = self._build_atri_response(query, rag_results)
        
        return response
    
    def _build_atri_response(self, query: str, rag_results: dict) -> dict:
        """Build response with Atri's contemplative style"""
        
        results = rag_results['results']
        
        # Greeting (Atri's style - contemplative)
        greeting = "*takes a deep breath*\n\nNamaste, dear seeker..."
        
        # Build wisdom from retrieved knowledge
        wisdom_parts = []
        
        for i, result in enumerate(results[:2], 1):  # Top 2 most relevant
            metadata = result['metadata']
            text = result['text']
            source = metadata.get('source', 'Unknown')
            
            if metadata.get('type') == 'sutra':
                # For sutras, show Sanskrit + translation
                wisdom_parts.append(f"\nAs Patanjali teaches in {source}:\n\n{text}\n")
            elif metadata.get('type') == 'upanishad':
                wisdom_parts.append(f"\nThe {source} illuminates:\n\n{text}\n")
            elif metadata.get('type') == 'technique':
                wisdom_parts.append(f"\nConsider this practice:\n\n{text}\n")
        
        wisdom = "".join(wisdom_parts)
        
        # Practical guidance (Atri's contemplative approach)
        practical = self._get_practical_steps(results)
        
        # Closing (Atri's style)
        closing = "\n*bows gently*\n\nMay your practice bring you peace and clarity. 🙏"
        
        return {
            'rishi': 'Atri',
            'greeting': greeting,
            'wisdom': wisdom,
            'practical_steps': practical,
            'closing': closing,
            'sources': [r['metadata'].get('source', 'Unknown') for r in results],
            'full_response': f"{greeting}\n{wisdom}\n{practical}\n{closing}"
        }
    
    def _get_practical_steps(self, results: list) -> str:
        """Extract practical steps from results"""
        
        steps = []
        
        for result in results:
            if result['metadata'].get('type') == 'technique':
                # Extract technique name
                technique = result['metadata'].get('name', 'practice')
                steps.append(f"Begin with {technique}")
                break
        
        if not steps:
            steps = [
                "Find a quiet space for meditation",
                "Sit comfortably with spine straight",
                "Begin with awareness of breath"
            ]
        
        return "\n\nPractical Guidance:\n" + "\n".join(f"  • {step}" for step in steps)


def main():
    """Test the simple Atri guide"""
    
    print("\n" + "="*70)
    print("🧘 SIMPLE ATRI GUIDE - RAG Integration Test")
    print("="*70 + "\n")
    
    # Initialize
    atri = SimpleAtriGuide()
    
    # Test questions
    test_questions = [
        "What is yoga according to Patanjali?",
        "How should I begin meditation practice?",
        "What is samadhi?",
        "Teach me So'ham meditation",
        "Explain the eight limbs of yoga"
    ]
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"❓ QUESTION: {question}")
        print("="*70)
        
        # Get guidance
        response = atri.get_guidance(question)
        
        # Display
        print(response['full_response'])
        print(f"\n📚 Sources: {', '.join(response['sources'][:3])}")
        print()
        input("Press Enter for next question...")


if __name__ == "__main__":
    main()
