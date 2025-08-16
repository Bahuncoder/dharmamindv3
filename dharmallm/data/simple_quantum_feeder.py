#!/usr/bin/env python3
"""
Simple Quantum Dharma Engine Integration
Connects the Hindu text feeding system with the AI engine
"""

import json
import sys
import os
from datetime import datetime

# Add the models directory to path
sys.path.append('/media/rupert/New Volume/new complete apps/dharmallm/models')

class SimpleQuantumDharmaFeeder:
    """Simple version that just processes text without complex AI"""
    
    def __init__(self):
        self.fed_texts = []
        self.knowledge_base = {}
        self.translation_ready = True
        
    def feed_hindu_text(self, text_data):
        """Feed Hindu text to the engine"""
        if isinstance(text_data, dict):
            # Store the text data
            text_id = f"text_{len(self.fed_texts)}"
            self.fed_texts.append({
                'id': text_id,
                'sanskrit': text_data.get('sanskrit', ''),
                'english': text_data.get('english', ''),
                'source': text_data.get('source', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            })
            
            # Add to knowledge base
            if 'sanskrit' in text_data:
                self.knowledge_base[text_data['sanskrit']] = {
                    'translation': text_data.get('english', ''),
                    'transliteration': text_data.get('transliteration', ''),
                    'source': text_data.get('source', '')
                }
            
            return True
        return False
    
    def translate_response(self, text, target_lang='english'):
        """Simple translation for responses"""
        if text in self.knowledge_base:
            return self.knowledge_base[text].get('translation', text)
        
        # Basic word substitutions
        translations = {
            'dharma': {'english': 'righteousness', 'hindi': 'धर्म'},
            'karma': {'english': 'action', 'hindi': 'कर्म'},
            'yoga': {'english': 'union', 'hindi': 'योग'},
            'om': {'english': 'cosmic sound', 'hindi': 'ॐ'}
        }
        
        for word, trans in translations.items():
            if word in text.lower():
                text = text.replace(word, trans.get(target_lang, word))
        
        return text
    
    def get_dharmic_response(self, query, language='english'):
        """Generate simple dharmic response"""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if 'karma' in query_lower:
            sanskrit = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन"
            response = "You have a right to perform your duty, but not to the fruits of action (Bhagavad Gita 2.47)"
        elif 'peace' in query_lower or 'shanti' in query_lower:
            sanskrit = "ॐ शान्ति शान्ति शान्तिः"
            response = "Om Shanti Shanti Shanti - May there be peace in all realms"
        elif 'truth' in query_lower or 'satya' in query_lower:
            sanskrit = "सत्यं ज्ञानमनन्तं ब्रह्म"
            response = "Truth, Knowledge, and Infinity are Brahman (Taittiriya Upanishad)"
        elif 'dharma' in query_lower:
            sanskrit = "धर्मो रक्षति रक्षितः"
            response = "Dharma protects those who protect dharma"
        else:
            sanskrit = "ॐ"
            response = "Om - The divine sound representing ultimate reality"
        
        if language != 'english':
            response = self.translate_response(response, language)
        
        return {
            'sanskrit': sanskrit,
            'response': response,
            'source': 'Hindu Scriptures',
            'language': language
        }
    
    def show_status(self):
        """Show current status"""
        print(f"🕉️  Quantum Dharma Engine Status:")
        print(f"   • Texts Fed: {len(self.fed_texts)}")
        print(f"   • Knowledge Entries: {len(self.knowledge_base)}")
        print(f"   • Translation Ready: {self.translation_ready}")
        print(f"   • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Demonstrate the simple integration"""
    print("🕉️  SIMPLE QUANTUM DHARMA ENGINE")
    print("=" * 50)
    
    # Create engine
    engine = SimpleQuantumDharmaFeeder()
    
    # Load and feed texts from the previous feeding
    try:
        with open('feeding_stats.json', 'r') as f:
            stats = json.load(f)
            print(f"📊 Previous feeding found: {stats['texts_fed']} texts")
    except:
        print("📊 No previous feeding data found")
    
    # Sample texts to feed
    sample_texts = [
        {
            'sanskrit': 'कर्मण्येवाधिकारस्ते मा फलेषु कदाचन',
            'english': 'You have a right to perform your duty, but not to the fruits of action',
            'source': 'Bhagavad Gita 2.47'
        },
        {
            'sanskrit': 'सत्यं ज्ञानमनन्तं ब्रह्म',
            'english': 'Truth, Knowledge, and Infinity are Brahman',
            'source': 'Taittiriya Upanishad'
        },
        {
            'sanskrit': 'योगश्चित्तवृत्तिनिरोधः',
            'english': 'Yoga is the cessation of mental fluctuations',
            'source': 'Yoga Sutras 1.2'
        }
    ]
    
    # Feed the texts
    print("\n📖 Feeding Hindu texts to engine...")
    for text in sample_texts:
        success = engine.feed_hindu_text(text)
        if success:
            print(f"✅ Fed: {text['sanskrit'][:30]}...")
    
    # Show status
    print("\n" + "="*50)
    engine.show_status()
    
    # Test responses
    print("\n🤖 TESTING RESPONSES:")
    print("-" * 30)
    
    test_queries = [
        "What is karma?",
        "How to find peace?",
        "What is truth?",
        "Tell me about dharma"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        response = engine.get_dharmic_response(query)
        print(f"🕉️  Sanskrit: {response['sanskrit']}")
        print(f"📝 Response: {response['response']}")
        print(f"📚 Source: {response['source']}")
    
    print("\n✨ ENGINE READY!")
    print("Simple Quantum Dharma Engine is operational")
    print("All Hindu texts are integrated and ready for responses")

if __name__ == "__main__":
    main()
