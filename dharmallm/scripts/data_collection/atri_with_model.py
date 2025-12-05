#!/usr/bin/env python3
"""
Atri Integration with Your Trained Model
========================================

This shows how to use RAG + Your trained DharmaLLM model together.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.create_atri_rag import AtriKnowledgeRAG


class AtriWithTrainedModel:
    """Atri using RAG + Your trained DharmaLLM"""
    
    def __init__(self, model_path=None):
        # Load RAG system
        self.rag = AtriKnowledgeRAG()
        
        # Try to load your trained model
        self.model = None
        self.model_available = False
        
        if model_path:
            try:
                self.model = self._load_trained_model(model_path)
                self.model_available = True
                print("✅ Trained DharmaLLM loaded")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("   Will use template-based responses")
    
    def _load_trained_model(self, model_path):
        """Load your trained DharmaLLM model"""
        # TODO: Implement based on your model format
        # Example for transformers:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # model = AutoModelForCausalLM.from_pretrained(model_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # return (model, tokenizer)
        
        raise NotImplementedError(
            "Add your model loading code here. "
            "Check model/dharmallm-v1/ for your trained model."
        )
    
    def get_guidance(self, query: str, use_trained_model=False):
        """Get Atri's guidance
        
        Args:
            query: User's question
            use_trained_model: If True, use trained model for generation
        """
        
        # Step 1: RAG retrieval (always do this - gets authentic scriptures)
        rag_results = self.rag.query(query, n_results=3)
        scriptures = self._format_scriptures(rag_results['results'])
        
        # Step 2: Generate response
        if use_trained_model and self.model_available:
            # Use your trained model
            response = self._generate_with_model(query, scriptures)
        else:
            # Use template-based (works without model)
            response = self._generate_with_templates(query, scriptures)
        
        return response
    
    def _format_scriptures(self, results):
        """Format retrieved scriptures for model context"""
        formatted = []
        
        for result in results:
            meta = result['metadata']
            text = result['text']
            
            formatted.append({
                'type': meta.get('type', 'unknown'),
                'source': meta.get('source', 'Unknown'),
                'text': text,
                'reference': meta.get('sutra_number', meta.get('section', ''))
            })
        
        return formatted
    
    def _generate_with_model(self, query, scriptures):
        """Generate response using your trained DharmaLLM"""
        
        # Build prompt for your model
        prompt = self._build_model_prompt(query, scriptures)
        
        # Generate with your model
        # model, tokenizer = self.model
        # inputs = tokenizer(prompt, return_tensors="pt")
        # outputs = model.generate(**inputs, max_length=500)
        # response_text = tokenizer.decode(outputs[0])
        
        # For now, return template (replace with above when ready)
        return self._generate_with_templates(query, scriptures)
    
    def _build_model_prompt(self, query, scriptures):
        """Build prompt for trained model"""
        
        # Context from RAG
        context = "Retrieved Scriptures:\n\n"
        for i, scripture in enumerate(scriptures[:2], 1):
            context += f"{i}. From {scripture['source']}:\n"
            context += f"{scripture['text'][:200]}...\n\n"
        
        # Atri's personality instruction
        personality = """You are Rishi Atri, the ancient meditation master.
        
Your speaking style:
- Contemplative and gentle
- Use phrases like "*takes deep breath*", "*pauses in reflection*"
- Reference authentic scriptures (given in context)
- Provide practical meditation guidance
- End with "*bows gently*"

Respond in Atri's voice, using the retrieved scriptures."""
        
        # Full prompt
        prompt = f"""{personality}

{context}

User's Question: {query}

Atri's Response:"""
        
        return prompt
    
    def _generate_with_templates(self, query, scriptures):
        """Generate response using templates (fallback)"""
        
        # Atri's greeting
        greeting = "*takes a deep breath*\n\nNamaste, dear seeker..."
        
        # Build wisdom from scriptures
        wisdom = ""
        for scripture in scriptures[:2]:
            if scripture['type'] == 'sutra':
                wisdom += f"\n\nAs Patanjali teaches in {scripture['source']}:\n\n"
                wisdom += scripture['text'][:300]
            elif scripture['type'] == 'upanishad':
                wisdom += f"\n\nThe {scripture['source']} reveals:\n\n"
                wisdom += scripture['text'][:250]
            elif scripture['type'] == 'technique':
                wisdom += f"\n\nConsider this practice:\n\n"
                wisdom += scripture['text'][:200]
        
        # Practical steps
        practical = """

*speaks with gentle encouragement*

Practical Guidance:
• Begin with 5-10 minutes daily
• Find a quiet, comfortable space
• Let your practice unfold naturally
• Be patient with yourself"""
        
        # Closing
        closing = "\n\n*bows gently*\n\nMay your practice bring peace and clarity. 🙏"
        
        # Full response
        full_response = f"{greeting}\n{wisdom}\n{practical}\n{closing}"
        
        return {
            'rishi': 'Atri',
            'response': full_response,
            'scriptures_used': [s['source'] for s in scriptures],
            'method': 'template',
            'rag_enhanced': True
        }


def compare_approaches():
    """Compare template vs model-based responses"""
    
    print("\n" + "="*70)
    print("🧘 COMPARING APPROACHES: Template vs Trained Model")
    print("="*70 + "\n")
    
    # Initialize
    atri = AtriWithTrainedModel()
    
    # Test question
    question = "What is meditation according to yoga?"
    
    print(f"Question: {question}\n")
    print("="*70)
    
    # Approach 1: Template-based (works now)
    print("\n📝 APPROACH 1: RAG + Templates (Current)")
    print("-" * 70)
    response1 = atri.get_guidance(question, use_trained_model=False)
    print(response1['response'])
    print(f"\n✅ Method: {response1['method']}")
    print(f"📚 Sources: {', '.join(response1['scriptures_used'][:2])}")
    
    # Approach 2: With trained model (future)
    print("\n\n🤖 APPROACH 2: RAG + Trained Model (Future)")
    print("-" * 70)
    print("This will use your trained DharmaLLM model to generate")
    print("natural responses while incorporating the RAG scriptures.")
    print("\nBenefits:")
    print("• More natural conversation flow")
    print("• Better context understanding")
    print("• Adaptive to user level")
    print("• Still uses authentic scriptures from RAG")
    
    print("\n" + "="*70)
    print("\n💡 RECOMMENDATION: Start with templates, add model later")
    print("   Templates work NOW, model integration can be added anytime")


def show_integration_guide():
    """Show how to integrate with trained model"""
    
    print("\n" + "="*70)
    print("📖 HOW TO INTEGRATE WITH YOUR TRAINED MODEL")
    print("="*70 + "\n")
    
    print("""
STEP 1: Find your trained model
--------------------------------
Your model should be in: model/dharmallm-v1/

Check with:
  ls -la model/dharmallm-v1/


STEP 2: Load your model
------------------------
Uncomment the code in _load_trained_model():

  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained(model_path)
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  return (model, tokenizer)


STEP 3: Implement generation
-----------------------------
Uncomment the code in _generate_with_model():

  model, tokenizer = self.model
  inputs = tokenizer(prompt, return_tensors="pt")
  outputs = model.generate(
      **inputs,
      max_length=500,
      temperature=0.7,
      top_p=0.9
  )
  response = tokenizer.decode(outputs[0])


STEP 4: Test it
---------------
  atri = AtriWithTrainedModel("model/dharmallm-v1")
  response = atri.get_guidance("What is yoga?", use_trained_model=True)


HYBRID FLOW:
------------
User Query → RAG (get scriptures) → Trained Model (generate natural response)
             ↓                      ↓
         Authentic Sources      Natural Language + Personality
    """)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_approaches()
    elif len(sys.argv) > 1 and sys.argv[1] == "guide":
        show_integration_guide()
    else:
        print("\n🧘 Atri Integration Options:\n")
        print("  python3 atri_with_model.py compare  - Compare approaches")
        print("  python3 atri_with_model.py guide    - Integration guide")
        print()
