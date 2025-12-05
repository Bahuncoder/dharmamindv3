#!/usr/bin/env python3
"""
DharmaLLM Response Generator
=============================
Generates dharmic responses using trained models.
Integrates with Rishi personalities and emotional intelligence.
"""

import torch
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class DharmaResponseGenerator:
    """Generate dharmic responses using trained LLM"""
    
    def __init__(self, model, tokenizer, device=None):
        """Initialize generator
        
        Args:
            model: Loaded GPT2LMHeadModel
            tokenizer: Tokenizer
            device: Device ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
        
        self.model.eval()
        
        logger.info(f"🕉️ DharmaResponseGenerator initialized on {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """Generate response using trained model
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            num_return_sequences: Number of responses to generate
            do_sample: Whether to use sampling (vs greedy)
            
        Returns:
            List of generated responses
        """
        # Encode prompt
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode responses
        responses = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from response
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            responses.append(text)
        
        return responses
    
    def generate_dharmic_response(
        self,
        query: str,
        rishi: Optional[str] = None,
        emotion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **gen_kwargs
    ) -> Dict[str, Any]:
        """Generate dharmic response with Rishi personality and context
        
        Args:
            query: User's question
            rishi: Rishi personality (Atri, Vashishta, etc.)
            emotion: Detected emotion (sadness, anger, etc.)
            context: Additional context
            **gen_kwargs: Generation parameters
            
        Returns:
            Dict with response, metadata, and dharmic insights
        """
        # Build dharmic prompt
        prompt = self._build_dharmic_prompt(query, rishi, emotion, context)
        
        # Generate response
        responses = self.generate(prompt, **gen_kwargs)
        response_text = responses[0] if responses else ""
        
        # Post-process and add dharmic insights
        result = {
            "response": response_text,
            "query": query,
            "rishi": rishi,
            "emotion": emotion,
            "prompt_used": prompt,
            "dharmic_alignment": self._calculate_dharmic_alignment(response_text),
            "confidence": self._calculate_confidence(response_text),
            "sources": self._extract_sources(response_text),
        }
        
        return result
    
    def _build_dharmic_prompt(
        self,
        query: str,
        rishi: Optional[str] = None,
        emotion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a prompt with dharmic context"""
        
        prompt_parts = []
        
        # Add Rishi personality if specified
        if rishi:
            rishi_intros = {
                "Atri": "As Rishi Atri, the silent witness of consciousness,",
                "Vashishta": "As Rishi Vashishta, teacher of kings and dharma,",
                "Vishwamitra": "As Rishi Vishwamitra, the warrior sage,",
                "Bhrigu": "As Rishi Bhrigu, knower of cosmic patterns,",
                "Gautama": "As Rishi Gautama, master of equanimity,",
                "Jamadagni": "As Rishi Jamadagni, disciplined in tapas,",
                "Kashyapa": "As Rishi Kashyapa, father of all beings,",
            }
            intro = rishi_intros.get(rishi, f"As Rishi {rishi},")
            prompt_parts.append(intro)
        
        # Add emotional context if specified
        if emotion:
            emotion_contexts = {
                "sadness": "speaking with compassion to one in sorrow,",
                "anger": "guiding one struggling with fierce emotions,",
                "fear": "offering courage to one facing fear,",
                "anxiety": "bringing peace to a troubled mind,",
                "confusion": "illuminating the path for one uncertain,",
                "joy": "celebrating with one in happiness,",
                "love": "honoring the divine in relationships,",
                "gratitude": "deepening appreciation for blessings,",
            }
            emotion_ctx = emotion_contexts.get(
                emotion,
                f"addressing {emotion},"
            )
            prompt_parts.append(emotion_ctx)
        
        # Add dharmic instruction
        prompt_parts.append("I share wisdom from the Vedas and Upanishads.")
        
        # Add the query
        prompt_parts.append(f"\n\nQuestion: {query}\n\nDharmic Response:")
        
        return " ".join(prompt_parts)
    
    def _calculate_dharmic_alignment(self, response: str) -> float:
        """Calculate how aligned response is with dharmic principles"""
        # Simple heuristic based on keywords
        dharmic_keywords = [
            'dharma', 'karma', 'atman', 'brahman', 'yoga',
            'vedas', 'upanishads', 'bhagavad gita',
            'compassion', 'detachment', 'wisdom', 'truth',
            'consciousness', 'meditation', 'practice',
            'om', 'namaste', 'self-realization'
        ]
        
        response_lower = response.lower()
        matches = sum(1 for kw in dharmic_keywords if kw in response_lower)
        
        # Score between 0.5 and 1.0
        alignment = 0.5 + (min(matches, 10) / 20)
        return round(alignment, 2)
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate response confidence"""
        # Heuristic: longer, more detailed = higher confidence
        if not response:
            return 0.0
        
        length_score = min(len(response) / 500, 1.0)  # Up to 500 chars
        
        # Bonus for structured content
        structure_bonus = 0.0
        if ':' in response:
            structure_bonus += 0.1
        if '\n' in response:
            structure_bonus += 0.1
        
        confidence = 0.7 + (length_score * 0.2) + structure_bonus
        return round(min(confidence, 1.0), 2)
    
    def _extract_sources(self, response: str) -> List[str]:
        """Extract referenced sources from response"""
        sources = []
        
        source_keywords = {
            'bhagavad gita': 'Bhagavad Gita',
            'upanishads': 'Upanishads',
            'vedas': 'Vedas',
            'yoga sutras': 'Yoga Sutras',
            'brahma sutra': 'Brahma Sutras',
            'puranas': 'Puranas',
            'ramayana': 'Ramayana',
            'mahabharata': 'Mahabharata',
        }
        
        response_lower = response.lower()
        for keyword, source_name in source_keywords.items():
            if keyword in response_lower and source_name not in sources:
                sources.append(source_name)
        
        # Default if no sources detected
        if not sources:
            sources = ['Vedic Wisdom']
        
        return sources


class RishiIntegratedGenerator(DharmaResponseGenerator):
    """Generator with full Rishi personality integration"""
    
    def __init__(self, model, tokenizer, device=None):
        super().__init__(model, tokenizer, device)
        
        # Try to load Rishi integration
        try:
            from engines.rishi_emotional_integration import (
                RishiEmotionalIntegration
            )
            self.rishi_integration = RishiEmotionalIntegration()
            logger.info("✅ Rishi emotional integration loaded")
        except ImportError:
            self.rishi_integration = None
            logger.warning("⚠️ Rishi integration not available")
        
        # Try to load emotional intelligence
        try:
            from engines.dharmic_emotional_intelligence import (
                DharmicEmotionalIntelligence
            )
            self.emotional_engine = DharmicEmotionalIntelligence()
            logger.info("✅ Dharmic emotional intelligence loaded")
        except ImportError:
            self.emotional_engine = None
            logger.warning("⚠️ Emotional intelligence not available")
    
    def generate_with_full_context(
        self,
        query: str,
        rishi: Optional[str] = None,
        **gen_kwargs
    ) -> Dict[str, Any]:
        """Generate with full dharmic context and personality"""
        
        # Detect emotion if engine available
        emotion = None
        if self.emotional_engine:
            emotions = self.emotional_engine.detect_emotions(query)
            if emotions:
                emotion = emotions[0].value  # Primary emotion
        
        # Generate LLM response
        llm_response = self.generate_dharmic_response(
            query,
            rishi=rishi,
            emotion=emotion,
            **gen_kwargs
        )
        
        # Add Rishi personality overlay if available
        if self.rishi_integration and rishi:
            rishi_response = self.rishi_integration.get_rishi_emotional_response(
                query,
                rishi,
                emotion
            )
            
            # Combine LLM and Rishi wisdom
            llm_response['rishi_wisdom'] = rishi_response.get('teaching')
            llm_response['rishi_practices'] = rishi_response.get('practices', [])
        
        # Add emotional wisdom if available
        if self.emotional_engine and emotion:
            emotional_wisdom = self.emotional_engine.get_dharmic_wisdom(emotion)
            llm_response['sanskrit_wisdom'] = emotional_wisdom.get('verses', [])
        
        return llm_response


if __name__ == '__main__':
    # Test generator
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🕉️ Testing DharmaLLM Response Generator")
    print("=" * 70)
    
    try:
        # Load model
        from inference.model_loader import load_best_checkpoint
        
        print("\n📂 Loading trained model...")
        model, tokenizer = load_best_checkpoint()
        
        print("\n🎯 Creating generator...")
        generator = RishiIntegratedGenerator(model, tokenizer)
        
        print("\n💬 Testing generation...")
        test_query = "What is the meaning of dharma?"
        
        response = generator.generate_with_full_context(
            test_query,
            rishi="Vashishta",
            max_length=256,
            temperature=0.7
        )
        
        print("\n" + "=" * 70)
        print("📝 GENERATED RESPONSE")
        print("=" * 70)
        print(f"Query: {response['query']}")
        print(f"Rishi: {response['rishi']}")
        print(f"Response: {response['response']}")
        print(f"Dharmic Alignment: {response['dharmic_alignment']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Sources: {', '.join(response['sources'])}")
        
        print("\n✅ Generator test PASSED!")
        print("Your LLM is generating dharmic wisdom! 🕉️")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
