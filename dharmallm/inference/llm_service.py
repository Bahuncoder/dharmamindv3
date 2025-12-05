"""
🕉️ DharmaLLM Inference Service
==============================

Production inference service using the custom DharmaLLM.
Pure PyTorch - NO GPT-2, NO HuggingFace dependencies.

This service provides:
- Model loading and management
- Text generation with various parameters
- Rishi persona integration
- Async-compatible API

May the wisdom flow through this service! 🙏
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

import torch

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.custom_dharmallm import DharmaLLM, DharmaLLMConfig, DharmicTokenizer

logger = logging.getLogger(__name__)


class DharmaLLMService:
    """
    Production inference service for custom DharmaLLM.
    
    Features:
    - Singleton pattern for resource efficiency
    - Async model loading
    - Flexible text generation
    - Rishi persona prompting
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DharmaLLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_path: str = "trained_models/dharmallm/dharma_llm_final",
        device: Optional[str] = None
    ):
        if self._initialized:
            return
        
        self.model_path = Path(__file__).parent.parent / model_path
        self.model: Optional[DharmaLLM] = None
        self.tokenizer: Optional[DharmicTokenizer] = None
        self.device = device
        self._generation_count = 0
        self._initialized = True
        
        # Default generation settings
        self.default_generation_config = {
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True
        }
        
        logger.info("🕉️ DharmaLLM Service initialized")
    
    async def load_model(self):
        """Load the trained DharmaLLM model."""
        if self.model is not None and self.tokenizer is not None:
            logger.info("Model already loaded.")
            return
        
        logger.info(f"🔄 Loading DharmaLLM from {self.model_path}...")
        
        try:
            # Determine device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"   Using device: {self.device}")
            
            # Check if model exists
            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}")
                logger.info("   Using fallback untrained model for demonstration")
                await self._create_fallback_model()
                return
            
            # Load tokenizer
            tokenizer_path = self.model_path / 'tokenizer.json'
            if tokenizer_path.exists():
                self.tokenizer = DharmicTokenizer.load(str(tokenizer_path))
                logger.info(f"   Tokenizer loaded: {len(self.tokenizer)} tokens")
            else:
                logger.warning("Tokenizer not found, creating default")
                self.tokenizer = DharmicTokenizer(vocab_size=8000)
            
            # Load model
            self.model = DharmaLLM.load(str(self.model_path), device=self.device)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Model loaded: {self.model.num_parameters/1e6:.1f}M parameters")
            logger.info(f"   Config: {self.model.config.num_layers} layers, "
                       f"{self.model.config.hidden_size} hidden")
            
        except Exception as e:
            logger.error(f"❌ Failed to load DharmaLLM: {e}")
            logger.info("   Creating fallback model...")
            await self._create_fallback_model()
    
    async def _create_fallback_model(self):
        """Create a fallback model when trained model is not available."""
        from model.custom_dharmallm import create_model_small
        
        # Create small model
        config, self.model = create_model_small()
        self.model.to(self.device)
        self.model.eval()
        
        # Create tokenizer with minimal training
        self.tokenizer = DharmicTokenizer(vocab_size=8000)
        
        # Train on some basic dharmic text for demonstration
        sample_texts = [
            "Dharma is the cosmic law and order that makes life and universe possible.",
            "Karma is the spiritual principle of cause and effect where intent and actions influence future.",
            "Moksha is liberation from the cycle of birth and death, the ultimate goal of spiritual practice.",
            "Yoga is the union of individual consciousness with universal consciousness.",
            "Meditation brings peace and clarity to the mind, revealing the true nature of Self.",
            "The Bhagavad Gita teaches the path of selfless action and devotion.",
            "The Vedas contain timeless wisdom for spiritual awakening.",
            "Ahimsa means non-violence and compassion towards all living beings.",
            "Satya is truth in thought, word, and deed.",
            "Om is the sacred sound representing the essence of ultimate reality."
        ]
        self.tokenizer.train(sample_texts * 10, min_frequency=1)
        
        logger.info(f"✅ Fallback model created: {self.model.num_parameters/1e6:.1f}M parameters")
    
    async def generate_response(
        self,
        prompt: str,
        rishi_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using DharmaLLM.
        
        Args:
            prompt: User's input/question
            rishi_name: Optional Rishi persona to use
            **kwargs: Generation parameters (temperature, max_new_tokens, etc.)
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            await self.load_model()
        
        if self.model is None or self.tokenizer is None:
            return "🙏 DharmaLLM is currently unavailable. Please try again later."
        
        try:
            # Build prompt with Rishi persona if specified
            if rishi_name:
                full_prompt = self._build_rishi_prompt(prompt, rishi_name)
            else:
                full_prompt = f"Question: {prompt}\n\nSpiritual guidance:"
            
            # Merge generation config
            gen_config = {**self.default_generation_config, **kwargs}
            
            # Tokenize
            if self.tokenizer.trained:
                input_ids = self.tokenizer.encode(full_prompt, add_special_tokens=True)
            else:
                # Fallback for untrained tokenizer
                input_ids = [self.tokenizer.bos_token_id]
            
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=gen_config.get('max_new_tokens', 256),
                    temperature=gen_config.get('temperature', 0.7),
                    top_p=gen_config.get('top_p', 0.9),
                    top_k=gen_config.get('top_k', 50),
                    do_sample=gen_config.get('do_sample', True),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True
            )
            
            # Remove prompt from response
            if response.startswith(full_prompt):
                response = response[len(full_prompt):].strip()
            
            # Clean up
            response = self._clean_response(response)
            
            self._generation_count += 1
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._get_fallback_response(prompt, rishi_name)
    
    def _build_rishi_prompt(self, question: str, rishi_name: str) -> str:
        """Build a prompt with Rishi persona."""
        rishi_info = {
            'marichi': {
                'title': 'Sage Marichi',
                'specialty': 'cosmic wisdom and light',
                'greeting': 'As Marichi, the ray of cosmic light'
            },
            'atri': {
                'title': 'Sage Atri',
                'specialty': 'meditation and inner peace',
                'greeting': 'As Atri, master of meditation'
            },
            'angiras': {
                'title': 'Sage Angiras',
                'specialty': 'sacred mantras and rituals',
                'greeting': 'As Angiras, keeper of sacred wisdom'
            },
            'pulastya': {
                'title': 'Sage Pulastya',
                'specialty': 'creation and preservation',
                'greeting': 'As Pulastya, seer of cosmic order'
            },
            'pulaha': {
                'title': 'Sage Pulaha',
                'specialty': 'dharmic living',
                'greeting': 'As Pulaha, guide of righteous path'
            },
            'kratu': {
                'title': 'Sage Kratu',
                'specialty': 'yogic wisdom and discipline',
                'greeting': 'As Kratu, master of yoga'
            },
            'bhrigu': {
                'title': 'Sage Bhrigu',
                'specialty': 'Vedic astrology and destiny',
                'greeting': 'As Bhrigu, seer of destinies'
            },
            'vasishtha': {
                'title': 'Sage Vasishtha',
                'specialty': 'supreme knowledge and consciousness',
                'greeting': 'As Vasishtha, master of supreme wisdom'
            },
            'daksha': {
                'title': 'Sage Daksha',
                'specialty': 'creation and manifestation',
                'greeting': 'As Daksha, lord of creation'
            }
        }
        
        info = rishi_info.get(rishi_name.lower(), {
            'title': f'Sage {rishi_name}',
            'specialty': 'spiritual wisdom',
            'greeting': f'As {rishi_name}'
        })
        
        prompt = f"""You are {info['title']}, one of the nine divine Manas Putra (mind-born sons) of Lord Brahma.
Your specialty is {info['specialty']}.

A seeker asks: {question}

{info['greeting']}, I offer this guidance:"""
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove artifacts
        response = response.strip()
        
        # Remove incomplete sentences at the end
        if response and response[-1] not in '.!?"\'।॥':
            last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_period > len(response) // 2:
                response = response[:last_period + 1]
        
        return response
    
    def _get_fallback_response(self, prompt: str, rishi_name: Optional[str] = None) -> str:
        """Return a fallback response when generation fails."""
        import random
        
        fallbacks = [
            "🙏 The path of wisdom requires patience. Let us reflect on your question with deeper meditation.",
            "🕉️ As the ancient texts teach, true understanding comes through contemplation and practice.",
            "🌸 Your question touches upon profound truths. May you find guidance through sincere seeking.",
            "📿 The Rishis remind us that all answers lie within. Turn inward with devotion.",
            "🌺 Dharma unfolds in its own time. Continue your spiritual practice with faith."
        ]
        
        if rishi_name:
            return f"🙏 As {rishi_name.title()}, I guide you to seek within. {random.choice(fallbacks)}"
        
        return random.choice(fallbacks)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'is_loaded': self.model is not None,
            'model_path': str(self.model_path),
            'device': str(self.device),
            'generation_count': self._generation_count,
            'model_params': self.model.num_parameters if self.model else 0,
            'tokenizer_vocab_size': len(self.tokenizer) if self.tokenizer else 0,
            'model_type': 'Custom DharmaLLM (Pure PyTorch)'
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def get_llm_service() -> DharmaLLMService:
    """Get or create the LLM service singleton."""
    service = DharmaLLMService()
    if not service.is_loaded():
        await service.load_model()
    return service


async def generate_dharmic_response(
    prompt: str,
    rishi_name: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function to generate a response."""
    service = await get_llm_service()
    return await service.generate_response(prompt, rishi_name, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_service():
    """Test the LLM service."""
    print("🕉️ Testing DharmaLLM Service")
    print("=" * 60)
    
    service = DharmaLLMService()
    await service.load_model()
    
    print(f"\nStatus: {service.get_status()}")
    
    # Test generation
    prompts = [
        ("What is the meaning of karma?", None),
        ("How can I find inner peace?", "atri"),
        ("What does the Gita teach about action?", "vasishtha")
    ]
    
    for prompt, rishi in prompts:
        print(f"\n📿 Prompt: {prompt}")
        if rishi:
            print(f"   Rishi: {rishi}")
        
        response = await service.generate_response(prompt, rishi)
        print(f"   Response: {response[:200]}...")
    
    print("\n✅ Service test complete!")


if __name__ == "__main__":
    asyncio.run(test_service())
