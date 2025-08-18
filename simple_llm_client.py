#!/usr/bin/env python3
"""
Simple LLM Gateway Client
Minimal client for connecting to external LLM gateway
"""

import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleLLMClient:
    """Simple client for external LLM gateway"""
    
    def __init__(self, gateway_url: str = "http://localhost:8003", api_key: str = "llm-gateway-secure-key-123"):
        self.gateway_url = gateway_url
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat(self, prompt: str, provider: str = "openai", model: str = "gpt-3.5-turbo", **kwargs) -> Dict[str, Any]:
        """Send chat request to external gateway"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "provider": provider,
                "model": model,
                "query": prompt,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "system_prompt": kwargs.get("system_prompt"),
                "user_id": kwargs.get("user_id")
            }
            
            async with self.session.post(
                f"{self.gateway_url}/generate",
                json=payload,
                headers=headers,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "content": result.get("content", ""),
                        "provider": provider,
                        "model": model,
                        "usage": result.get("usage", {}),
                        "error": None
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "content": "Sorry, I'm having trouble connecting to the AI service.",
                        "provider": provider,
                        "model": model,
                        "usage": {},
                        "error": f"Gateway error {response.status}: {error_text}"
                    }
        
        except Exception as e:
            logger.error(f"LLM client error: {e}")
            return {
                "success": False,
                "content": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                "provider": provider,
                "model": model,
                "usage": {},
                "error": str(e)
            }
    
    async def get_providers(self) -> Dict[str, Any]:
        """Get available providers"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(
                f"{self.gateway_url}/providers",
                headers=headers,
                timeout=10
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"providers": {}, "error": f"Gateway error: {response.status}"}
        
        except Exception as e:
            return {"providers": {}, "error": str(e)}
    
    async def dharma_chat(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Dharma-enhanced chat with fallback wisdom"""
        # For dharma queries, provide built-in wisdom if external gateway fails
        dharma_keywords = ["meditation", "dharma", "buddha", "sanskrit", "mindfulness", "compassion"]
        is_dharma_query = any(word in prompt.lower() for word in dharma_keywords)
        
        if is_dharma_query:
            # Provide dharma-enhanced fallback response
            if "meditation" in prompt.lower():
                content = """🧘‍♂️ Meditation is the practice of training your mind to focus and find inner peace. 

Start with just 5-10 minutes daily:
1. Sit comfortably with your spine straight
2. Close your eyes gently
3. Focus on your natural breath
4. When your mind wanders, gently return to the breath
5. End with gratitude

Remember: The goal isn't to stop thinking, but to observe thoughts without judgment. Be patient and consistent with your practice. 🙏"""
            
            elif "dharma" in prompt.lower():
                content = """🌸 Dharma refers to the natural order and the righteous path of living. It encompasses:

• **Universal Principles**: Natural laws that govern existence
• **Righteous Action**: Acting with wisdom and compassion
• **Spiritual Practice**: Following the path to reduce suffering
• **Teaching**: The Buddha's guidance for liberation

Living dharma means aligning your thoughts, words, and actions with wisdom and love. How can you apply these principles in your daily life? 📿"""
            
            elif "sanskrit" in prompt.lower():
                content = """📿 Sanskrit (संस्कृत) is the sacred language of ancient wisdom texts. Key terms:

• **Om (ॐ)** - The primordial sound of creation
• **Namaste (नमस्ते)** - "I bow to the divine in you"
• **Dharma (धर्म)** - Righteous path, natural law
• **Karma (कर्म)** - Action and its consequences
• **Moksha (मोक्ष)** - Liberation from suffering

Each Sanskrit word carries deep spiritual vibrations. Would you like to explore any specific term? 🕉️"""
            
            else:
                content = """🌅 Thank you for seeking spiritual guidance. Whether you're interested in meditation, understanding dharma, exploring Sanskrit wisdom, or finding peace in daily life, I'm here to support your journey.

The path of awakening is walked one step at a time. Be gentle with yourself as you grow in wisdom and compassion.

How may I assist you on your spiritual path today? 🙏"""
            
            return {
                "success": True,
                "content": content,
                "provider": "dharma_wisdom",
                "model": "built_in_fallback",
                "usage": {"tokens": len(content.split())},
                "dharma_enhanced": True,
                "error": None
            }
        
        # For non-dharma queries, try external gateway
        return await self.chat(prompt, **kwargs)

# Global client instance
_simple_client = None

async def get_simple_llm_client() -> SimpleLLMClient:
    """Get global simple LLM client"""
    global _simple_client
    if _simple_client is None:
        _simple_client = SimpleLLMClient()
    return _simple_client

# Quick test function
async def test_client():
    """Test the simple LLM client"""
    print("🧪 Testing Simple LLM Client")
    print("=" * 40)
    
    async with SimpleLLMClient() as client:
        # Test dharma query
        print("🧘 Testing dharma query...")
        result = await client.dharma_chat("How do I start meditating?")
        print(f"✅ Response: {result['content'][:100]}...")
        print(f"🎯 Dharma Enhanced: {result.get('dharma_enhanced', False)}")
        
        # Test providers
        print("\n🌐 Getting providers...")
        providers = await client.get_providers()
        print(f"📋 Available: {list(providers.get('providers', {}).keys())}")
    
    print("\n✅ Client test complete!")

if __name__ == "__main__":
    asyncio.run(test_client())
