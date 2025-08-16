#!/usr/bin/env python3
"""
Test Local LLM Service

This script tests the local LLM functionality without API keys.
Shows real text generation using Hugging Face models running locally.
"""

import asyncio
import sys
import os
import logging

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.app.services.local_llm import LocalLLMService, LocalModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_local_llm():
    """Test the local LLM service with different models"""
    
    try:
        print("🤖 Testing Local LLM Service (No API Keys Required)")
        print("=" * 60)
        
        # Initialize the service
        print("📦 Initializing Local LLM Service...")
        service = LocalLLMService()
        
        # Test message
        test_message = "Hello, can you tell me about the nature of consciousness?"
        print(f"📝 Test Message: {test_message}")
        print()
        
        # Test with lightweight DistilGPT-2 model
        print("🔥 Testing with DistilGPT-2 (Lightweight Model)")
        print("-" * 40)
        
        try:
            result = await service.generate_response(
                message=test_message,
                model_name="distilgpt2",
                max_length=128,
                temperature=0.8
            )
            
            print(f"✅ Model: {result['model_name']}")
            print(f"⚡ Device: {result['metadata']['device']}")
            print(f"⏱️ Processing Time: {result['processing_time']:.2f}s")
            print(f"🔢 Tokens Used: {result['tokens_used']}")
            print(f"📄 Response: {result['content']}")
            print()
            
        except Exception as e:
            print(f"❌ DistilGPT-2 test failed: {e}")
            print()
        
        # Test memory usage
        print("💾 Memory Usage Information")
        print("-" * 30)
        memory_info = await service.get_memory_usage()
        for key, value in memory_info.items():
            print(f"{key}: {value}")
        print()
        
        # Test available models
        print("📋 Available Models")
        print("-" * 20)
        models = await service.get_available_models()
        for model in models:
            print(f"• {model}")
        print()
        
        print("✅ Local LLM Service Test Completed Successfully!")
        print("🎉 You can now use local models without any API keys!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_local_llm())
