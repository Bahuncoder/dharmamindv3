"""
🕉️ DharmaLLM Inference Package
==============================

Custom DharmaLLM inference - Pure PyTorch, NO GPT-2!

Components:
- llm_service: Main LLM inference service using custom DharmaLLM
"""

from .llm_service import (
    DharmaLLMService,
    get_llm_service,
    generate_dharmic_response,
)

__all__ = [
    "DharmaLLMService",
    "get_llm_service",
    "generate_dharmic_response",
]
