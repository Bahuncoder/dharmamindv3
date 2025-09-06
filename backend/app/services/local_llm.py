"""
Local LLM Service - Run External LLMs Without API Keys

This service allows running various open-source LLMs locally:
- Hugging Face transformers models
- GGML/llama.cpp models
- Ollama models
- Local GPU/CPU inference

No API keys required - everything runs on your hardware!
"""

import asyncio
import logging
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import gc
import psutil
import time

logger = logging.getLogger(__name__)

@dataclass
class LocalModelConfig:
    """Configuration for local model"""
    model_name: str
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
class LocalLLMService:
    """Service for running LLMs locally without APIs"""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = self._get_best_device()
        logger.info(f"🔧 LocalLLMService initialized with device: {self.device}")
        
        self.model_configs = {
            # Small efficient models (good for testing)
            "microsoft/DialoGPT-small": LocalModelConfig(
                model_name="microsoft/DialoGPT-small",
                max_length=512,
                temperature=0.8
            ),
            "microsoft/DialoGPT-medium": LocalModelConfig(
                model_name="microsoft/DialoGPT-medium", 
                max_length=1024,
                temperature=0.7
            ),
            
            # Larger conversational models
            "facebook/blenderbot-400M-distill": LocalModelConfig(
                model_name="facebook/blenderbot-400M-distill",
                max_length=1024,
                temperature=0.7
            ),
            "facebook/blenderbot-1B-distill": LocalModelConfig(
                model_name="facebook/blenderbot-1B-distill",
                max_length=1024,
                temperature=0.7
            ),
            
            # Instruction-following models
            "microsoft/DialoGPT-large": LocalModelConfig(
                model_name="microsoft/DialoGPT-large",
                max_length=1024,
                temperature=0.7
            ),
            
            # Lightweight alternatives
            "distilgpt2": LocalModelConfig(
                model_name="distilgpt2",
                max_length=512,
                temperature=0.8
            ),
            "gpt2": LocalModelConfig(
                model_name="gpt2",
                max_length=1024,
                temperature=0.7
            )
        }
        
    async def initialize(self):
        """Initialize local LLM service"""
        logger.info("🤖 Initializing Local LLM Service...")
        
        try:
            # Check available hardware
            self.device = self._get_best_device()
            logger.info(f"🔧 Using device: {self.device}")
            
            # Load a default lightweight model for testing
            await self.load_model("distilgpt2")
            
            logger.info("✅ Local LLM Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Local LLM Service: {e}")
            raise
    
    def _get_best_device(self) -> str:
        """Determine best device for inference"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 CUDA available - GPU memory: {gpu_memory:.1f}GB")
            if gpu_memory >= 4:  # At least 4GB for medium models
                return "cuda"
        
        ram_gb = psutil.virtual_memory().total / 1024**3
        logger.info(f"💾 RAM available: {ram_gb:.1f}GB")
        return "cpu"
    
    async def load_model(self, model_name: str) -> bool:
        """Load a model for inference"""
        if model_name in self.loaded_models:
            logger.info(f"📦 Model {model_name} already loaded")
            return True
            
        try:
            config = self.model_configs.get(model_name)
            if not config:
                logger.error(f"❌ Unknown model: {model_name}")
                return False
                
            logger.info(f"📥 Loading model: {model_name}")
            
            # Import here to avoid startup delays
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Move to appropriate device
            if self.device == "cuda":
                model = model.to("cuda")
            
            # Create text generation pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.loaded_models[model_name] = {
                "generator": generator,
                "tokenizer": tokenizer,
                "model": model,
                "config": config
            }
            
            logger.info(f"✅ Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            return False
    
    async def generate_response(
        self,
        message: str,
        model_name: str = "distilgpt2",
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using local LLM"""
        
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            if model_name not in self.loaded_models:
                success = await self.load_model(model_name)
                if not success:
                    raise Exception(f"Failed to load model: {model_name}")
            
            model_info = self.loaded_models[model_name]
            generator = model_info["generator"]
            config = model_info["config"]
            
            # Prepare prompt
            if context:
                prompt = f"Context: {context}\\n\\nUser: {message}\\nAssistant:"
            else:
                prompt = f"User: {message}\\nAssistant:"
            
            # Generation parameters
            gen_kwargs = {
                "max_length": max_length or config.max_length,
                "temperature": temperature or config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "do_sample": config.do_sample,
                "pad_token_id": generator.tokenizer.eos_token_id,
                "num_return_sequences": 1
            }
            
            # Generate response
            logger.info(f"🔄 Generating response with {model_name}")
            outputs = generator(prompt, **gen_kwargs)
            
            # Extract response
            generated_text = outputs[0]["generated_text"]
            
            # Clean up response (remove prompt)
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            # Calculate metrics
            processing_time = time.time() - start_time
            tokens_used = len(generator.tokenizer.encode(generated_text))
            
            # Clean memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                "content": response,
                "model_name": model_name,
                "provider": "local-huggingface",
                "processing_time": processing_time,
                "tokens_used": tokens_used,
                "confidence": 0.8,  # Default confidence for local models
                "metadata": {
                    "device": self.device,
                    "parameters": gen_kwargs,
                    "prompt_length": len(prompt)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise
    
    async def get_available_models(self) -> List[str]:
        """Get list of available local models"""
        return list(self.model_configs.keys())
    
    async def unload_model(self, model_name: str):
        """Unload model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"🗑️ Model {model_name} unloaded")
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        ram = psutil.virtual_memory()
        result = {
            "ram_used_gb": (ram.total - ram.available) / 1024**3,
            "ram_total_gb": ram.total / 1024**3,
            "ram_percent": ram.percent,
            "loaded_models": list(self.loaded_models.keys())
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            result.update({
                "gpu_used_gb": gpu_memory,
                "gpu_total_gb": gpu_total,
                "gpu_percent": (gpu_memory / gpu_total) * 100
            })
        
        return result

# Global instance
local_llm_service = LocalLLMService()

async def get_local_llm_service() -> LocalLLMService:
    """Get local LLM service instance"""
    return local_llm_service
