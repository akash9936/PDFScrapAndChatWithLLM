"""
Groq Model Provider
==================

Concrete implementation of BaseModelProvider for Groq API.
Supports various LLaMA models available on Groq's free tier.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from .base_provider import BaseModelProvider, ModelConfig, ModelResponse


class GroqProvider(BaseModelProvider):
    """Groq API provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('GROQ_API_KEY')
        self.client = None
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available Groq models"""
        return [
            ModelConfig(
                name="llama-3.3-70b-versatile",
                display_name="LLaMA 3.3 70B (Versatile)",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 30,
                    "rpd": 1000,
                    "tpm": 6000,
                    "tpd": 100000,
                    "description": "Most capable model, best for complex reasoning"
                }
            ),
            ModelConfig(
                name="llama-3.1-70b-versatile",
                display_name="LLaMA 3.1 70B (Versatile)",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 30,
                    "rpd": 14400,
                    "tpm": 6000,
                    "tpd": 500000,
                    "description": "High performance model with good reasoning"
                }
            ),
            ModelConfig(
                name="llama-3.1-8b-instant",
                display_name="LLaMA 3.1 8B (Instant)",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 30,
                    "rpd": 14400,
                    "tpm": 6000,
                    "tpd": 500000,
                    "description": "Fast and efficient, good for most tasks"
                }
            ),
            ModelConfig(
                name="gemma2-9b-it",
                display_name="Gemma 2 9B (Instruct)",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 30,
                    "rpd": 14400,
                    "tpm": 15000,
                    "tpd": 500000,
                    "description": "Google's Gemma model, good balance of speed and quality"
                }
            ),
            ModelConfig(
                name="mixtral-8x7b-32768",
                display_name="Mixtral 8x7B",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 30,
                    "rpd": 14400,
                    "tpm": 6000,
                    "tpd": 500000,
                    "description": "Mistral's mixture of experts model"
                }
            )
        ]
    
    def initialize_model(self, model_config: ModelConfig) -> bool:
        """Initialize Groq model"""
        try:
            if not self.api_key:
                return False
            
            self.client = ChatGroq(
                groq_api_key=self.api_key,
                model_name=model_config.name,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
            return True
        except Exception as e:
            print(f"Error initializing Groq model: {e}")
            return False
    
    def generate_response(self, messages: List[Dict[str, str]], model_config: ModelConfig) -> ModelResponse:
        """Generate response using Groq model"""
        try:
            if not self.client:
                if not self.initialize_model(model_config):
                    return ModelResponse(
                        content="",
                        model_used=model_config.name,
                        error="Failed to initialize Groq model"
                    )
            
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            # Generate response
            response = self.client(langchain_messages)
            
            return ModelResponse(
                content=response.content,
                model_used=model_config.name,
                finish_reason="completed"
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_config.name,
                error=f"Groq API error: {str(e)}"
            )
    
    def is_available(self) -> bool:
        """Check if Groq provider is available"""
        return bool(self.api_key)
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get Groq rate limit information"""
        return {
            "requests_per_minute": 30,
            "requests_per_day": "1,000-14,400 (model dependent)",
            "tokens_per_minute": "6,000-15,000 (model dependent)",
            "tokens_per_day": "100,000-500,000 (model dependent)",
            "note": "Free tier limits, varies by model"
        }
    
    def validate_config(self) -> bool:
        """Validate Groq configuration"""
        if not self.api_key:
            return False
        
        # Basic API key format validation
        if not self.api_key.startswith('gsk_'):
            return False
            
        return True
