"""
Gemini Model Provider
====================

Concrete implementation of BaseModelProvider for Google Gemini API.
Supports various Gemini models including Gemini Pro and Gemini Flash.
"""

import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .base_provider import BaseModelProvider, ModelConfig, ModelResponse


class GeminiProvider(BaseModelProvider):
    """Google Gemini API provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('GEMINI_API_KEY')
        self.client = None
        
        # Configure Gemini if API key is available
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available Gemini models"""
        return [
            ModelConfig(
                name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                max_tokens=8192,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 2,  # Free tier: 2 requests per minute
                    "rpd": 50,  # Free tier: 50 requests per day
                    "tpm": 32000,  # 32K tokens per minute
                    "tpd": 1000000,  # 1M tokens per day
                    "description": "Most capable model, best for complex reasoning and long contexts",
                    "context_window": 2000000  # 2M token context window
                }
            ),
            ModelConfig(
                name="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                max_tokens=8192,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 15,  # Free tier: 15 requests per minute
                    "rpd": 1500,  # Free tier: 1500 requests per day
                    "tpm": 1000000,  # 1M tokens per minute
                    "tpd": 50000000,  # 50M tokens per day
                    "description": "Fast and efficient model, optimized for speed",
                    "context_window": 1000000  # 1M token context window
                }
            ),
            ModelConfig(
                name="gemini-1.5-flash-8b",
                display_name="Gemini 1.5 Flash 8B",
                max_tokens=8192,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 15,  # Free tier: 15 requests per minute
                    "rpd": 1500,  # Free tier: 1500 requests per day
                    "tpm": 1000000,  # 1M tokens per minute
                    "tpd": 50000000,  # 50M tokens per day
                    "description": "Smaller, faster model with good performance",
                    "context_window": 1000000  # 1M token context window
                }
            ),
            ModelConfig(
                name="gemini-pro",
                display_name="Gemini Pro (Legacy)",
                max_tokens=4096,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 60,  # Higher rate limits for legacy model
                    "rpd": 1500,
                    "tpm": 32000,
                    "tpd": 1000000,
                    "description": "Legacy Gemini Pro model, still capable",
                    "context_window": 30720  # ~30K tokens
                }
            )
        ]
    
    def initialize_model(self, model_config: ModelConfig) -> bool:
        """Initialize Gemini model"""
        try:
            if not self.api_key:
                return False
            
            # Configure safety settings to be less restrictive for spiritual content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Generation config
            generation_config = genai.types.GenerationConfig(
                temperature=model_config.temperature,
                max_output_tokens=model_config.max_tokens,
                top_p=0.8,
                top_k=40
            )
            
            self.client = genai.GenerativeModel(
                model_name=model_config.name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return True
            
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            return False
    
    def generate_response(self, messages: List[Dict[str, str]], model_config: ModelConfig) -> ModelResponse:
        """Generate response using Gemini model"""
        try:
            if not self.client:
                if not self.initialize_model(model_config):
                    return ModelResponse(
                        content="",
                        model_used=model_config.name,
                        error="Failed to initialize Gemini model"
                    )
            
            # Format messages for Gemini
            formatted_prompt = self._format_messages_for_gemini(messages)
            
            # Generate response
            response = self.client.generate_content(formatted_prompt)
            
            # Check if response was blocked
            try:
                if (response.candidates and 
                    hasattr(response.candidates[0], 'finish_reason') and
                    str(response.candidates[0].finish_reason) == "SAFETY"):
                    return ModelResponse(
                        content="I apologize, but I cannot provide a response to that query due to safety guidelines. Please try rephrasing your question about spiritual guidance.",
                        model_used=model_config.name,
                        finish_reason="safety_blocked"
                    )
            except Exception:
                # If safety check fails, continue with normal processing
                pass
            
            # Extract response text
            if response.text:
                # Safely extract token usage information
                tokens_used = None
                try:
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        tokens_used = getattr(response.usage_metadata, 'total_token_count', None)
                except Exception:
                    # If usage metadata access fails, just continue without token count
                    pass
                
                return ModelResponse(
                    content=response.text,
                    model_used=model_config.name,
                    tokens_used=tokens_used,
                    finish_reason="completed"
                )
            else:
                return ModelResponse(
                    content="I apologize, but I couldn't generate a response. Please try again.",
                    model_used=model_config.name,
                    error="Empty response from Gemini"
                )
            
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_config.name,
                error=f"Gemini API error: {str(e)}"
            )
    
    def _format_messages_for_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Gemini prompt"""
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"Instructions: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def is_available(self) -> bool:
        """Check if Gemini provider is available"""
        if not self.api_key:
            return False
        
        try:
            # Test API connection with a simple request
            genai.configure(api_key=self.api_key)
            models = genai.list_models()
            return True
        except:
            return False
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get Gemini rate limit information"""
        return {
            "requests_per_minute": "2-60 (model dependent)",
            "requests_per_day": "50-1500 (model dependent)",
            "tokens_per_minute": "32K-1M (model dependent)",
            "tokens_per_day": "1M-50M (model dependent)",
            "note": "Free tier limits, varies significantly by model"
        }
    
    def validate_config(self) -> bool:
        """Validate Gemini configuration"""
        if not self.api_key:
            return False
        
        # Basic API key format validation (Gemini keys start with 'AI')
        if not self.api_key.startswith('AI'):
            return False
            
        return True
