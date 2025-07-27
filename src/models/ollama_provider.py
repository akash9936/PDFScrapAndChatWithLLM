"""
Ollama Model Provider
====================

Concrete implementation of BaseModelProvider for Ollama local models.
Supports various open-source models that can be run locally via Ollama.
"""

import requests
import json
from typing import List, Dict, Any, Optional

from .base_provider import BaseModelProvider, ModelConfig, ModelResponse


class OllamaProvider(BaseModelProvider):
    """Ollama local model provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.timeout = config.get('timeout', 30)
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available Ollama models"""
        # Default models that work well for spiritual guidance
        default_models = [
            ModelConfig(
                name="llama3.2:latest",
                display_name="LLaMA 3.2 (Latest)",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "description": "Meta's latest LLaMA model, excellent for conversations",
                    "size": "~4.7GB",
                    "pull_command": "ollama pull llama3.2"
                }
            ),
            ModelConfig(
                name="llama3.1:8b",
                display_name="LLaMA 3.1 8B",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "description": "Efficient 8B parameter model, good balance of speed and quality",
                    "size": "~4.7GB",
                    "pull_command": "ollama pull llama3.1:8b"
                }
            ),
            ModelConfig(
                name="llama3.1:70b",
                display_name="LLaMA 3.1 70B",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "description": "Large 70B parameter model, highest quality but requires more resources",
                    "size": "~40GB",
                    "pull_command": "ollama pull llama3.1:70b"
                }
            ),
            ModelConfig(
                name="gemma2:9b",
                display_name="Gemma 2 9B",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "description": "Google's Gemma 2 model, good for instruction following",
                    "size": "~5.4GB",
                    "pull_command": "ollama pull gemma2:9b"
                }
            ),
            ModelConfig(
                name="mistral:7b",
                display_name="Mistral 7B",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "description": "Mistral's efficient 7B model, fast and capable",
                    "size": "~4.1GB",
                    "pull_command": "ollama pull mistral:7b"
                }
            ),
            ModelConfig(
                name="phi3:mini",
                display_name="Phi-3 Mini",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "description": "Microsoft's compact model, very fast and lightweight",
                    "size": "~2.3GB",
                    "pull_command": "ollama pull phi3:mini"
                }
            )
        ]
        
        # Try to get actually installed models from Ollama
        try:
            installed_models = self._get_installed_models()
            if installed_models:
                # Filter default models to only show installed ones
                installed_names = {model['name'] for model in installed_models}
                available_models = [
                    model for model in default_models 
                    if model.name in installed_names
                ]
                return available_models if available_models else default_models
        except:
            pass
        
        return default_models
    
    def _get_installed_models(self) -> List[Dict[str, Any]]:
        """Get list of models installed in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return response.json().get('models', [])
        except:
            pass
        return []
    
    def initialize_model(self, model_config: ModelConfig) -> bool:
        """Initialize Ollama model (check if available)"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if specific model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            return model_config.name in model_names
            
        except Exception as e:
            print(f"Error checking Ollama model: {e}")
            return False
    
    def generate_response(self, messages: List[Dict[str, str]], model_config: ModelConfig) -> ModelResponse:
        """Generate response using Ollama model"""
        try:
            # Prepare the prompt from messages
            prompt = self._format_messages_for_ollama(messages)
            
            # Prepare request payload
            payload = {
                "model": model_config.name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": model_config.temperature,
                    "num_predict": model_config.max_tokens
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return ModelResponse(
                    content=result.get('response', ''),
                    model_used=model_config.name,
                    finish_reason="completed"
                )
            else:
                return ModelResponse(
                    content="",
                    model_used=model_config.name,
                    error=f"Ollama API error: {response.status_code}"
                )
                
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_config.name,
                error=f"Ollama connection error: {str(e)}"
            )
    
    def _format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Ollama prompt"""
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        # Add final prompt for assistant response
        formatted_parts.append("Assistant:")
        
        return "\n\n".join(formatted_parts)
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get Ollama rate limit information"""
        return {
            "requests_per_minute": "Unlimited (local)",
            "requests_per_day": "Unlimited (local)",
            "tokens_per_minute": "Hardware dependent",
            "tokens_per_day": "Unlimited (local)",
            "note": "Local execution, limited only by hardware resources"
        }
    
    def validate_config(self) -> bool:
        """Validate Ollama configuration"""
        return self.is_available()
    
    def pull_model(self, model_name: str) -> bool:
        """Pull/download a model in Ollama"""
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # 5 minutes timeout for model download
            )
            return response.status_code == 200
        except:
            return False
