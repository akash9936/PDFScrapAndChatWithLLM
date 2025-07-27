"""
Model Factory
=============

Factory class for creating and managing different model providers.
Provides a unified interface for model selection and initialization.
"""

from typing import Dict, List, Optional, Any
from .base_provider import BaseModelProvider, ModelConfig
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .gemini_provider import GeminiProvider


class ModelFactory:
    """Factory for creating and managing model providers"""
    
    # Registry of available providers
    PROVIDERS = {
        'groq': GroqProvider,
        'ollama': OllamaProvider,
        'gemini': GeminiProvider
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._provider_instances = {}
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.PROVIDERS.keys())
    
    def get_provider(self, provider_name: str) -> Optional[BaseModelProvider]:
        """Get or create a provider instance"""
        if provider_name not in self.PROVIDERS:
            return None
        
        if provider_name not in self._provider_instances:
            provider_class = self.PROVIDERS[provider_name]
            provider_config = self.config.get(provider_name, {})
            self._provider_instances[provider_name] = provider_class(provider_config)
        
        return self._provider_instances[provider_name]
    
    def get_available_models(self, provider_name: str) -> List[ModelConfig]:
        """Get available models for a specific provider"""
        provider = self.get_provider(provider_name)
        if provider and provider.is_available():
            return provider.get_available_models()
        return []
    
    def get_all_available_models(self) -> Dict[str, List[ModelConfig]]:
        """Get all available models from all providers"""
        all_models = {}
        for provider_name in self.PROVIDERS:
            provider = self.get_provider(provider_name)
            if provider and provider.is_available():
                all_models[provider_name] = provider.get_available_models()
        return all_models
    
    def create_model_instance(self, provider_name: str, model_name: str) -> Optional[tuple]:
        """Create a model instance with the specified provider and model
        
        Returns:
            tuple: (provider_instance, model_config) or None if not found
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return None
        
        # Find the model config
        available_models = provider.get_available_models()
        model_config = None
        for model in available_models:
            if model.name == model_name:
                model_config = model
                break
        
        if not model_config:
            return None
        
        # Initialize the model
        if provider.initialize_model(model_config):
            return provider, model_config
        
        return None
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider"""
        provider = self.get_provider(provider_name)
        if provider:
            return provider.get_provider_info()
        return None
    
    def get_all_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all providers"""
        info = {}
        for provider_name in self.PROVIDERS:
            provider_info = self.get_provider_info(provider_name)
            if provider_info:
                info[provider_name] = provider_info
        return info
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new provider class"""
        if not issubclass(provider_class, BaseModelProvider):
            raise ValueError("Provider class must inherit from BaseModelProvider")
        cls.PROVIDERS[name] = provider_class
    
    def validate_provider_config(self, provider_name: str) -> bool:
        """Validate configuration for a specific provider"""
        provider = self.get_provider(provider_name)
        if provider:
            return provider.validate_config()
        return False
