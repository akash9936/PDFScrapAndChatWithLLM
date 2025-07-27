"""
Base Model Provider Interface
============================

Abstract base class for all model providers to ensure consistent interface
across different LLM providers (Groq, Ollama, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    display_name: str
    max_tokens: int
    temperature: float = 0.7
    provider_specific_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.provider_specific_config is None:
            self.provider_specific_config = {}


@dataclass
class ModelResponse:
    """Standardized response from model providers"""
    content: str
    model_used: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None


class BaseModelProvider(ABC):
    """Abstract base class for all model providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.replace('Provider', '').lower()
    
    @abstractmethod
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available models for this provider"""
        pass
    
    @abstractmethod
    def initialize_model(self, model_config: ModelConfig) -> bool:
        """Initialize the model with given configuration"""
        pass
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], model_config: ModelConfig) -> ModelResponse:
        """Generate response using the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured"""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limit information for this provider"""
        pass
    
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        return True
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get general information about this provider"""
        return {
            "name": self.provider_name,
            "class": self.__class__.__name__,
            "available": self.is_available(),
            "rate_limits": self.get_rate_limits()
        }
