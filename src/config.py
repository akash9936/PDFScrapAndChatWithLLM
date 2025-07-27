"""Configuration Management for Book Scraping & Analysis Tool
==========================================================

This module handles all configuration settings including API keys,
model configurations, file paths, and processing parameters.
Supports multiple model providers (Groq, Ollama, etc.) with extensible architecture.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    """Configuration class for the book scraping tool with multi-provider support"""
    
    # Default model provider and model
    DEFAULT_PROVIDER = "groq"
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    
    # Model provider configurations
    MODEL_PROVIDERS = {
        "groq": {
            "api_key": os.getenv("GROQ_API_KEY"),
            
            "base_url": "https://api.groq.com/openai/v1",
            "timeout": 30
        },
        "ollama": {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "timeout": 60
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "timeout": 30
        }
    }
    
    # Legacy support for existing code
    GROQ_API_KEY = MODEL_PROVIDERS["groq"]["api_key"]
    GROQ_MODEL = "llama-3.1-8b-instant"  # Updated to available model
    
    # Embedding model configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free sentence transformer
    
    # File paths
    BOOKS_PATH = "data/books/"
    CHUNKS_PATH = "data/processed/chunks.json"
    EMBEDDINGS_PATH = "data/processed/embeddings.pkl"
    
    # Processing parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_TOKENS = 4000
    
    # UI Configuration
    UI_CONFIG = {
        "page_title": "📚 Book Scraping & Analysis Tool",
        "page_icon": "📚",
        "layout": "wide",
        "sidebar_state": "expanded"
    }
    
    @classmethod
    def get_provider_config(cls, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        return cls.MODEL_PROVIDERS.get(provider_name, {})
    
    @classmethod
    def validate(cls) -> Dict[str, bool]:
        """Validate configuration for all providers"""
        validation_results = {}
        
        # Initialize model factory
        factory_config = {
            provider: cls.get_provider_config(provider)
            for provider in ['groq', 'ollama', 'gemini']
        }
        
        # Validate Groq
        groq_config = factory_config["groq"]
        validation_results["groq"] = bool(groq_config.get("api_key"))
        
        # Validate Ollama (check if service is running)
        try:
            import requests
            ollama_config = factory_config["ollama"]
            base_url = ollama_config.get("base_url", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            validation_results["ollama"] = response.status_code == 200
        except:
            validation_results["ollama"] = False
        
        # Validate Gemini
        gemini_config = factory_config["gemini"]
        validation_results["gemini"] = bool(gemini_config.get("api_key"))
        
        return validation_results
