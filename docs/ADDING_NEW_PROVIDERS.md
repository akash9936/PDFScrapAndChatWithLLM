# Adding New Model Providers

This guide explains how to extend the Bhagavad Gita Spiritual Chatbot to support additional model providers beyond Groq, Ollama, and Gemini.

## Overview

The chatbot uses an extensible architecture with:
- **Abstract Base Provider**: `BaseModelProvider` defines the interface
- **Concrete Providers**: Implement specific APIs (Groq, Ollama, Gemini)
- **Model Factory**: Manages provider registration and instantiation
- **Configuration**: Centralized provider settings

## Steps to Add a New Provider

### 1. Create Provider Implementation

Create a new file `src/models/your_provider.py`:

```python
"""
Your Provider Implementation
===========================

Concrete implementation of BaseModelProvider for Your API.
"""

import os
from typing import List, Dict, Any, Optional
from .base_provider import BaseModelProvider, ModelConfig, ModelResponse

class YourProvider(BaseModelProvider):
    """Your API provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('YOUR_API_KEY')
        self.base_url = config.get('base_url', 'https://api.yourprovider.com')
        # Initialize your client here
    
    def get_available_models(self) -> List[ModelConfig]:
        """Return list of available models"""
        return [
            ModelConfig(
                name="your-model-name",
                display_name="Your Model Display Name",
                max_tokens=4000,
                temperature=0.7,
                provider_specific_config={
                    "rpm": 60,  # requests per minute
                    "rpd": 1000,  # requests per day
                    "description": "Model description"
                }
            )
        ]
    
    def initialize_model(self, model_config: ModelConfig) -> bool:
        """Initialize the model"""
        try:
            # Initialize your model client
            return True
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         model_config: ModelConfig) -> ModelResponse:
        """Generate response using your model"""
        try:
            # Convert messages to your API format
            # Make API call
            # Return ModelResponse
            return ModelResponse(
                content="Generated response",
                model_used=model_config.name,
                finish_reason="completed"
            )
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_config.name,
                error=f"API error: {str(e)}"
            )
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return bool(self.api_key)
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Return rate limit information"""
        return {
            "requests_per_minute": 60,
            "requests_per_day": 1000,
            "note": "Free tier limits"
        }
    
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        return bool(self.api_key)
```

### 2. Register Provider in Factory

Update `src/models/model_factory.py`:

```python
from .your_provider import YourProvider

class ModelFactory:
    PROVIDERS = {
        'groq': GroqProvider,
        'ollama': OllamaProvider,
        'gemini': GeminiProvider,
        'yourprovider': YourProvider  # Add your provider
    }
```

### 3. Update Package Imports

Update `src/models/__init__.py`:

```python
from .your_provider import YourProvider

__all__ = [
    'BaseModelProvider',
    'GroqProvider', 
    'OllamaProvider',
    'GeminiProvider',
    'YourProvider',  # Add your provider
    'ModelFactory'
]
```

### 4. Add Configuration

Update `src/config.py`:

```python
MODEL_PROVIDERS = {
    "groq": { ... },
    "ollama": { ... },
    "gemini": { ... },
    "yourprovider": {  # Add your provider config
        "api_key": os.getenv("YOUR_API_KEY"),
        "base_url": "https://api.yourprovider.com",
        "timeout": 30
    }
}
```

Update the validation method:

```python
@classmethod
def validate(cls) -> Dict[str, bool]:
    # Add validation for your provider
    factory_config = {
        provider: cls.get_provider_config(provider)
        for provider in ['groq', 'ollama', 'gemini', 'yourprovider']
    }
    
    # Add validation logic
    your_config = factory_config["yourprovider"]
    validation_results["yourprovider"] = bool(your_config.get("api_key"))
```

### 5. Update UI (Optional)

Update `src/ui/streamlit_app.py` to include your provider in the UI:

```python
provider_display_names = {
    'groq': '🚀 Groq (Fast & Free)',
    'ollama': '🏠 Ollama (Local)',
    'gemini': '🧠 Google Gemini',
    'yourprovider': '🔥 Your Provider'  # Add display name
}

# Add API key input in setup_sidebar()
elif selected_provider == 'yourprovider':
    your_key = st.sidebar.text_input(
        "Your API Key", 
        type="password",
        value=config.MODEL_PROVIDERS['yourprovider']['api_key'] or "",
        help="Get API key from your provider"
    )
    if your_key:
        os.environ["YOUR_API_KEY"] = your_key
```

### 6. Add Dependencies

Update `requirements.txt`:

```text
# Your provider dependencies
your-provider-sdk>=1.0.0
```

### 7. Update Environment Template

Update `.env.template`:

```bash
# Your Provider API Key
YOUR_API_KEY=your_api_key_here
```

## Testing Your Provider

1. **Unit Tests**: Test your provider implementation
2. **Integration Tests**: Test with the spiritual guidance agent
3. **UI Tests**: Test model selection in Streamlit

## Example: Adding OpenAI Provider

Here's a complete example for adding OpenAI support:

```python
# src/models/openai_provider.py
import openai
from .base_provider import BaseModelProvider, ModelConfig, ModelResponse

class OpenAIProvider(BaseModelProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
    
    def get_available_models(self) -> List[ModelConfig]:
        return [
            ModelConfig(
                name="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                max_tokens=4000,
                provider_specific_config={
                    "rpm": 3500,
                    "rpd": 10000,
                    "description": "Fast and efficient GPT model"
                }
            ),
            ModelConfig(
                name="gpt-4",
                display_name="GPT-4",
                max_tokens=8000,
                provider_specific_config={
                    "rpm": 500,
                    "rpd": 10000,
                    "description": "Most capable GPT model"
                }
            )
        ]
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         model_config: ModelConfig) -> ModelResponse:
        try:
            response = openai.ChatCompletion.create(
                model=model_config.name,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model_used=model_config.name,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_config.name,
                error=f"OpenAI API error: {str(e)}"
            )
    
    # ... implement other required methods
```

## Best Practices

1. **Error Handling**: Always handle API errors gracefully
2. **Rate Limiting**: Respect provider rate limits
3. **Configuration**: Use environment variables for sensitive data
4. **Validation**: Validate API keys and configurations
5. **Documentation**: Document model capabilities and limitations
6. **Testing**: Test thoroughly before deployment

## Troubleshooting

- **Import Errors**: Ensure all imports are correct in `__init__.py`
- **API Errors**: Check API key validity and network connectivity
- **Model Not Found**: Verify model names match provider specifications
- **Rate Limits**: Implement proper rate limiting and error handling
