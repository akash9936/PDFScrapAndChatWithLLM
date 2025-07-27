"""
Book Analysis Agent for Book Scraping Tool
==========================================

This module provides the main agent for book analysis and insights
using multiple model providers (Groq, Ollama, etc.).
"""

from typing import List, Dict, Optional
from ..config import Config
from ..utils.vector_store import VectorStoreAgent
from ..models.model_factory import ModelFactory
from ..models.base_provider import BaseModelProvider, ModelConfig, ModelResponse


class BookAnalysisAgent:
    """Main agent for providing book analysis and insights with multi-provider support"""
    
    def __init__(self, config: Config, vector_store: VectorStoreAgent, 
                 provider_name: str = None, model_name: str = None):
        self.config = config
        self.vector_store = vector_store
        
        # Initialize model factory
        factory_config = {
            provider: config.get_provider_config(provider)
            for provider in ['groq', 'ollama']
        }
        self.model_factory = ModelFactory(factory_config)
        
        # Set default provider and model if not specified
        self.provider_name = provider_name or config.DEFAULT_PROVIDER
        self.model_name = model_name or config.DEFAULT_MODEL
        
        # Initialize model
        self.provider = None
        self.model_config = None
        self._initialize_model()
    
    def _initialize_model(self) -> bool:
        """Initialize the selected model provider"""
        try:
            result = self.model_factory.create_model_instance(
                self.provider_name, 
                self.model_name
            )
            
            if result:
                self.provider, self.model_config = result
                return True
            else:
                print(f"Failed to initialize {self.provider_name} model: {self.model_name}")
                return False
                
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
    
    def switch_model(self, provider_name: str, model_name: str) -> bool:
        """Switch to a different model provider and model"""
        self.provider_name = provider_name
        self.model_name = model_name
        return self._initialize_model()
    
    def get_available_models(self) -> Dict[str, List]:
        """Get all available models from all providers"""
        return self.model_factory.get_all_available_models()
    
    def get_current_model_info(self) -> Dict[str, str]:
        """Get information about the currently selected model"""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "status": "initialized" if self.provider else "not_initialized"
        }
    
    def get_book_analysis(self, question: str) -> str:
        """Provide book analysis and insights based on book content"""
        print(f"🔍 Searching for insights on: {question}")
        
        # Search for relevant chunks
        relevant_chunks = self.vector_store.search_similar(question, k=5)
        
        if not relevant_chunks:
            return "I apologize, but I couldn't find relevant content in the book for your question. Please try rephrasing or ask about themes, characters, plot, or writing style."
        
        # Analyze question type for specialized response
        question_type = self._analyze_question_type(question)
        
        # Generate contextual response
        response = self._generate_response(question, relevant_chunks, question_type)
        
        return response
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of book-related question"""
        question_lower = question.lower()
        
        # Define question categories
        if any(word in question_lower for word in ['theme', 'meaning', 'message', 'purpose', 'symbolism']):
            return 'thematic'
        elif any(word in question_lower for word in ['character', 'protagonist', 'development', 'personality']):
            return 'character'
        elif any(word in question_lower for word in ['plot', 'story', 'narrative', 'events', 'sequence']):
            return 'plot'
        elif any(word in question_lower for word in ['style', 'writing', 'author', 'technique', 'language']):
            return 'style'
        elif any(word in question_lower for word in ['summary', 'overview', 'about', 'what happens']):
            return 'summary'
        else:
            return 'general'
    
    def _generate_response(self, question: str, chunks: List[Dict], question_type: str) -> str:
        """Generate contextual book analysis response"""
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"Chapter: {chunk['chapter']}\nContent: {chunk['content']}"
            for chunk in chunks[:3]  # Use top 3 most relevant
        ])
        
        # Get specialized system prompt
        system_prompt = self._get_system_prompt(question_type)
        
        # Prepare messages for the model
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"""Based on the following passages from the book:

{context}

Question: {question}

Please provide a thoughtful, insightful response that:
1. Directly addresses the question
2. References specific content from the provided passages
3. Offers clear analysis and interpretation
4. Maintains accuracy to the source material
5. Is informative and helpful

Response:"""
            }
        ]
        
        try:
            if not self.provider or not self.model_config:
                return "I apologize, but the spiritual guidance system is not properly initialized. Please check your model configuration."
            
            # Generate response using the model provider
            response = self.provider.generate_response(messages, self.model_config)
            
            if response.error:
                return f"I apologize, but I encountered an error while seeking guidance: {response.error}. Please try again or check your configuration."
            
            return response.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error while seeking guidance: {str(e)}. Please try again or check your configuration."
    
    def _get_system_prompt(self, question_type: str) -> str:
        """Get specialized system prompt based on question type"""
        
        base_prompt = """You are a knowledgeable literary analyst well-versed in book analysis. You provide insightful, accurate analysis based on the book's content."""
        
        specialized_prompts = {
            'thematic': base_prompt + " Focus on themes, meanings, symbolism, and deeper messages within the book.",
            
            'character': base_prompt + " Emphasize character development, personality traits, motivations, and relationships between characters.",
            
            'plot': base_prompt + " Guide through plot structure, narrative events, story progression, and key plot points.",
            
            'style': base_prompt + " Analyze writing style, author techniques, literary devices, and language use.",
            
            'summary': base_prompt + " Provide clear summaries and overviews of book content and key points.",
            
            'general': base_prompt + " Provide balanced book analysis drawing from comprehensive understanding of the text."
        }
        
        return specialized_prompts.get(question_type, specialized_prompts['general'])
