"""
Streamlit UI for Book Scraping & Analysis Tool
==============================================

This module provides the web interface for the book analysis tool
using Streamlit for interactive chat functionality.
"""

import streamlit as st
import os
from pathlib import Path
from ..config import Config
from ..utils.pdf_processor import PDFProcessor
from ..utils.vector_store import VectorStoreAgent
from ..agents.spiritual_guidance import BookAnalysisAgent


def main():
    """Main Streamlit application"""
    config = Config()
    
    st.set_page_config(
        page_title=config.UI_CONFIG["page_title"],
        page_icon=config.UI_CONFIG["page_icon"],
        layout=config.UI_CONFIG["layout"],
        initial_sidebar_state=config.UI_CONFIG["sidebar_state"]
    )
    
    # Header
    st.title("📚 Book Scraping & Analysis Tool")
    st.markdown("*Analyze and explore your books with AI-powered insights*")
    
    # Sidebar for configuration and controls
    model_selection = setup_sidebar()
    
    if model_selection and len(model_selection) == 2:
        selected_provider, selected_model = model_selection
        # Check that both values are not None
        if selected_provider and selected_model:
            # Main chat interface with selected model
            chat_interface(selected_provider, selected_model)
        else:
            st.error("Please configure a model provider in the sidebar to continue.")
    else:
        st.error("Please configure a model provider in the sidebar to continue.")


def setup_sidebar():
    """Setup sidebar with configuration and controls"""
    st.sidebar.header("🔧 Configuration")
    
    config = Config()
    
    # Book Processing
    st.sidebar.subheader("📖 Data Processing")
    
    # PDF Upload Section
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Book", 
        type=['pdf'],
        help="Upload a PDF book to analyze",
        key="pdf_uploader"
    )
    
    # Store uploaded file in session state
    if uploaded_file is not None:
        st.session_state.uploaded_pdf = uploaded_file
        st.sidebar.success(f"✅ PDF uploaded: {uploaded_file.name}")
    
    # Process and Reset buttons in columns
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        process_clicked = st.button(
            "🔄 Process", 
            help="Process the uploaded PDF",
            disabled=not hasattr(st.session_state, 'uploaded_pdf'),
            use_container_width=True
        )
    
    with col2:
        reset_clicked = st.button(
            "🗑️ Reset", 
            help="Clear all data and reset",
            use_container_width=True
        )
    
    # Handle process button click
    if process_clicked and hasattr(st.session_state, 'uploaded_pdf'):
        process_uploaded_pdf(st.session_state.uploaded_pdf)
    
    # Handle reset button click
    if reset_clicked:
        reset_pdf_data()
    
    # Legacy process all PDFs button (for existing PDFs in data folder)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Or process existing PDFs:**")
    if st.sidebar.button("Process All PDFs in Data Folder", help="Extract and process all PDF files from data directory"):
        process_pdf_data()
    
    # Check if data is processed
    if Path(config.CHUNKS_PATH).exists():
        st.sidebar.success("✅ Books processed and ready")
        
        # Show data statistics
        try:
            import json
            with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            st.sidebar.info(f"📊 {len(chunks)} text chunks available")
        except:
            pass
    else:
        st.sidebar.warning("⚠️ Please process books first")
    
    # Model Provider Selection
    st.sidebar.subheader("🤖 Model Configuration")
    
    # Get available providers and their validation status
    validation_results = config.validate()
    
    # Provider selection
    available_providers = []
    provider_display_names = {
        'groq': '🚀 Groq (Fast & Free)',
        'ollama': '🏠 Ollama (Local)',
        'gemini': '🧠 Google Gemini'
    }
    
    for provider, is_valid in validation_results.items():
        if is_valid:
            available_providers.append(provider)
    
    if not available_providers:
        st.sidebar.error("❌ No model providers available. Please configure API keys or start Ollama.")
        return None, None
    
    # Provider selection dropdown
    selected_provider = st.sidebar.selectbox(
        "Select Model Provider",
        available_providers,
        format_func=lambda x: provider_display_names.get(x, x.title()),
        key="provider_selection"
    )
    
    # Model selection based on provider
    from ..models.model_factory import ModelFactory
    factory_config = {
        provider: config.get_provider_config(provider)
        for provider in ['groq', 'ollama', 'gemini']
    }
    model_factory = ModelFactory(factory_config)
    
    available_models = model_factory.get_available_models(selected_provider)
    
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            [model.name for model in available_models],
            format_func=lambda x: next((model.display_name for model in available_models if model.name == x), x),
            key="model_selection"
        )
        
        # Show model information
        selected_model_config = next((model for model in available_models if model.name == selected_model), None)
        if selected_model_config:
            st.sidebar.info(f"**Model**: {selected_model_config.display_name}")
            
            # Show provider-specific info
            provider_config = selected_model_config.provider_specific_config
            if provider_config:
                if 'description' in provider_config:
                    st.sidebar.caption(provider_config['description'])
                
                # Show rate limits for API providers
                if selected_provider in ['groq', 'gemini']:
                    with st.sidebar.expander("ℹ️ Rate Limits"):
                        if 'rpm' in provider_config:
                            st.write(f"• **Requests/min**: {provider_config['rpm']}")
                        if 'rpd' in provider_config:
                            st.write(f"• **Requests/day**: {provider_config['rpd']}")
                        if 'tpm' in provider_config:
                            st.write(f"• **Tokens/min**: {provider_config['tpm']}")
    else:
        st.sidebar.error(f"❌ No models available for {selected_provider}")
        return None, None
    
    # API Key configuration based on selected provider
    st.sidebar.subheader("🔑 API Configuration")
    
    if selected_provider == 'groq':
        groq_key = st.sidebar.text_input(
            "Groq API Key", 
            type="password",
            value=config.MODEL_PROVIDERS['groq']['api_key'],
            help="Get free API key from console.groq.com"
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
    
    elif selected_provider == 'gemini':
        gemini_key = st.sidebar.text_input(
            "Gemini API Key", 
            type="password",
            value=config.MODEL_PROVIDERS['gemini']['api_key'] or "",
            help="Get free API key from aistudio.google.com"
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
    
    elif selected_provider == 'ollama':
        ollama_url = st.sidebar.text_input(
            "Ollama Base URL",
            value=config.MODEL_PROVIDERS['ollama']['base_url'],
            help="URL where Ollama is running (default: http://localhost:11434)"
        )
        if ollama_url:
            os.environ["OLLAMA_BASE_URL"] = ollama_url
        
        # Show Ollama-specific info
        st.sidebar.info("💡 Make sure Ollama is running and the selected model is pulled")
        if selected_model_config and 'pull_command' in selected_model_config.provider_specific_config:
            st.sidebar.code(selected_model_config.provider_specific_config['pull_command'])
    
    # Show embedding model info
    st.sidebar.subheader("🔤 Embeddings")
    st.sidebar.info(f"**Model**: {config.EMBEDDING_MODEL}")
    
    return selected_provider, selected_model
    
    # Usage tips
    st.sidebar.subheader("💡 Usage Tips")
    st.sidebar.markdown("""
    **Ask about:**
    - Life's purpose and dharma
    - Dealing with challenges
    - Karma and righteous action
    - Spiritual practices
    - Inner peace and wisdom
    
    **Example questions:**
    - "What is my dharma?"
    - "How to find inner peace?"
    - "What does Krishna say about suffering?"
    """)


def chat_interface(selected_provider: str, selected_model: str):
    """Main chat interface with model selection support"""
    config = Config()
    
    # Check prerequisites
    if not Path(config.CHUNKS_PATH).exists():
        st.error("📖 Please process the PDF first using the sidebar button.")
        return
    
    # Check if model selection has changed
    current_model_key = f"{selected_provider}:{selected_model}"
    if 'current_model' not in st.session_state or st.session_state.current_model != current_model_key:
        # Clear existing chatbot when model changes
        if 'chatbot' in st.session_state:
            del st.session_state.chatbot
        st.session_state.current_model = current_model_key
    
    # Initialize chatbot with selected model
    if 'chatbot' not in st.session_state:
        # Validate that we have a selected provider and model
        if not selected_provider or not selected_model:
            st.error("❌ Please select both a model provider and a model from the sidebar.")
            return
        
        with st.spinner(f"🔮 Initializing book analyzer with {selected_provider.title()} {selected_model}..."):
            st.session_state.chatbot = initialize_chatbot(selected_provider, selected_model)
    
    if st.session_state.chatbot is None:
        st.error("❌ Failed to initialize chatbot. Please check your setup and API keys.")
        return
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "📚 Hello! I'm here to help you analyze and understand your books. Ask me about themes, characters, plot, writing style, or any specific content you'd like to explore."
            }
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about themes, characters, plot, writing style, or book content..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing book content..."):
                try:
                    response = st.session_state.chatbot.get_book_analysis(prompt)
                    st.write(response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def process_uploaded_pdf(uploaded_file):
    """Process a single uploaded PDF file"""
    config = Config()
    
    with st.spinner(f"📖 Processing {uploaded_file.name}..."):
        try:
            # Save uploaded file temporarily
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            temp_file_path = temp_dir / uploaded_file.name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the uploaded PDF
            processor = PDFProcessor(config)
            
            # Extract text from the uploaded PDF
            text = processor.extract_text_from_pdf(str(temp_file_path))
            if not text:
                st.error("❌ Failed to extract text from PDF")
                return
            
            # Create chunks from the text
            chunks = processor.create_chapters_and_verses(text, source_file=uploaded_file.name)
            
            if chunks:
                # Save chunks
                processor.save_chunks(chunks)
                
                # Create embeddings
                vector_store = VectorStoreAgent(config)
                vector_store.create_embeddings(chunks)
                
                st.success(f"✅ {uploaded_file.name} processed successfully!")
                st.sidebar.info(f"📊 {len(chunks)} text chunks created")
                
                # Clear chatbot to reinitialize with new data
                if 'chatbot' in st.session_state:
                    del st.session_state.chatbot
                
                st.rerun()
            else:
                st.error("❌ Failed to create chunks from PDF")
            
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
                
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            # Clean up on error
            temp_file_path = Path("temp_uploads") / uploaded_file.name
            if temp_file_path.exists():
                temp_file_path.unlink()


def reset_pdf_data():
    """Reset all PDF data and clear session state"""
    config = Config()
    
    with st.spinner("🗑️ Resetting data..."):
        try:
            # Clear processed chunks file
            chunks_path = Path(config.CHUNKS_PATH)
            if chunks_path.exists():
                chunks_path.unlink()
            
            # Clear embeddings
            embeddings_path = Path(config.EMBEDDINGS_PATH)
            if embeddings_path.exists():
                embeddings_path.unlink()
            
            # Clear metadata
            metadata_path = Path("data/processed/processing_metadata.json")
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Clear session state
            if 'uploaded_pdf' in st.session_state:
                del st.session_state.uploaded_pdf
            if 'chatbot' in st.session_state:
                del st.session_state.chatbot
            if 'messages' in st.session_state:
                del st.session_state.messages
            
            # Clear temporary uploads directory
            temp_dir = Path("temp_uploads")
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    file.unlink()
                temp_dir.rmdir()
            
            st.success("✅ All data reset successfully!")
            st.sidebar.success("Ready for new PDF upload")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error resetting data: {str(e)}")


def process_pdf_data():
    """Process books and create embeddings (legacy function for data folder PDFs)"""
    config = Config()
    
    with st.spinner("📖 Processing books..."):
        try:
            # Process books
            processor = PDFProcessor(config)
            chunks = processor.process_all_pdfs()
            
            if chunks:
                # Create embeddings
                vector_store = VectorStoreAgent(config)
                vector_store.create_embeddings(chunks)
                st.success("✅ Books processed successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to process books. Please check the file path.")
                
        except Exception as e:
            st.error(f"❌ Error processing books: {str(e)}")


def initialize_chatbot(provider_name: str, model_name: str):
    """Initialize the book analysis system with selected model"""
    try:
        config = Config()
        
        # Load vector store
        vector_store = VectorStoreAgent(config)
        if not vector_store.load_embeddings():
            st.error("Failed to load embeddings. Please process your books first.")
            return None
        
        # Create book analysis agent with selected model
        chatbot = BookAnalysisAgent(config, vector_store, provider_name, model_name)
        
        # Verify the model is properly initialized
        model_info = chatbot.get_current_model_info()
        if model_info['status'] != 'initialized':
            st.error(f"Failed to initialize {provider_name} model: {model_name}")
            return None
        
        return chatbot
        
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None


if __name__ == "__main__":
    main()
