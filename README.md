source .venv/bin/activate && streamlit run app.py
pkill -f streamlit

# 📚 PDF Scraping & Chat with LLM

A sophisticated AI-powered document analysis and chatbot system that extracts content from PDFs and provides intelligent responses using advanced RAG (Retrieval Augmented Generation) with multiple LLM providers.

## ✨ Features

- **Multi-Provider LLM Support**: Choose from Groq, Google Gemini, or Ollama
- **PDF Upload Interface**: Upload PDFs directly through the web interface
- **One-Click Processing**: Process uploaded PDFs with a single button click
- **Reset & Multi-PDF Support**: Reset data and process multiple PDFs sequentially
- **Intelligent PDF Processing**: Advanced text extraction and chunking
- **Semantic Search**: Vector-based document retrieval using embeddings
- **Context-Aware Responses**: Specialized AI agents for document analysis
- **Modern RAG Architecture**: Built with LangChain and state-of-the-art NLP models
- **Flexible Configuration**: Support for multiple model providers and configurations
- **Interactive Web Interface**: Streamlit-based chat interface
- **Free & Local Options**: Support for free APIs and local inference

## 🏗️ Project Structure

```
PDFScrapAndChatWithLLM/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── project.txt                 # Detailed project documentation
├── .env.template               # Environment variables template
├── .env                        # Environment variables (create from template)
├── chatbot.py                  # Original monolithic version (backup)
├── src/                        # Source code package
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── agents/                 # AI agents
│   │   ├── __init__.py
│   │   └── spiritual_guidance.py
│   ├── models/                 # LLM provider implementations
│   │   ├── __init__.py
│   │   ├── base_provider.py    # Base provider interface
│   │   ├── groq_provider.py    # Groq API integration
│   │   ├── gemini_provider.py  # Google Gemini integration
│   │   ├── ollama_provider.py  # Ollama local integration
│   │   └── model_factory.py    # Provider factory
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py
│   │   ├── pdf_processor.py    # PDF processing utilities
│   │   └── vector_store.py     # Vector database operations
│   └── ui/                     # User interface
│       ├── __init__.py
│       └── streamlit_app.py    # Streamlit web interface
├── data/                       # Data directory (create this)
│   ├── *.pdf                  # Source PDF files
│   └── processed/              # Processed data
│       ├── chunks.json         # Text chunks
│       └── embeddings.pkl      # Vector embeddings
└── docs/                       # Documentation
    └── *.md                    # Additional documentation
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd PDFScrapAndChatWithLLM
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the template and configure your API keys:
```bash
cp .env.template .env
# Edit .env with your preferred editor
```

### 3. Choose Your LLM Provider

#### Option A: Groq (Recommended - Fast & Free)
- Visit: [console.groq.com](https://console.groq.com)
- Sign up and get free API key (30 RPM, 6K TPM)
- Add to `.env`: `GROQ_API_KEY=your_key_here`

#### Option B: Google Gemini (Powerful Reasoning)
- Visit: [aistudio.google.com](https://aistudio.google.com)
- Get free API key (15 RPM Flash, 2 RPM Pro)
- Add to `.env`: `GEMINI_API_KEY=your_key_here`

#### Option C: Ollama (Local & Private)
- Install: [ollama.ai](https://ollama.ai)
- Pull a model: `ollama pull llama3.1:8b`
- Configure: `OLLAMA_BASE_URL=http://localhost:11434`

### 4. Prepare Your Data
```bash
mkdir -p data/processed
# Place your PDF files in the data/ directory
```

### 5. Run the Application
```bash
streamlit run app.py
```

### 6. First Time Setup
1. Select your preferred model provider in the sidebar
2. Upload or select PDF files to process
3. Click "Process PDF" and wait for completion
4. Start asking questions about your documents!

## 💬 Usage Examples

### Document Analysis Questions
- "What are the main topics covered in this document?"
- "Summarize the key findings from chapter 3"
- "What does the author say about [specific topic]?"
- "Find all references to [keyword]"
- "Compare the arguments presented in different sections"
- "What are the conclusions and recommendations?"

### Supported Document Types
- **Research Papers**: Academic publications, studies, reports
- **Books**: Textbooks, manuals, guides
- **Legal Documents**: Contracts, policies, regulations
- **Technical Documentation**: Specifications, user manuals
- **Business Documents**: Reports, proposals, presentations

## 🛠️ Technical Architecture

### Core Components
1. **Configuration Management** (`src/config.py`): Centralized settings and API management
2. **Model Providers** (`src/models/`): Abstracted LLM provider implementations
3. **PDF Processor** (`src/utils/pdf_processor.py`): Advanced text extraction and chunking
4. **Vector Store** (`src/utils/vector_store.py`): Semantic search and retrieval
5. **AI Agents** (`src/agents/`): Specialized response generation
6. **Web Interface** (`src/ui/streamlit_app.py`): Interactive chat application

### Technology Stack
- **LangChain**: LLM application framework
- **Multiple LLM Providers**: Groq, Google Gemini, Ollama
- **Vector Search**: FAISS for similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Web Framework**: Streamlit
- **PDF Processing**: PyMuPDF (fitz)
- **Configuration**: Python-dotenv
- **HTTP Requests**: Requests library

## 📋 Dependencies

```
langchain==0.1.0
langchain-community==0.0.13
langchain-groq==0.0.1
sentence-transformers==2.2.2
streamlit==1.29.0
pymupdf==1.23.14
python-dotenv==1.0.0
tiktoken==0.5.2
google-generativeai>=0.3.0
requests>=2.28.0
```

## ⚙️ Configuration Options

### Model Providers
- **Groq**: Fast inference, free tier available
  - Models: `llama-3.1-8b-instant`, `mixtral-8x7b-32768`
- **Google Gemini**: Advanced reasoning capabilities
  - Models: `gemini-1.5-flash`, `gemini-1.5-pro`
- **Ollama**: Local inference, complete privacy
  - Models: Any Ollama-compatible model

### Processing Parameters
- **CHUNK_SIZE**: 1000 characters (configurable)
- **CHUNK_OVERLAP**: 200 characters (configurable)
- **MAX_TOKENS**: 4000 tokens per response
- **EMBEDDING_MODEL**: all-MiniLM-L6-v2

### File Paths
- **PDF_PATH**: `data/` directory for source files
- **CHUNKS_PATH**: `data/processed/chunks.json`
- **EMBEDDINGS_PATH**: `data/processed/embeddings.pkl`

## 🔧 Development

### Architecture Benefits
- **Provider Abstraction**: Easy to add new LLM providers
- **Modular Design**: Separated concerns for maintainability
- **Scalable Structure**: Simple to extend functionality
- **Error Handling**: Comprehensive error management
- **Type Safety**: Proper Python typing throughout

### Adding New Features
1. **New LLM Provider**: Implement `BaseProvider` in `src/models/`
2. **Processing Logic**: Extend utilities in `src/utils/`
3. **UI Components**: Modify `src/ui/streamlit_app.py`
4. **Configuration**: Update `src/config.py` and `.env.template`

### Development Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --server.runOnSave true
```

## 🚀 Deployment

### Local Deployment
```bash
# Activate environment and run
source .venv/bin/activate && streamlit run app.py

# Stop the application
pkill -f streamlit
```

### Docker Deployment
```dockerfile
# Example Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is designed for educational and research purposes. Please ensure you have the right to process any documents you upload and respect copyright laws.

## 🙏 Acknowledgments

- **LangChain Community**: For the excellent framework
- **Groq**: For providing fast, free LLM inference
- **Google**: For Gemini API access
- **Ollama**: For local LLM capabilities
- **Streamlit**: For the intuitive web framework
- **Open Source Community**: For all the amazing tools and libraries

---

*"The best way to find out if you can trust somebody is to trust them."* - Build responsibly with AI.
