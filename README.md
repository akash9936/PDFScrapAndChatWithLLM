
source .venv/bin/activate && streamlit run app.py
pkill -f streamlit

# 🕉️ Bhagavad Gita Spiritual Chatbot

A sophisticated AI-powered spiritual guidance chatbot that provides wisdom from the Bhagavad Gita using advanced RAG (Retrieval Augmented Generation) with agentic workflows.

## ✨ Features

- **Intelligent Semantic Search**: Advanced search through Gita content using vector embeddings
- **Context-Aware Spiritual Guidance**: Specialized responses based on question type analysis
- **Chapter-wise Organization**: Structured processing of Gita content into chapters and verses
- **Agentic Workflow**: Multiple specialized agents for different aspects
- **Modern RAG Architecture**: Built with LangChain and state-of-the-art NLP models
- **Free to Run**: Uses free tiers of Groq API and open-source models
- **Interactive Web Interface**: Streamlit-based chat interface

## 🏗️ Project Structure

```
SpritualGuru/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── project.txt                 # Detailed project documentation
├── chatbot.py                  # Original monolithic version (backup)
├── .env                        # Environment variables (create this)
├── src/                        # Source code package
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── agents/                 # AI agents
│   │   ├── __init__.py
│   │   └── spiritual_guidance.py
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py
│   │   ├── pdf_processor.py    # PDF processing agent
│   │   └── vector_store.py     # Vector database agent
│   └── ui/                     # User interface
│       ├── __init__.py
│       └── streamlit_app.py    # Streamlit web interface
└── data/                       # Data directory (create this)
    ├── shreemad_bhagavad_gita.pdf    # Source PDF
    └── processed/              # Processed data
        ├── chunks.json         # Text chunks
        └── embeddings.faiss    # Vector embeddings
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Free Groq API Key
- Visit: [console.groq.com](https://console.groq.com)
- Sign up for free account
- Create API key (30 requests/minute free tier)

### 3. Setup Environment
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Prepare Data
- Create `data` folder in project root
- Place your Bhagavad Gita PDF as `data/shreemad_bhagavad_gita.pdf`

### 5. Run the Application
```bash
streamlit run app.py
```

### 6. First Time Setup
- Click "Process PDF" in the sidebar
- Wait for processing to complete
- Start asking spiritual questions!

## 💬 Usage Examples

### Sample Questions
- "What is my dharma in life?"
- "How do I deal with suffering?"
- "What does the Gita say about karma?"
- "How can I find inner peace?"
- "What is the meaning of life according to Krishna?"
- "How should I handle difficult decisions?"
- "What is the path to self-realization?"

### Question Types Supported
- **Dharma & Purpose**: Life's duties and righteous path
- **Karma & Action**: Understanding action and consequences
- **Spiritual Practice**: Meditation, devotion, and self-discipline
- **Life Challenges**: Dealing with suffering, doubt, and conflict
- **Philosophy**: Deep spiritual and philosophical concepts

## 🛠️ Technical Architecture

### Core Components
1. **Config Management** (`src/config.py`): Centralized configuration
2. **PDF Processor Agent** (`src/utils/pdf_processor.py`): Extracts and structures content
3. **Vector Store Agent** (`src/utils/vector_store.py`): Handles embeddings and similarity search
4. **Spiritual Guidance Agent** (`src/agents/spiritual_guidance.py`): Main AI agent
5. **Streamlit Interface** (`src/ui/streamlit_app.py`): Web-based chat interface

### Technology Stack
- **LangChain**: Framework for building LLM applications
- **Groq API**: Fast inference with Mixtral-8x7b model
- **FAISS**: Vector database for similarity search
- **Sentence Transformers**: Text embeddings using all-MiniLM-L6-v2
- **Streamlit**: Web interface framework
- **PyMuPDF**: PDF text extraction
- **Python-dotenv**: Environment variable management

## 📋 Dependencies

```
langchain==0.1.0
langchain-community==0.0.13
langchain-groq==0.0.1
faiss-cpu==1.7.4
sentence-transformers==2.2.2
streamlit==1.29.0
pymupdf==1.23.14
python-dotenv==1.0.0
tiktoken==0.5.2
```

## ⚙️ Configuration

Key configuration parameters in `src/config.py`:
- **GROQ_MODEL**: "mixtral-8x7b-32768" (Free and powerful)
- **EMBEDDING_MODEL**: "all-MiniLM-L6-v2" (Free sentence transformer)
- **CHUNK_SIZE**: 1000 characters
- **CHUNK_OVERLAP**: 200 characters
- **MAX_TOKENS**: 4000 tokens per response

## 🔧 Development

### Project Structure Benefits
- **Modular Design**: Separated concerns for better maintainability
- **Scalable Architecture**: Easy to add new agents or features
- **Clean Imports**: Proper Python package structure
- **Error Handling**: Comprehensive error handling throughout
- **Documentation**: Well-documented code with docstrings

### Adding New Features
1. **New Agents**: Add to `src/agents/`
2. **Utility Functions**: Add to `src/utils/`
3. **UI Components**: Extend `src/ui/streamlit_app.py`
4. **Configuration**: Update `src/config.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is designed for educational and spiritual guidance purposes. Please respect the sacred nature of the Bhagavad Gita and use this tool mindfully.

## 🙏 Acknowledgments

- The eternal wisdom of the Bhagavad Gita
- The open-source community for the amazing tools
- Groq for providing free API access
- All seekers of spiritual wisdom

---

*"You have the right to perform your actions, but you are not entitled to the fruits of action."* - Bhagavad Gita 2.47
