"""
Book Scraping Tool using LangChain, Groq, and Agentic Flow
==========================================================

This project creates a sophisticated tool that can scrape and analyze books
using advanced RAG (Retrieval Augmented Generation) with agentic workflows.

Project Structure:
├── requirements.txt
├── config.py
├── pdf_processor.py
├── vector_store.py
├── agents.py
├── chatbot.py
└── data/
    ├── books/
    └── processed/
        ├── chunks.json
        └── embeddings.faiss
"""

# ================================
# requirements.txt
# ================================
"""
langchain==0.1.0
langchain-community==0.0.13
langchain-groq==0.0.1
faiss-cpu==1.7.4
sentence-transformers==2.2.2
streamlit==1.29.0
pymupdf==1.23.14
python-dotenv==1.0.0
tiktoken==0.5.2
"""

# ================================
# config.py - Configuration Management
# ================================
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API (Free tier: 30 requests/minute)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Get free API key from console.groq.com
    
    # Model configurations
    GROQ_MODEL = "mixtral-8x7b-32768"  # Free and powerful
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free sentence transformer
    
    # File paths
    BOOKS_PATH = "data/books/"  # Directory containing PDF books
    CHUNKS_PATH = "data/processed/chunks.json"  # JSON file for processed chunks
    EMBEDDINGS_PATH = "data/processed/embeddings.faiss"  # FAISS index for embeddings
    
    # Processing parameters
    CHUNK_SIZE = 1000  # Size of each chunk in characters
    CHUNK_OVERLAP = 200  # Overlap between chunks in characters
    MAX_TOKENS = 4000  # Maximum tokens for embedding model

# ================================
# pdf_processor.py - PDF Processing Agent
# ================================
import fitz  # PyMuPDF
import json
import re
from typing import List, Dict
from pathlib import Path

class BookProcessor:
    """Agent responsible for extracting and processing book content from PDFs"""
    
    def __init__(self, config: Config):
        self.config = config
        Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract clean text from PDF"""
        print(f"📖 Extracting text from {pdf_path}...")
        
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Clean the text
            text = self._clean_text(text)
            full_text += text + "\n"
        
        doc.close()
        print(f"✅ Extracted {len(full_text)} characters from PDF")
        return full_text
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove page numbers, headers, footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Page numbers
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text)  # Page headers
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        return text.strip()
    
    def create_chapters_and_sections(self, text: str) -> List[Dict]:
        """Intelligently split text into chapters and sections"""
        print("📝 Creating structured chapters and sections...")
        
        chunks = []
        
        # Split by chapters (adjust regex based on book format)
        chapter_pattern = r'(Chapter \d+|CHAPTER \d+|Part \d+|Section \d+)'
        chapters = re.split(chapter_pattern, text)
        
        current_chapter = "Introduction"
        chunk_id = 0
        
        for i, section in enumerate(chapters):
            if re.match(chapter_pattern, section):
                current_chapter = section
                continue
            
            if len(section.strip()) < 50:  # Skip very short sections
                continue
            
            # Split chapter into paragraphs/sections
            sections = self._split_into_sections(section)
            
            for section in sections:
                if len(section.strip()) > 100:  # Only meaningful content
                    chunks.append({
                        "id": chunk_id,
                        "chapter": current_chapter,
                        "content": verse.strip(),
                        "type": "verse" if self._is_verse(verse) else "commentary"
                    })
                    chunk_id += 1
        
        print(f"✅ Created {len(chunks)} structured chunks")
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split chapter text into sections"""
        # Look for verse patterns (adjust based on your PDF format)
        verse_patterns = [
            r'\d+\.\d+',  # 2.47, 3.21 etc.
            r'Verse \d+',
            r'श्लोक \d+'
        ]
        
        for pattern in verse_patterns:
            if re.search(pattern, text):
                return re.split(pattern, text)
        
        # Fallback: split by sentences/paragraphs
        return [chunk for chunk in text.split('\n\n') if len(chunk.strip()) > 50]
    
    def _is_section(self, text: str) -> bool:
        """Determine if text is a section or commentary"""
        # Verses usually contain Sanskrit or are shorter
        sanskrit_pattern = r'[\u0900-\u097F]+'  # Devanagari script
        return bool(re.search(sanskrit_pattern, text)) or len(text) < 200
    
    def save_chunks(self, chunks: List[Dict]) -> None:
        """Save processed chunks to JSON"""
        with open(self.config.CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(chunks)} chunks to {self.config.CHUNKS_PATH}")
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Main processing pipeline"""
        print("🚀 Starting PDF processing pipeline...")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Create structured chunks
        chunks = self.create_chapters_and_sections(text)
        
        # Save chunks
        self.save_chunks(chunks)
        
        return chunks

# ================================
# vector_store.py - Vector Database Agent
# ================================
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class VectorStoreAgent:
    """Agent responsible for embeddings and similarity search"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, chunks: List[Dict]) -> None:
        """Create and store embeddings"""
        print("🧮 Creating embeddings...")
        
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Save index and chunks
        faiss.write_index(self.index, self.config.EMBEDDINGS_PATH)
        self.chunks = chunks
        
        print(f"✅ Created embeddings for {len(chunks)} chunks")
    
    def load_embeddings(self) -> bool:
        """Load existing embeddings"""
        try:
            self.index = faiss.read_index(self.config.EMBEDDINGS_PATH)
            with open(self.config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"✅ Loaded {len(self.chunks)} chunks from storage")
            return True
        except:
            print("⚠️ No existing embeddings found")
            return False
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results

# ================================
# agents.py - Spiritual Guidance Agents
# ================================
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class BookAnalysisAgent:
    """Main agent for providing book analysis and insights"""
    
    def __init__(self, config: Config, vector_store: VectorStoreAgent):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=config.MAX_TOKENS
        )
    
    def get_book_analysis(self, question: str) -> str:
        """Provide book analysis and insights based on book content"""
        
        # Step 1: Retrieve relevant passages
        relevant_chunks = self.vector_store.search_similar(question, k=3)
        
        if not relevant_chunks:
            return "I apologize, but I couldn't find relevant guidance in the Bhagavad Gita for your question."
        
        # Step 2: Analyze question type
        question_type = self._analyze_question_type(question)
        
        # Step 3: Generate contextual response
        response = self._generate_response(question, relevant_chunks, question_type)
        
        return response
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of book-related question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['theme', 'meaning', 'why', 'purpose']):
            return 'thematic'
        elif any(word in question_lower for word in ['character', 'plot', 'story']):
            return 'narrative'
        elif any(word in question_lower for word in ['author', 'style', 'writing']):
            return 'literary_analysis'
        elif any(word in question_lower for word in ['summary', 'overview', 'about']):
            return 'summary'
        else:
            return 'general'
    
    def _generate_response(self, question: str, chunks: List[Dict], question_type: str) -> str:
        """Generate contextual book analysis response"""
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"From {chunk['chapter']}:\n{chunk['content']}"
            for chunk in chunks[:3]
        ])
        
        # Create specialized prompt based on question type
        system_prompt = self._get_system_prompt(question_type)
        
        prompt = f"""
{system_prompt}

Question: {question}

Relevant teachings from Bhagavad Gita:
{context}

Please provide a thoughtful, compassionate response that:
1. Directly addresses the question
2. References specific teachings from the provided context
3. Offers practical spiritual guidance
4. Maintains the wisdom and tone of the Gita
5. Is accessible to modern readers

Response:"""

        try:
            messages = [
                SystemMessage(content="You are a wise spiritual teacher well-versed in the Bhagavad Gita."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            return response.content.strip()
            
        except Exception as e:
            return f"I apologize, but I'm having difficulty accessing the spiritual guidance right now. Please try again. ({str(e)})"
    
    def _get_system_prompt(self, question_type: str) -> str:
        """Get specialized system prompt based on question type"""
        
        prompts = {
            'thematic': """You are a literary analyst helping someone understand the themes, meanings, and deeper messages in the book. Focus on analyzing the central ideas and their significance.""",
            
            'narrative': """You are a literature expert explaining plot elements, character development, and story structure. Help the person understand the narrative components and their relationships.""",
            
            'literary_analysis': """You are a literary critic analyzing writing style, author techniques, and literary devices. Guide the person in understanding the craft and artistry of the work.""",
            
            'summary': """You are a knowledgeable reader providing clear summaries and overviews of book content. Help the person understand the key points and main ideas.""",
            
            'general': """You are a knowledgeable book analyst providing insights and analysis based on the content of the book."""
        }
        
        return prompts.get(question_type, prompts['general'])

# ================================
# chatbot.py - Main Streamlit Application
# ================================
import streamlit as st
import time

def main():
    st.set_page_config(
        page_title="📚 Book Scraping & Analysis Tool",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #2E86AB, #A23B72, #F18F01);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
        background-color: #f8f9fa;
    }
    .spiritual-quote {
        font-style: italic;
        text-align: center;
        color: #666;
        border-left: 3px solid #138808;
        padding-left: 1rem;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📚 Book Scraping & Analysis Tool</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="spiritual-quote">"A reader lives a thousand lives before he dies. The man who never reads lives only one."<br>- George R.R. Martin</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if data is processed
        config = Config()
        
        if not os.path.exists(config.CHUNKS_PATH):
            st.warning("📖 No books processed yet!")
            if st.button("🔄 Process Books"):
                process_book_data()
        else:
            st.success("✅ Books processed successfully!")
            
            # Show data stats
            with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            st.info(f"📊 **{len(chunks)} chunks** ready for search")
        
        st.header("🎯 Features")
        st.markdown("""
        - **Intelligent Search**: Find relevant verses instantly
        - **Contextual Answers**: AI-powered spiritual guidance
        - **Chapter Navigation**: Organized by Gita chapters
        - **Free & Open Source**: No cost to run
        """)
        
        st.header("🔑 Setup")
        groq_key = st.text_input("Groq API Key", type="password", 
                                help="Get free API key from console.groq.com")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
    
    # Main chat interface
    if not os.path.exists(config.CHUNKS_PATH):
        st.error("Please process your books first using the sidebar button.")
        return
    
    if not config.GROQ_API_KEY and not os.getenv("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar.")
        return
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "📚 Hello! I'm here to help you analyze and understand your books. Ask me about themes, characters, plot, writing style, or any specific content you'd like to explore."}
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
            with st.spinner("Analyzing book content..."):
                response = st.session_state.chatbot.get_book_analysis(prompt)
                st.write(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

def process_book_data():
    """Process books and create embeddings"""
    config = Config()
    
    with st.spinner("📖 Processing books..."):
        # Process books
        processor = BookProcessor(config)
        # For now, we'll need to specify a book path - this should be made configurable
        book_path = "data/books/sample_book.pdf"  # This should be made configurable
        chunks = processor.process_pdf(book_path)
        
        if chunks:
            # Create embeddings
            vector_store = VectorStoreAgent(config)
            vector_store.create_embeddings(chunks)
            st.success("✅ Books processed successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to process books. Please check the file path.")

def initialize_chatbot():
    """Initialize the book analysis system"""
    config = Config()
    
    # Load vector store
    vector_store = VectorStoreAgent(config)
    if not vector_store.load_embeddings():
        st.error("Failed to load embeddings. Please process your books first.")
        return None
    
    # Create book analysis agent
    chatbot = BookAnalysisAgent(config, vector_store)
    return chatbot

if __name__ == "__main__":
    main()

# ================================
# Setup Instructions
# ================================
"""
🚀 SETUP INSTRUCTIONS:

1. **Install Dependencies:**
   pip install -r requirements.txt

2. **Get Free Groq API Key:**
   - Visit: https://console.groq.com
   - Sign up for free account
   - Create API key (30 requests/minute free)

3. **Prepare Data:**
   - Create 'data/books' folder in project root
   - Place your PDF books in the 'data/books/' directory

4. **Environment Setup:**
   - Create .env file with: GROQ_API_KEY=your_key_here
   - Or enter key in the Streamlit sidebar

5. **Run the Application:**
   streamlit run chatbot.py

6. **First Time Setup:**
   - Click "Process Books" in sidebar
   - Wait for processing to complete
   - Start asking questions about your books!

🎯 FEATURES:
- Intelligent semantic search through book content
- Context-aware book analysis and insights
- Chapter-wise organization
- Free to run (uses free tiers)
- Agentic workflow with specialized responses
- Modern RAG architecture with LangChain

📚 Example Questions:
- "What are the main themes in this book?"
- "Can you summarize chapter 3?"
- "How does the author develop the main character?"
- "What is the writing style like?"
- "What are the key plot points?"
"""