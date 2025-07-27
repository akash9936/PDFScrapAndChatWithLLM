
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStoreAgent:
    """Enhanced agent for embeddings and similarity search"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.embeddings = None
        self.chunks = []
        self.index_metadata = {}
        
        # Initialize model with error handling
        try:
            logger.info(f"🤖 Loading embedding model: {config.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def create_embeddings(self, chunks: List[Dict], batch_size: int = 32) -> None:
        """Create embeddings with batching and progress tracking"""
        if not chunks:
            logger.warning("No chunks provided for embedding creation")
            return
        
        logger.info(f"🔮 Creating embeddings for {len(chunks)} chunks...")
        
        try:
            # Extract text content
            texts = [chunk['content'] for chunk in chunks]
            
            # Create embeddings in batches for memory efficiency
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches}...")
                
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Store embeddings and chunks
            self.embeddings = embeddings
            self.chunks = chunks
            
            # Create metadata
            self.index_metadata = {
                'creation_date': datetime.now().isoformat(),
                'model_name': self.config.EMBEDDING_MODEL,
                'total_chunks': len(chunks),
                'embedding_dimension': embeddings.shape[1],
                'normalization': 'l2',
                'chunk_statistics': self._calculate_chunk_statistics(chunks)
            }
            
            # Save to disk
            self._save_embeddings()
            
            logger.info(f"✅ Created embeddings: {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings: {e}")
            raise
    
    def _calculate_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """Calculate statistics about the chunks"""
        stats = {
            'total_chunks': len(chunks),
            'avg_word_count': 0,
            'avg_char_count': 0,
            'content_types': {},
            'languages': {},
            'book_types': {},
            'sources': {}
        }
        
        if not chunks:
            return stats
        
        total_words = sum(chunk.get('word_count', 0) for chunk in chunks)
        total_chars = sum(chunk.get('character_count', 0) for chunk in chunks)
        
        stats['avg_word_count'] = total_words / len(chunks)
        stats['avg_char_count'] = total_chars / len(chunks)
        
        # Count different categories
        for chunk in chunks:
            # Content types
            content_type = chunk.get('type', 'unknown')
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            # Languages
            language = chunk.get('language', 'unknown')
            stats['languages'][language] = stats['languages'].get(language, 0) + 1
            
            # Book types
            book_type = chunk.get('book_type', 'unknown')
            stats['book_types'][book_type] = stats['book_types'].get(book_type, 0) + 1
            
            # Source files
            source = chunk.get('source_file', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        return stats
    
    def _save_embeddings(self) -> None:
        """Save embeddings and metadata to disk"""
        try:
            embeddings_data = {
                'embeddings': self.embeddings,
                'chunks': self.chunks,
                'metadata': self.index_metadata
            }
            
            embeddings_file = self.config.EMBEDDINGS_PATH.replace('.faiss', '.pkl')
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Also save metadata separately for easy inspection
            metadata_file = Path(embeddings_file).parent / "embeddings_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.index_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved embeddings to {embeddings_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {e}")
            raise
    
    def load_embeddings(self) -> bool:
        """Load existing embeddings with validation"""
        try:
            embeddings_file = self.config.EMBEDDINGS_PATH.replace('.faiss', '.pkl')
            
            if not Path(embeddings_file).exists():
                logger.warning("❌ Embeddings file not found")
                return False
            
            logger.info(f"📂 Loading embeddings from {embeddings_file}")
            
            with open(embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            self.embeddings = embeddings_data['embeddings']
            self.chunks = embeddings_data['chunks']
            self.index_metadata = embeddings_data.get('metadata', {})
            
            # Validate loaded data
            if not self._validate_embeddings():
                logger.error("❌ Loaded embeddings failed validation")
                return False
            
            logger.info(f"✅ Loaded {len(self.chunks)} embeddings successfully")
            logger.info(f"📊 Model: {self.index_metadata.get('model_name', 'unknown')}")
            logger.info(f"📊 Dimensions: {self.embeddings.shape[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            return False
    
    def _validate_embeddings(self) -> bool:
        """Validate loaded embeddings"""
        try:
            if self.embeddings is None or self.chunks is None:
                return False
            
            if len(self.embeddings) != len(self.chunks):
                logger.error("Mismatch between embeddings and chunks count")
                return False
            
            if self.embeddings.shape[0] == 0:
                logger.error("Empty embeddings array")
                return False
            
            # Check if embeddings are normalized
            norms = np.linalg.norm(self.embeddings, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-5):
                logger.warning("Embeddings are not normalized, normalizing now...")
                self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5, 
                      filter_by: Optional[Dict[str, str]] = None,
                      min_similarity: float = 0.0) -> List[Dict]:
        """Enhanced similarity search with filtering"""
        if self.embeddings is None or not self.chunks:
            logger.warning("❌ No embeddings loaded for search")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(query_embedding, self.embeddings.T)[0]
            
            # Apply minimum similarity threshold
            valid_indices = np.where(similarities >= min_similarity)[0]
            
            if len(valid_indices) == 0:
                logger.info(f"No results found above similarity threshold {min_similarity}")
                return []
            
            # Sort by similarity (descending)
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
            
            # Apply filters if provided
            if filter_by:
                sorted_indices = self._apply_filters(sorted_indices, filter_by)
            
            # Get top k results
            top_indices = sorted_indices[:k]
            
            # Build results
            results = []
            for i, idx in enumerate(top_indices):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(similarities[idx])
                    chunk['rank'] = i + 1
                    results.append(chunk)
            
            logger.info(f"🔍 Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def _apply_filters(self, indices: np.ndarray, filters: Dict[str, str]) -> np.ndarray:
        """Apply filters to search results"""
        filtered_indices = []
        
        for idx in indices:
            chunk = self.chunks[idx]
            match = True
            
            for filter_key, filter_value in filters.items():
                if chunk.get(filter_key) != filter_value:
                    match = False
                    break
            
            if match:
                filtered_indices.append(idx)
        
        return np.array(filtered_indices)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the vector store"""
        if not self.chunks:
            return {"error": "No data loaded"}
        
        stats = {
            "total_chunks": len(self.chunks),
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model_info": self.index_metadata,
            "chunk_statistics": self._calculate_chunk_statistics(self.chunks)
        }
        
        return stats
    
    def search_by_chapter(self, chapter: str, k: int = 10) -> List[Dict]:
        """Search for chunks from a specific chapter"""
        return self.search_similar("", k=k, filter_by={"chapter": chapter})
    
    def search_by_type(self, content_type: str, k: int = 10) -> List[Dict]:
        """Search for chunks of a specific type"""
        return self.search_similar("", k=k, filter_by={"type": content_type})
    
    def export_chunks(self, output_file: str, format: str = 'json') -> bool:
        """Export chunks to different formats"""
        try:
            if format.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(self.chunks)
                df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"✅ Exported {len(self.chunks)} chunks to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False


# ================================
# Enhanced Configuration Class
# ================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Enhanced configuration with validation and defaults"""
    
    def __init__(self):
        # API Keys
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        
        # Model configurations
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Alternative embedding models (uncomment to use)
        # self.EMBEDDING_MODEL = "all-mpnet-base-v2"  # Higher quality but slower
        # self.EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # Good for Q&A
        
        # File paths
        self.DATA_DIR = Path("data")
        self.PROCESSED_DIR = Path("data/processed")
        self.CHUNKS_PATH = str(self.PROCESSED_DIR / "chunks.json")
        self.EMBEDDINGS_PATH = str(self.PROCESSED_DIR / "embeddings.pkl")
        
        # Processing parameters
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
        self.MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "30"))
        
        # Search parameters
        self.DEFAULT_SEARCH_K = int(os.getenv("DEFAULT_SEARCH_K", "5"))
        self.MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.1"))
        
        # Batch processing
        self.EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        
        # Create directories
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required")
        
        if not self.DATA_DIR.exists():
            errors.append(f"Data directory does not exist: {self.DATA_DIR}")
        
        if errors:
            for error in errors:
                logger.error(f"Config error: {error}")
            return False
        
        return True