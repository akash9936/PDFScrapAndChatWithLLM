"""
Enhanced PDF Processing and Vector Store Agent for Spiritual Knowledge Base
===========================================================================

Improved modules with better error handling, logging, and multi-format support.
"""

# ================================
# pdf_processor.py - Enhanced PDF Processing
# ================================

import fitz  # PyMuPDF
import json
import re
import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class PDFProcessor:
    """Enhanced agent for extracting and processing PDF content"""
    
    def __init__(self, config):
        self.config = config
        self.stats = ProcessingStats()
        
        # Create necessary directories
        self.data_dir = Path("data")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced text cleaning patterns
        self.cleaning_patterns = {
            'page_numbers': r'\n\s*\d+\s*\n',
            'page_headers': r'\n\s*Page \d+.*?\n',
            'multiple_newlines': r'\n{3,}',
            'multiple_spaces': r'\s+',
            'header_footer': r'(Header|Footer|Page \d+)',
            'table_of_contents': r'Table of Contents.*?(?=Chapter|\n\n)',
            'index_pattern': r'Index.*?(?=\n\n|\Z)',
        }
        
        # Chapter detection patterns (more comprehensive)
        self.chapter_patterns = [
            r'Chapter \d+',           # English chapters
            r'अध्याय \d+',             # Hindi chapters  
            r'Chapter [IVX]+',        # Roman numerals
            r'Adhyaya \d+',           # Transliterated
            r'श्री.*?अध्याय',          # Formal Sanskrit
            r'CHAPTER \d+',           # Uppercase
        ]
        
        # Verse patterns for better splitting
        self.verse_patterns = [
            r'\n\s*\d+\.\d+\s*',      # 2.47 format
            r'\n\s*(\d+)\s*\n',       # Simple numbers
            r'श्लोक \d+',              # Sanskrit verse markers
            r'Verse \d+',             # English verse markers
            r'Sloka \d+',             # Transliterated
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract clean text from PDF with error handling"""
        try:
            logger.info(f"📖 Extracting text from {pdf_path}...")
            
            if not Path(pdf_path).exists():
                error_msg = f"PDF file not found: {pdf_path}"
                logger.error(error_msg)
                self.stats.errors.append(error_msg)
                return None
            
            doc = fitz.open(pdf_path)
            full_text = ""
            page_count = len(doc)
            
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    # Clean the text
                    text = self._clean_text(text)
                    full_text += text + "\n"
                    
                    # Progress indicator for large files
                    if page_count > 50 and page_num % 10 == 0:
                        logger.info(f"Processed {page_num}/{page_count} pages...")
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
            
            doc.close()
            
            char_count = len(full_text)
            self.stats.total_characters += char_count
            
            logger.info(f"✅ Extracted {char_count:,} characters from {page_count} pages")
            return full_text
            
        except Exception as e:
            error_msg = f"Failed to extract text from {pdf_path}: {e}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return None
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with multiple patterns"""
        if not text:
            return ""
        
        # Apply all cleaning patterns
        for pattern_name, pattern in self.cleaning_patterns.items():
            if pattern_name == 'multiple_spaces':
                text = re.sub(pattern, ' ', text)
            elif pattern_name == 'multiple_newlines':
                text = re.sub(pattern, '\n\n', text)
            else:
                text = re.sub(pattern, '\n', text)
        
        # Additional cleaning
        text = text.strip()
        
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only keep non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def detect_book_structure(self, text: str) -> Dict[str, any]:
        """Analyze the book structure for better processing"""
        structure_info = {
            'has_chapters': False,
            'chapter_count': 0,
            'has_verses': False,
            'verse_count': 0,
            'language': 'unknown',
            'book_type': 'unknown'
        }
        
        # Detect chapters
        for pattern in self.chapter_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                structure_info['has_chapters'] = True
                structure_info['chapter_count'] = len(matches)
                break
        
        # Detect verses
        for pattern in self.verse_patterns:
            matches = re.findall(pattern, text)
            if matches:
                structure_info['has_verses'] = True
                structure_info['verse_count'] = len(matches)
                break
        
        # Detect language
        if re.search(r'[\u0900-\u097F]+', text):  # Devanagari script
            structure_info['language'] = 'sanskrit_hindi'
        elif re.search(r'[a-zA-Z]', text):
            structure_info['language'] = 'english'
        
        # Detect book type
        if any(word in text.lower() for word in ['gita', 'भगवद्गीता', 'bhagavad']):
            structure_info['book_type'] = 'bhagavad_gita'
        elif any(word in text.lower() for word in ['upanishad', 'उपनिषद्']):
            structure_info['book_type'] = 'upanishad'
        elif any(word in text.lower() for word in ['purana', 'पुराण']):
            structure_info['book_type'] = 'purana'
        
        logger.info(f"📊 Book structure: {structure_info}")
        return structure_info
    
    def create_chapters_and_verses(self, text: str, source_file: str = "unknown") -> List[Dict]:
        """Intelligently split text into chapters and verses with enhanced structure detection"""
        logger.info(f"📝 Creating structured chunks for {source_file}...")
        
        # Analyze book structure first
        structure = self.detect_book_structure(text)
        
        chunks = []
        chunk_id = 0
        
        # Choose processing strategy based on structure
        if structure['has_chapters']:
            chunks = self._process_with_chapters(text, source_file, structure)
        else:
            chunks = self._process_without_chapters(text, source_file, structure)
        
        # Update chunk IDs
        for i, chunk in enumerate(chunks):
            chunk['id'] = i
            chunk['processing_date'] = datetime.now().isoformat()
        
        logger.info(f"✅ Created {len(chunks)} structured chunks from {source_file}")
        return chunks
    
    def _process_with_chapters(self, text: str, source_file: str, structure: Dict) -> List[Dict]:
        """Process text that has clear chapter divisions"""
        chunks = []
        
        # Find the best chapter pattern
        best_pattern = None
        for pattern in self.chapter_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                best_pattern = pattern
                break
        
        if not best_pattern:
            return self._process_without_chapters(text, source_file, structure)
        
        # Split by chapters
        chapter_splits = re.split(f'({best_pattern})', text, flags=re.IGNORECASE)
        
        current_chapter = "Introduction"
        current_chapter_number = 0
        
        for i, section in enumerate(chapter_splits):
            if re.match(best_pattern, section, re.IGNORECASE):
                current_chapter = section.strip()
                current_chapter_number += 1
                continue
            
            if len(section.strip()) < 50:  # Skip very short sections
                continue
            
            # Process chapter content into verses/paragraphs
            chapter_chunks = self._split_chapter_content(
                section, current_chapter, current_chapter_number, source_file, structure
            )
            chunks.extend(chapter_chunks)
        
        return chunks
    
    def _process_without_chapters(self, text: str, source_file: str, structure: Dict) -> List[Dict]:
        """Process text without clear chapter divisions"""
        chunks = []
        
        # Split by paragraphs or verses
        if structure['has_verses']:
            sections = self._split_by_verses(text)
        else:
            sections = self._split_by_paragraphs(text)
        
        for i, section in enumerate(sections):
            if len(section.strip()) < 30:
                continue
            
            chunk = self._create_chunk(
                content=section.strip(),
                chapter="General",
                chapter_number=1,
                verse_number=i + 1,
                source_file=source_file,
                structure=structure
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_chapter_content(self, content: str, chapter: str, chapter_num: int, 
                              source_file: str, structure: Dict) -> List[Dict]:
        """Split chapter content into meaningful chunks"""
        chunks = []
        
        if structure['has_verses']:
            verses = self._split_by_verses(content)
        else:
            verses = self._split_by_paragraphs(content)
        
        for i, verse in enumerate(verses):
            if len(verse.strip()) < 30:
                continue
            
            chunk = self._create_chunk(
                content=verse.strip(),
                chapter=chapter,
                chapter_number=chapter_num,
                verse_number=i + 1,
                source_file=source_file,
                structure=structure
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_verses(self, text: str) -> List[str]:
        """Split text by verse patterns"""
        verses = [text]
        
        for pattern in self.verse_patterns:
            new_verses = []
            for verse in verses:
                split_verses = re.split(pattern, verse)
                new_verses.extend([v for v in split_verses if v and v.strip()])
            verses = new_verses
        
        return verses
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph boundaries"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _create_chunk(self, content: str, chapter: str, chapter_number: int, 
                     verse_number: int, source_file: str, structure: Dict) -> Dict:
        """Create a standardized chunk dictionary"""
        return {
            "id": 0,  # Will be updated later
            "source_file": source_file,
            "chapter": chapter,
            "chapter_number": chapter_number,
            "verse_number": verse_number,
            "content": content,
            "type": self._classify_content_type(content, structure),
            "word_count": len(content.split()),
            "character_count": len(content),
            "language": structure.get('language', 'unknown'),
            "book_type": structure.get('book_type', 'unknown'),
            "processing_date": datetime.now().isoformat()
        }
    
    def _classify_content_type(self, content: str, structure: Dict) -> str:
        """Classify the type of content (verse, commentary, etc.)"""
        # Sanskrit/Hindi verse detection
        if re.search(r'[\u0900-\u097F]+', content):
            return 'sanskrit_verse'
        
        # Short content is likely a verse
        if len(content.split()) < 50:
            return 'verse'
        
        # Content with explanatory words
        if any(word in content.lower() for word in ['meaning', 'explanation', 'commentary', 'thus']):
            return 'commentary'
        
        # Default
        return 'text'
    
    def save_chunks(self, chunks: List[Dict]) -> None:
        """Save processed chunks with metadata"""
        try:
            # Save main chunks file
            with open(self.config.CHUNKS_PATH, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            # Save processing metadata
            metadata = {
                "processing_date": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "total_files": self.stats.processed_files,
                "total_characters": self.stats.total_characters,
                "chunk_types": {},
                "languages": {},
                "book_types": {},
                "errors": self.stats.errors
            }
            
            # Analyze chunk statistics
            for chunk in chunks:
                # Count by type
                chunk_type = chunk.get('type', 'unknown')
                metadata['chunk_types'][chunk_type] = metadata['chunk_types'].get(chunk_type, 0) + 1
                
                # Count by language
                language = chunk.get('language', 'unknown')
                metadata['languages'][language] = metadata['languages'].get(language, 0) + 1
                
                # Count by book type
                book_type = chunk.get('book_type', 'unknown')
                metadata['book_types'][book_type] = metadata['book_types'].get(book_type, 0) + 1
            
            # Save metadata
            metadata_path = self.processed_dir / "processing_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved {len(chunks)} chunks and metadata")
            logger.info(f"📊 Chunk types: {metadata['chunk_types']}")
            
        except Exception as e:
            error_msg = f"Failed to save chunks: {e}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
    
    def process_all_pdfs(self) -> List[Dict]:
        """Main processing pipeline - processes all PDFs in data directory"""
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return []
        
        # Find all PDF files
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"❌ No PDF files found in {self.data_dir}")
            return []
        
        self.stats.total_files = len(pdf_files)
        logger.info(f"📚 Found {len(pdf_files)} PDF files to process:")
        for pdf_file in pdf_files:
            logger.info(f"  - {pdf_file.name}")
        
        all_chunks = []
        
        # Process each PDF file
        for pdf_file in pdf_files:
            logger.info(f"\n🔄 Processing {pdf_file.name}...")
            
            try:
                # Extract text
                text = self.extract_text_from_pdf(str(pdf_file))
                if not text:
                    continue
                
                # Create structured chunks
                chunks = self.create_chapters_and_verses(text, source_file=pdf_file.name)
                all_chunks.extend(chunks)
                self.stats.processed_files += 1
                
            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {e}"
                logger.error(error_msg)
                self.stats.errors.append(error_msg)
                continue
        
        # Update chunk IDs to be sequential across all files
        for i, chunk in enumerate(all_chunks):
            chunk['id'] = i
        
        # Save all chunks
        if all_chunks:
            self.save_chunks(all_chunks)
            self.stats.total_chunks = len(all_chunks)
        
        # Print final statistics
        self._print_processing_summary()
        
        return all_chunks
    
    def _print_processing_summary(self):
        """Print processing summary"""
        logger.info(f"\n📊 PROCESSING SUMMARY")
        logger.info(f"=" * 50)
        logger.info(f"Total files found: {self.stats.total_files}")
        logger.info(f"Successfully processed: {self.stats.processed_files}")
        logger.info(f"Total chunks created: {self.stats.total_chunks}")
        logger.info(f"Total characters processed: {self.stats.total_characters:,}")
        
        if self.stats.errors:
            logger.warning(f"Errors encountered: {len(self.stats.errors)}")
            for error in self.stats.errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(self.stats.errors) > 5:
                logger.warning(f"  ... and {len(self.stats.errors) - 5} more errors")
