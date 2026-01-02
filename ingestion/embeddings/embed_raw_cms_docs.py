"""
Medical Coding System - Raw CMS Document Embedder
Embed raw CMS files with RICH METADATA for robust retrieval and rendering
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from openai import AzureOpenAI
import hashlib

# LangChain imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredXMLLoader,
    UnstructuredExcelLoader
)


# ============================================================================
# 1. IMPORT DEPENDENCIES
# ============================================================================
from config.settings import (
    ChromaDBConfig, CMSConfig, AzureOpenAIConfig,
    RAW_DATA_DIR
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 2. RAW DOCUMENT EMBEDDER (WITH RICH METADATA)
# ============================================================================

class RawCMSDocumentEmbedder:
    """
    Embed raw CMS documents with comprehensive metadata
    for robust retrieval, rendering, and source tracing
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize raw document embedder
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=AzureOpenAIConfig.API_KEY,
            api_version=AzureOpenAIConfig.API_VERSION,
            azure_endpoint=AzureOpenAIConfig.ENDPOINT
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(ChromaDBConfig.PERSIST_DIRECTORY),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Initialize LangChain text splitter with metadata preservation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True  # Track position in original document
        )
        
        logger.info("✓ Raw Document Embedder initialized (LangChain + Rich Metadata)")
        logger.info(f"  Azure OpenAI Endpoint: {AzureOpenAIConfig.ENDPOINT}")
        logger.info(f"  Embedding Model: {AzureOpenAIConfig.EMBED_DEPLOYMENT}")
        logger.info(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text batch"""
        response = self.openai_client.embeddings.create(
            input=texts,
            model=AzureOpenAIConfig.EMBED_DEPLOYMENT
        )
        return [item.embedding for item in response.data]
    
    def generate_chunk_id(self, content: str) -> str:
        """Generate unique hash for chunk deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    
    def load_document(self, file_path: Path) -> List:
        """Load document using appropriate LangChain loader"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif extension == '.xml':
                loader = UnstructuredXMLLoader(str(file_path), mode="elements")
            elif extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(str(file_path), mode="elements")
            else:
                logger.warning(f"  ⚠ Unsupported file type: {extension}")
                return []
            
            documents = loader.load()
            return documents
        
        except Exception as e:
            logger.error(f"  ✗ Failed to load {file_path.name}: {e}")
            return []
    
    def create_rich_metadata(
            self,
            document,
            file_path: Path,
            code_type: str,
            year: int,
            chunk_index: int,
            total_chunks: int
        ) -> Dict[str, Any]:
            """
            Create comprehensive metadata for robust retrieval
            ChromaDB doesn't accept None values, so we filter them out
            
            Args:
                document: LangChain Document object
                file_path: Original file path
                code_type: Code type
                year: Year
                chunk_index: Index of this chunk
                total_chunks: Total number of chunks in document
            
            Returns:
                Rich metadata dictionary (without None values)
            """
            metadata = {
                # Source identification
                'source_file': file_path.name,
                'source_path': str(file_path.relative_to(RAW_DATA_DIR)),
                'file_type': file_path.suffix[1:],
                'file_size_kb': round(file_path.stat().st_size / 1024, 2),
                
                # CMS classification
                'code_type': code_type,
                'year': str(year),
                'cms_category': code_type.upper(),
                
                # Chunk information
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'chunk_id': self.generate_chunk_id(document.page_content),
                'chunk_size': len(document.page_content),
                
                # Context preservation
                'has_previous_chunk': chunk_index > 0,
                'has_next_chunk': chunk_index < total_chunks - 1,
            }
            
            # Add optional fields only if they exist (not None)
            # Document structure (from LangChain metadata)
            if document.metadata.get('page') is not None:
                metadata['page_number'] = int(document.metadata['page'])
            
            if document.metadata.get('start_index') is not None:
                metadata['start_index'] = int(document.metadata['start_index'])
            
            # Add PDF-specific metadata
            if file_path.suffix.lower() == '.pdf':
                if document.metadata.get('page_label') is not None:
                    metadata['page_label'] = str(document.metadata['page_label'])
                if document.metadata.get('total_pages') is not None:
                    metadata['total_pages'] = int(document.metadata['total_pages'])
            
            # Add XML-specific metadata
            if file_path.suffix.lower() == '.xml':
                if document.metadata.get('category') is not None:
                    metadata['element_type'] = str(document.metadata['category'])
                if document.metadata.get('parent_id') is not None:
                    metadata['parent_element'] = str(document.metadata['parent_id'])
            
            # Add Excel-specific metadata
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                if document.metadata.get('sheet_name') is not None:
                    metadata['sheet_name'] = str(document.metadata['sheet_name'])
                if document.metadata.get('row') is not None:
                    metadata['row_number'] = int(document.metadata['row'])
            
            # CRITICAL: Filter out any None values that might have slipped through
            # ChromaDB doesn't accept None values
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            return filtered_metadata
    
    def embed_document(
            self,
            code_type: str,
            year: int,
            document_path: Path,
            force_recreate: bool = False
        ) -> Dict[str, Any]:
            """
            Embed a single raw document with rich metadata
            
            Args:
                code_type: Code type (icd10cm, icd10pcs, cpt)
                year: Year
                document_path: Path to document
                force_recreate: Force recreate collection
            
            Returns:
                Embedding results
            """
            collection_name = f"cms_docs_{code_type}_{year}"
            
            logger.info(f"\n  📄 {document_path.name}")
            
            # Get or create collection with rich metadata schema
            try:
                if force_recreate:
                    try:
                        self.chroma_client.delete_collection(collection_name)
                    except:
                        pass
                
                collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "code_type": code_type,
                        "year": str(year),
                        "description": f"CMS {code_type.upper()} documents for {year} with rich metadata"
                    }
                )
            except Exception as e:
                logger.error(f"    ✗ Failed to create collection: {e}")
                return {'success': False, 'error': str(e)}
            
            # Load document using LangChain
            documents = self.load_document(document_path)
            
            if not documents:
                logger.warning(f"    ⚠ No content loaded")
                return {'success': False, 'error': 'No content loaded'}
            
            # Chunk documents using LangChain (preserves metadata)
            try:
                chunks = self.text_splitter.split_documents(documents)
            except Exception as e:
                logger.error(f"    ✗ Failed to chunk: {e}")
                return {'success': False, 'error': str(e)}
            
            if not chunks:
                logger.warning(f"    ⚠ No chunks created")
                return {'success': False, 'error': 'No chunks created'}
            
            logger.info(f"    ✓ Created {len(chunks)} chunks")
            
            # Generate embeddings in batches with rich metadata
            batch_size = 100
            total_embedded = 0
            total_chunks = len(chunks)
            
            # Add progress bar for batches
            num_batches = (len(chunks) + batch_size - 1) // batch_size
            pbar = tqdm(
                total=num_batches,
                desc=f"    Embedding {document_path.name[:30]}",
                unit="batch",
                leave=False
            )
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk.page_content for chunk in batch_chunks]
                
                # Generate embeddings for batch
                try:
                    batch_embeddings = self.generate_embeddings(batch_texts)
                except Exception as e:
                    logger.error(f"    ✗ Failed to generate embeddings: {e}")
                    pbar.update(1)
                    continue
                
                # Prepare batch data with rich metadata
                ids = []
                documents_batch = []
                metadatas = []
                
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    chunk_idx = i + j
                    
                    # Create unique ID
                    chunk_id = f"{code_type}_{year}_{document_path.stem}_chunk_{chunk_idx}"
                    
                    # Create rich metadata
                    rich_metadata = self.create_rich_metadata(
                        document=chunk,
                        file_path=document_path,
                        code_type=code_type,
                        year=year,
                        chunk_index=chunk_idx,
                        total_chunks=total_chunks
                    )
                    
                    ids.append(chunk_id)
                    documents_batch.append(chunk.page_content)
                    metadatas.append(rich_metadata)
                
                # Add to collection
                try:
                    collection.add(
                        ids=ids,
                        documents=documents_batch,
                        embeddings=batch_embeddings,
                        metadatas=metadatas
                    )
                    total_embedded += len(batch_chunks)
                except Exception as e:
                    logger.error(f"    ✗ Failed to add batch: {e}")
                
                pbar.update(1)
            
            pbar.close()
            
            logger.info(f"    ✓ Embedded {total_embedded} chunks with rich metadata")
            
            return {
                'success': True,
                'document': document_path.name,
                'chunks_embedded': total_embedded,
                'collection': collection_name
            }
    
    def embed_code_type(
            self,
            code_type: str,
            year: int,
            force_recreate: bool = False,
            include_xsd: bool = False
        ) -> Dict[str, Any]:
            """Embed all documents for a code type"""
            logger.info(f"\n{'='*80}")
            logger.info(f"EMBEDDING: {code_type.upper()} - Year {year}")
            logger.info(f"{'='*80}")
            
            # Find raw data directory
            code_type_dir = RAW_DATA_DIR / code_type
            
            if not code_type_dir.exists():
                logger.error(f"  ✗ Directory not found: {code_type_dir}")
                return {'success': False, 'error': 'Directory not found'}
            
            # Find all extracted directories
            extracted_dirs = [d for d in code_type_dir.iterdir() if d.is_dir()]
            
            if not extracted_dirs:
                logger.error(f"  ✗ No extracted directories found")
                return {'success': False, 'error': 'No extracted directories'}
            
            total_chunks = 0
            embedded_files = []
            failed_files = []
            
            # Process each directory
            for extract_dir in extracted_dirs:
                logger.info(f"\n📁 {extract_dir.name}")
                
                # Find all supported files
                txt_files = list(extract_dir.glob("*.txt"))
                xml_files = list(extract_dir.glob("*.xml"))
                pdf_files = list(extract_dir.glob("*.pdf"))
                xlsx_files = list(extract_dir.glob("*.xlsx")) + list(extract_dir.glob("*.xls"))
                
                all_files = txt_files + xml_files + pdf_files + xlsx_files
                
                # Optionally include XSD files
                if include_xsd:
                    xsd_files = list(extract_dir.glob("*.xsd"))
                    all_files += xsd_files
                
                if not all_files:
                    logger.warning(f"  ⚠ No supported files found")
                    continue
                
                logger.info(f"  Found {len(all_files)} files:")
                logger.info(f"    TXT: {len(txt_files)}, XML: {len(xml_files)}, PDF: {len(pdf_files)}, XLSX: {len(xlsx_files)}")
                
                # Add progress bar for files
                file_pbar = tqdm(
                    all_files,
                    desc=f"  Processing {extract_dir.name[:30]}",
                    unit="file"
                )
                
                # Embed each file
                for file_path in file_pbar:
                    file_pbar.set_postfix({"file": file_path.name[:20]})
                    
                    result = self.embed_document(
                        code_type=code_type,
                        year=year,
                        document_path=file_path,
                        force_recreate=force_recreate and file_path == all_files[0]
                    )
                    
                    if result['success']:
                        total_chunks += result['chunks_embedded']
                        embedded_files.append(file_path.name)
                    else:
                        failed_files.append(file_path.name)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"EMBEDDING COMPLETE: {code_type.upper()}")
            logger.info(f"{'='*80}")
            logger.info(f"  ✓ Files embedded: {len(embedded_files)}")
            if failed_files:
                logger.warning(f"  ✗ Files failed: {len(failed_files)}")
                for failed in failed_files:
                    logger.warning(f"      - {failed}")
            logger.info(f"  ✓ Total chunks: {total_chunks:,}")
            
            return {
                'success': True,
                'code_type': code_type,
                'files_embedded': len(embedded_files),
                'files_failed': len(failed_files),
                'total_chunks': total_chunks
            }

# ============================================================================
# 3. COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Embed raw CMS documents with rich metadata for robust retrieval"
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=CMSConfig.CURRENT_YEAR,
        help=f'Fiscal year (default: {CMSConfig.CURRENT_YEAR})'
    )
    
    parser.add_argument(
        '--codes',
        nargs='+',
        default=['all'],
        choices=['all', 'icd10cm', 'icd10pcs', 'cpt'],
        help='Code types to embed'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreate collections'
    )
    
    parser.add_argument(
        '--include-xsd',
        action='store_true',
        help='Include XSD schema files'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size (default: 1000)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Chunk overlap (default: 200)'
    )
    
    args = parser.parse_args()
    
    # Expand 'all'
    if 'all' in args.codes:
        code_types = ['icd10cm', 'icd10pcs', 'cpt']
    else:
        code_types = args.codes
    
    logger.info("=" * 80)
    logger.info("CMS RAW DOCUMENT EMBEDDER - RICH METADATA MODE")
    logger.info("=" * 80)
    logger.info(f"Supported: TXT, XML, PDF, XLSX")
    logger.info(f"Metadata: Source tracking, page numbers, context preservation")
    logger.info(f"Chunking: size={args.chunk_size}, overlap={args.chunk_overlap}")
    logger.info("=" * 80)
    
    # Initialize embedder
    embedder = RawCMSDocumentEmbedder(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Embed each code type
    total_chunks = 0
    total_files = 0
    
    for code_type in code_types:
        result = embedder.embed_code_type(
            code_type=code_type,
            year=args.year,
            force_recreate=args.force,
            include_xsd=args.include_xsd
        )
        
        if result['success']:
            total_chunks += result['total_chunks']
            total_files += result['files_embedded']
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("EMBEDDING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total files embedded: {total_files}")
    logger.info(f"Total document chunks: {total_chunks:,}")
    logger.info(f"✓ All chunks include rich metadata for robust retrieval")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()