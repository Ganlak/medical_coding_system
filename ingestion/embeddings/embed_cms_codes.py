"""
Medical Coding System - CMS Code Embedder
Embed normalized CMS codes into ChromaDB for semantic search
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import chromadb
from chromadb.config import Settings


# ============================================================================
# 1. IMPORT CONFIGURATION (NO FALLBACK - USE REAL CONFIG)
# ============================================================================
from config.settings import (
    ChromaDBConfig, EmbeddingConfig, CMSConfig,
    PROCESSED_DATA_DIR, VECTOR_DB_DIR,
    AzureOpenAIConfig
)
from utils.logger import get_logger
from ingestion.cms.year_manager import YearManager

logger = get_logger(__name__)


# ============================================================================
# 2. EMBEDDING CLIENT
# ============================================================================

class EmbeddingClient:
    """Azure OpenAI embedding client"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        try:
            from openai import AzureOpenAI
            
            # Validate configuration first
            if not AzureOpenAIConfig.API_KEY:
                raise ValueError("AZURE_OPENAI_API_KEY not found in environment")
            
            if not AzureOpenAIConfig.ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment")
            
            self.client = AzureOpenAI(
                api_key=AzureOpenAIConfig.API_KEY,
                api_version=AzureOpenAIConfig.API_VERSION,
                azure_endpoint=AzureOpenAIConfig.ENDPOINT
            )
            self.deployment = AzureOpenAIConfig.EMBED_DEPLOYMENT
            logger.info(f"✓ Azure OpenAI client initialized")
            logger.info(f"  Endpoint: {AzureOpenAIConfig.ENDPOINT}")
            logger.info(f"  Deployment: {self.deployment}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
        
        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for batch of texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embeddings
        """
        max_retries = EmbeddingConfig.MAX_RETRIES
        retry_delay = EmbeddingConfig.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.deployment
                )
                return [item.embedding for item in response.data]
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise
        
        return []


# ============================================================================
# 3. CHROMA DB MANAGER
# ============================================================================

class ChromaDBManager:
    """Manage ChromaDB collections"""
    
    def __init__(self):
        """Initialize ChromaDB client"""
        persist_dir = Path(ChromaDBConfig.PERSIST_DIRECTORY)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info(f"✓ ChromaDB initialized")
        logger.info(f"  Persist directory: {persist_dir}")
    
    def get_or_create_collection(self, collection_name: str):
        """
        Get existing collection or create new one
        
        Args:
            collection_name: Name of collection
        
        Returns:
            ChromaDB collection
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": ChromaDBConfig.DISTANCE_METRIC}
            )
            logger.info(f"✓ Collection ready: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            self.client.get_collection(collection_name)
            return True
        except:
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get collection statistics"""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            return {
                'name': collection_name,
                'count': count,
                'exists': True
            }
        except:
            return {
                'name': collection_name,
                'count': 0,
                'exists': False
            }
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"✓ Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        collections = self.client.list_collections()
        return [col.name for col in collections]


# ============================================================================
# 4. CODE EMBEDDER
# ============================================================================

class CMSCodeEmbedder:
    """Embed CMS codes into ChromaDB"""
    
    def __init__(self):
        """Initialize embedder"""
        self.embedding_client = EmbeddingClient()
        self.chroma_manager = ChromaDBManager()
        self.year_manager = YearManager()
    
    def prepare_text_for_embedding(self, code_row: pd.Series, code_type: str) -> str:
        """
        Prepare text for embedding from code row
        
        Args:
            code_row: Pandas Series with code data
            code_type: Type of code
        
        Returns:
            Formatted text for embedding
        """
        code = code_row['code']
        description = code_row.get('description', '')
        
        # Format: "CODE: description [code_type]"
        text = f"{code}: {description} [{code_type}]"
        
        return text
    
    def embed_codes(
        self,
        code_type: str,
        year: int,
        force_recreate: bool = False,
        batch_size: Optional[int] = None
    ) -> Dict:
        """
        Embed codes for specific type and year
        
        Args:
            code_type: Code type (icd10cm, icd10pcs, cpt)
            year: Year
            force_recreate: If True, delete and recreate collection
            batch_size: Batch size for embedding (uses config default if None)
        
        Returns:
            Dictionary with embedding results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"EMBEDDING: {code_type.upper()} - Year {year}")
        logger.info(f"{'='*80}")
        
        # Load codes
        logger.info("Loading codes...")
        df = self.year_manager.load_codes(code_type, year)
        
        if df.empty:
            logger.error(f"No codes found for {code_type} year {year}")
            return {'success': False, 'error': 'No codes found'}
        
        total_codes = len(df)
        logger.info(f"  Total codes to embed: {total_codes:,}")
        
        # Get or create collection
        collection_name = ChromaDBConfig.get_collection_name(code_type, year)
        
        if force_recreate and self.chroma_manager.collection_exists(collection_name):
            logger.info(f"  Deleting existing collection...")
            self.chroma_manager.delete_collection(collection_name)
        
        collection = self.chroma_manager.get_or_create_collection(collection_name)
        
        # Check if already embedded
        existing_count = collection.count()
        if existing_count > 0 and not force_recreate:
            logger.warning(f"  Collection already has {existing_count:,} embeddings")
            logger.info(f"  Skipping to avoid duplicates. Use --force to recreate.")
            return {
                'success': True,
                'code_type': code_type,
                'year': year,
                'collection_name': collection_name,
                'embedded_count': 0,
                'skipped': True,
                'message': 'Collection already exists',
                'total_in_collection': existing_count
            }
        
        # Prepare for embedding
        batch_size = batch_size or EmbeddingConfig.BATCH_SIZE
        
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Starting embedding process...")
        
        # Process in batches
        embedded_count = 0
        failed_count = 0
        start_time = time.time()
        
        for batch_start in tqdm(range(0, total_codes, batch_size), desc="Embedding batches"):
            batch_end = min(batch_start + batch_size, total_codes)
            batch_df = df.iloc[batch_start:batch_end]
            
            try:
                # Prepare texts
                texts = [
                    self.prepare_text_for_embedding(row, code_type)
                    for _, row in batch_df.iterrows()
                ]
                
                # Create embeddings
                embeddings = self.embedding_client.create_embeddings_batch(texts)
                
                # Prepare metadata
                ids = [f"{code_type}_{year}_{row['code']}" for _, row in batch_df.iterrows()]
                metadatas = [
                    {
                        'code': row['code'],
                        'description': row.get('description', ''),
                        'code_type': code_type,
                        'year': year,
                        **{k: str(v) for k, v in row.items() 
                           if k not in ['code', 'description'] and pd.notna(v)}
                    }
                    for _, row in batch_df.iterrows()
                ]
                
                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                embedded_count += len(batch_df)
            
            except Exception as e:
                logger.error(f"  Batch {batch_start}-{batch_end} failed: {e}")
                failed_count += len(batch_df)
                continue
        
        duration = time.time() - start_time
        
        # Results
        logger.info(f"\n{'='*80}")
        logger.info(f"EMBEDDING COMPLETE: {code_type.upper()} - Year {year}")
        logger.info(f"{'='*80}")
        logger.info(f"  ✓ Successfully embedded: {embedded_count:,}")
        if failed_count > 0:
            logger.info(f"  ✗ Failed: {failed_count:,}")
        logger.info(f"  ⏱  Duration: {duration:.1f}s")
        logger.info(f"  📊 Codes/second: {embedded_count/duration:.1f}")
        logger.info(f"  📦 Collection: {collection_name}")
        logger.info(f"  💾 Total in collection: {collection.count():,}")
        
        return {
            'success': True,
            'code_type': code_type,
            'year': year,
            'collection_name': collection_name,
            'embedded_count': embedded_count,
            'failed_count': failed_count,
            'duration_seconds': duration,
            'total_in_collection': collection.count(),
            'skipped': False
        }


# ============================================================================
# 5. COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Embed CMS codes into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed all 2025 codes
  python -m ingestion.embeddings.embed_cms_codes --year 2025 --codes all
  
  # Embed only ICD-10-CM
  python -m ingestion.embeddings.embed_cms_codes --year 2025 --codes icd10cm
  
  # Force recreate collection
  python -m ingestion.embeddings.embed_cms_codes --year 2025 --codes icd10cm --force
  
  # List existing collections
  python -m ingestion.embeddings.embed_cms_codes --list
  
  # Show collection stats
  python -m ingestion.embeddings.embed_cms_codes --stats --year 2025 --codes all
        """
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=CMSConfig.CURRENT_YEAR,
        help=f'Year to embed (default: {CMSConfig.CURRENT_YEAR})'
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
        help='Force recreate collections (deletes existing)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {EmbeddingConfig.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all collections'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show collection statistics'
    )
    
    args = parser.parse_args()
    
    # Initialize
    chroma_manager = ChromaDBManager()
    
    # List collections
    if args.list:
        collections = chroma_manager.list_collections()
        print(f"\n{'='*80}")
        print("CHROMADB COLLECTIONS")
        print(f"{'='*80}")
        
        if collections:
            for col in sorted(collections):
                stats = chroma_manager.get_collection_stats(col)
                print(f"  • {col}: {stats['count']:,} embeddings")
        else:
            print("  No collections found")
        
        print()
        return 0
    
    # Show stats
    if args.stats:
        code_types = ['icd10cm', 'icd10pcs', 'cpt'] if 'all' in args.codes else args.codes
        
        print(f"\n{'='*80}")
        print(f"COLLECTION STATISTICS - Year {args.year}")
        print(f"{'='*80}")
        
        for code_type in code_types:
            collection_name = ChromaDBConfig.get_collection_name(code_type, args.year)
            stats = chroma_manager.get_collection_stats(collection_name)
            
            status = "✓" if stats['exists'] else "✗"
            count = f"{stats['count']:,}" if stats['exists'] else "Not found"
            
            print(f"  {status} {code_type.upper()}: {count}")
        
        print()
        return 0
    
    # Embed codes
    code_types = ['icd10cm', 'icd10pcs', 'cpt'] if 'all' in args.codes else args.codes
    
    print(f"\n{'='*80}")
    print("CMS CODE EMBEDDER")
    print(f"{'='*80}")
    print(f"Year: {args.year}")
    print(f"Code Types: {', '.join(code_types)}")
    print(f"Force Recreate: {args.force}")
    if args.batch_size:
        print(f"Batch Size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    embedder = CMSCodeEmbedder()
    
    results = []
    for code_type in code_types:
        result = embedder.embed_codes(
            code_type=code_type,
            year=args.year,
            force_recreate=args.force,
            batch_size=args.batch_size
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("EMBEDDING SUMMARY")
    print(f"{'='*80}")
    
    total_embedded = sum(r.get('embedded_count', 0) for r in results if r.get('success'))
    total_failed = sum(r.get('failed_count', 0) for r in results if r.get('success'))
    skipped = [r for r in results if r.get('skipped')]
    
    print(f"Total embedded: {total_embedded:,}")
    if total_failed > 0:
        print(f"Total failed: {total_failed:,}")
    if skipped:
        print(f"Skipped (already exist): {len(skipped)}")
    
    print(f"\nCollections:")
    for result in results:
        if result.get('success'):
            status = "↷" if result.get('skipped') else "✓"
            print(f"  {status} {result['collection_name']}: {result['total_in_collection']:,}")
        else:
            print(f"  ✗ {result.get('code_type', 'unknown')}: Failed")
    
    print(f"{'='*80}\n")
    
    return 0


# ============================================================================
# 6. MAIN BLOCK
# ============================================================================

if __name__ == "__main__":
    """
    Test and run CMS code embedder
    Usage: python ingestion/embeddings/embed_cms_codes.py --year 2025 --codes all
    """
    sys.exit(main())