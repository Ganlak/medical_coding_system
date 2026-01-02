"""
Medical Coding System - Vector Store Interface
Unified interface for ChromaDB vector operations with year-wise code retrieval
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Any
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection


# ============================================================================
# 1. IMPORT CONFIGURATION
# ============================================================================
from config.settings import ChromaDBConfig, CMSConfig, RAGConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 2. VECTOR STORE CLASS
# ============================================================================

class VectorStore:
    """
    Unified vector store interface for medical code retrieval
    Supports year-wise collections and hybrid search
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize vector store
        
        Args:
            persist_directory: ChromaDB persist directory (uses config default if None)
        """
        persist_dir = persist_directory or ChromaDBConfig.PERSIST_DIRECTORY
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        self.current_year = CMSConfig.CURRENT_YEAR
        
        logger.info(f"✓ Vector store initialized")
        logger.info(f"  Persist directory: {persist_path}")
    
    def get_collection(
        self,
        code_type: str,
        year: Optional[int] = None
    ) -> Optional[Collection]:
        """
        Get collection for specific code type and year
        
        Args:
            code_type: Code type (icd10cm, icd10pcs, cpt)
            year: Year (uses current year if None)
        
        Returns:
            ChromaDB collection or None if not found
        """
        year = year or self.current_year
        collection_name = ChromaDBConfig.get_collection_name(code_type, year)
        
        try:
            collection = self.client.get_collection(collection_name)
            return collection
        except Exception as e:
            logger.warning(f"Collection not found: {collection_name}")
            return None
    
    def search_codes(
        self,
        query: str,
        code_type: str,
        year: Optional[int] = None,
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Semantic search for medical codes
        
        Args:
            query: Search query
            code_type: Code type to search
            year: Year to search (uses current year if None)
            n_results: Number of results to return
            where: Optional metadata filter
        
        Returns:
            List of search results with code, description, and similarity
        """
        collection = self.get_collection(code_type, year)
        
        if not collection:
            logger.error(f"Collection not available for {code_type} year {year}")
            return []
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'code': results['metadatas'][0][i].get('code', ''),
                        'description': results['metadatas'][0][i].get('description', ''),
                        'code_type': results['metadatas'][0][i].get('code_type', code_type),
                        'year': results['metadatas'][0][i].get('year', year),
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i]
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query[:50]}...'")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_all_types(
        self,
        query: str,
        year: Optional[int] = None,
        n_results_per_type: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Search across all code types
        
        Args:
            query: Search query
            year: Year to search
            n_results_per_type: Results per code type
        
        Returns:
            Dictionary mapping code type to results
        """
        results = {}
        
        for code_type in CMSConfig.CODE_TYPES:
            code_results = self.search_codes(
                query=query,
                code_type=code_type,
                year=year,
                n_results=n_results_per_type
            )
            
            if code_results:
                results[code_type] = code_results
        
        return results
    
    def get_code_by_id(
        self,
        code: str,
        code_type: str,
        year: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get specific code by ID
        
        Args:
            code: Medical code
            code_type: Code type
            year: Year
        
        Returns:
            Code information or None
        """
        collection = self.get_collection(code_type, year)
        
        if not collection:
            return None
        
        year = year or self.current_year
        doc_id = f"{code_type}_{year}_{code}"
        
        try:
            result = collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            
            if result['ids']:
                return {
                    'code': result['metadatas'][0].get('code', ''),
                    'description': result['metadatas'][0].get('description', ''),
                    'code_type': result['metadatas'][0].get('code_type', code_type),
                    'year': result['metadatas'][0].get('year', year),
                    'metadata': result['metadatas'][0]
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get code {code}: {e}")
            return None
    
    def search_with_filters(
        self,
        query: str,
        code_type: str,
        filters: Dict[str, Any],
        year: Optional[int] = None,
        n_results: int = 10
    ) -> List[Dict]:
        """
        Search with metadata filters
        
        Args:
            query: Search query
            code_type: Code type
            filters: Metadata filters (e.g., {'category': 'E11'})
            year: Year
            n_results: Number of results
        
        Returns:
            Filtered search results
        """
        collection = self.get_collection(code_type, year)
        
        if not collection:
            return []
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'code': results['metadatas'][0][i].get('code', ''),
                        'description': results['metadatas'][0][i].get('description', ''),
                        'similarity': 1 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            return []
    
    def get_collection_stats(
        self,
        code_type: str,
        year: Optional[int] = None
    ) -> Dict:
        """
        Get statistics for a collection
        
        Args:
            code_type: Code type
            year: Year
        
        Returns:
            Collection statistics
        """
        collection = self.get_collection(code_type, year)
        
        if not collection:
            return {
                'exists': False,
                'count': 0
            }
        
        try:
            count = collection.count()
            return {
                'exists': True,
                'count': count,
                'name': collection.name,
                'code_type': code_type,
                'year': year or self.current_year
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'exists': False, 'count': 0}
    
    def list_all_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def verify_all_years(self) -> Dict[str, Dict[int, bool]]:
        """
        Verify all year-wise collections exist
        
        Returns:
            Dictionary mapping code type to year availability
        """
        verification = {}
        
        for code_type in CMSConfig.CODE_TYPES:
            verification[code_type] = {}
            
            for year in CMSConfig.get_all_years():
                collection = self.get_collection(code_type, year)
                verification[code_type][year] = collection is not None
        
        return verification


# ============================================================================
# 3. HYBRID SEARCH IMPLEMENTATION
# ============================================================================

class HybridSearchEngine:
    """
    Hybrid search combining semantic and keyword-based retrieval
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize hybrid search engine
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.semantic_weight = RAGConfig.SEMANTIC_WEIGHT
        self.keyword_weight = RAGConfig.KEYWORD_WEIGHT
    
    def keyword_search(
        self,
        query: str,
        code_type: str,
        year: Optional[int] = None,
        n_results: int = 20
    ) -> List[Dict]:
        """
        Keyword-based search (using metadata filters)
        
        Args:
            query: Search query
            code_type: Code type
            year: Year
            n_results: Number of results
        
        Returns:
            Keyword search results
        """
        # Simple keyword matching - can be enhanced with BM25
        collection = self.vector_store.get_collection(code_type, year)
        
        if not collection:
            return []
        
        try:
            # Get all codes and filter by keyword match
            # This is a simplified implementation
            results = collection.get(
                limit=n_results * 5,  # Get more for filtering
                include=['documents', 'metadatas']
            )
            
            query_lower = query.lower()
            matched = []
            
            if results['ids']:
                for i, doc in enumerate(results['documents']):
                    if query_lower in doc.lower():
                        score = doc.lower().count(query_lower) / len(doc.split())
                        matched.append({
                            'code': results['metadatas'][i].get('code', ''),
                            'description': results['metadatas'][i].get('description', ''),
                            'score': score,
                            'metadata': results['metadatas'][i]
                        })
            
            # Sort by score and limit
            matched.sort(key=lambda x: x['score'], reverse=True)
            return matched[:n_results]
        
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        code_type: str,
        year: Optional[int] = None,
        n_results: int = 10
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword results
        
        Args:
            query: Search query
            code_type: Code type
            year: Year
            n_results: Number of results
        
        Returns:
            Hybrid search results with combined scores
        """
        # Semantic search
        semantic_results = self.vector_store.search_codes(
            query=query,
            code_type=code_type,
            year=year,
            n_results=n_results * 2
        )
        
        # Keyword search
        keyword_results = self.keyword_search(
            query=query,
            code_type=code_type,
            year=year,
            n_results=n_results * 2
        )
        
        # Combine and re-rank
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            code = result['code']
            combined[code] = {
                **result,
                'semantic_score': result['similarity'],
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            code = result['code']
            if code in combined:
                combined[code]['keyword_score'] = result['score']
            else:
                combined[code] = {
                    **result,
                    'semantic_score': 0.0,
                    'keyword_score': result['score']
                }
        
        # Calculate hybrid score
        for code in combined:
            combined[code]['hybrid_score'] = (
                self.semantic_weight * combined[code]['semantic_score'] +
                self.keyword_weight * combined[code]['keyword_score']
            )
        
        # Sort by hybrid score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        return sorted_results[:n_results]


# ============================================================================
# 4. MAIN BLOCK - Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate vector store
    Usage: python database/vector_store.py
    """
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - VECTOR STORE TEST")
    print("=" * 80)
    print()
    
    # 1. Initialize vector store
    print("📦 Initializing vector store...")
    vector_store = VectorStore()
    print()
    
    # 2. List all collections
    print("📋 Available collections:")
    print("-" * 80)
    collections = vector_store.list_all_collections()
    for col in collections:
        print(f"  • {col}")
    print()
    
    # 3. Get collection stats
    print("📊 Collection statistics:")
    print("-" * 80)
    for code_type in CMSConfig.CODE_TYPES:
        stats = vector_store.get_collection_stats(code_type)
        if stats['exists']:
            print(f"  ✓ {code_type.upper()}: {stats['count']:,} codes")
        else:
            print(f"  ✗ {code_type.upper()}: Not found")
    print()
    
    # 4. Test semantic search
    print("🔍 Testing semantic search:")
    print("-" * 80)
    
    test_queries = [
        ("diabetes", "icd10cm"),
        ("heart surgery", "icd10pcs"),
        ("office visit", "cpt")
    ]
    
    for query, code_type in test_queries:
        print(f"\nQuery: '{query}' in {code_type.upper()}")
        results = vector_store.search_codes(query, code_type, n_results=5)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['code']}: {result['description'][:60]}...")
                print(f"     Similarity: {result['similarity']:.3f}")
        else:
            print("  No results found")
    print()
    
    # 5. Test multi-type search
    print("🔍 Testing multi-type search:")
    print("-" * 80)
    query = "hypertension"
    print(f"Query: '{query}' across all types\n")
    
    all_results = vector_store.search_all_types(query, n_results_per_type=3)
    
    for code_type, results in all_results.items():
        print(f"{code_type.upper()}:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['code']}: {result['description'][:60]}...")
        print()
    
    # 6. Test code lookup
    print("🔍 Testing direct code lookup:")
    print("-" * 80)
    
    test_codes = [
        ("E11.9", "icd10cm"),
        ("0DBJ8ZZ", "icd10pcs")
    ]
    
    for code, code_type in test_codes:
        print(f"\nLooking up: {code} ({code_type.upper()})")
        result = vector_store.get_code_by_id(code, code_type)
        
        if result:
            print(f"  ✓ Found: {result['description']}")
        else:
            print(f"  ✗ Not found")
    print()
    
    # 7. Test hybrid search
    print("🔍 Testing hybrid search:")
    print("-" * 80)
    
    hybrid_engine = HybridSearchEngine(vector_store)
    query = "type 2 diabetes"
    print(f"Query: '{query}' (hybrid)\n")
    
    hybrid_results = hybrid_engine.hybrid_search(query, "icd10cm", n_results=5)
    
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. {result['code']}: {result['description'][:60]}...")
        print(f"   Hybrid: {result['hybrid_score']:.3f} "
              f"(Semantic: {result['semantic_score']:.3f}, "
              f"Keyword: {result['keyword_score']:.3f})")
    print()
    
    # 8. Verify all years
    print("📅 Verifying year-wise collections:")
    print("-" * 80)
    verification = vector_store.verify_all_years()
    
    for code_type, years in verification.items():
        print(f"\n{code_type.upper()}:")
        for year, exists in years.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {year}")
    print()
    
    # 9. Summary
    print("=" * 80)
    print("✅ VECTOR STORE TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Usage Examples:")
    print()
    print("  from database.vector_store import VectorStore")
    print("  vector_store = VectorStore()")
    print("  results = vector_store.search_codes('diabetes', 'icd10cm', n_results=10)")
    print()