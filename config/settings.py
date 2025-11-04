"""
Medical Coding System - Production Configuration
Complete system settings with year-wise CMS management, RAG configuration, and HIPAA compliance
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# 1. BASE PATHS & DIRECTORIES
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

# CMS code directories (year-wise)
ICD10CM_RAW_DIR = RAW_DATA_DIR / "icd10cm"
ICD10PCS_RAW_DIR = RAW_DATA_DIR / "icd10pcs"
CPT_RAW_DIR = RAW_DATA_DIR / "cpt"

# Create all necessary directories
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, UPLOADS_DIR,
    OUTPUTS_DIR, CACHE_DIR, LOGS_DIR, VECTOR_DB_DIR,
    ICD10CM_RAW_DIR, ICD10PCS_RAW_DIR, CPT_RAW_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. AZURE OPENAI CONFIGURATION
# ============================================================================
class AzureOpenAIConfig:
    """Azure OpenAI configuration"""
    ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    # Deployments
    CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")
    
    # LLM Parameters
    TEMPERATURE = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "4096"))
    TOP_P = float(os.getenv("AZURE_OPENAI_TOP_P", "0.95"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate Azure OpenAI configuration"""
        required = [cls.ENDPOINT, cls.API_KEY, cls.CHAT_DEPLOYMENT, cls.EMBED_DEPLOYMENT]
        if not all(required):
            raise ValueError(
                "Missing Azure OpenAI configuration. Please check .env file:\n"
                f"ENDPOINT: {'✓' if cls.ENDPOINT else '✗'}\n"
                f"API_KEY: {'✓' if cls.API_KEY else '✗'}\n"
                f"CHAT_DEPLOYMENT: {'✓' if cls.CHAT_DEPLOYMENT else '✗'}\n"
                f"EMBED_DEPLOYMENT: {'✓' if cls.EMBED_DEPLOYMENT else '✗'}"
            )
        return True

# ============================================================================
# 3. VECTOR DATABASE CONFIGURATION (ChromaDB)
# ============================================================================
class ChromaDBConfig:
    """ChromaDB configuration"""
    PERSIST_DIRECTORY = str(VECTOR_DB_DIR / os.getenv(
        "CHROMA_PERSIST_DIRECTORY", ".chroma_production"
    ).lstrip("./"))
    
    COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "medical_codes")
    
    # Collection names (year-wise)
    COLLECTION_ICD10CM = os.getenv("CHROMA_COLLECTION_ICD10CM", "icd10cm_2025")
    COLLECTION_ICD10PCS = os.getenv("CHROMA_COLLECTION_ICD10PCS", "icd10pcs_2025")
    COLLECTION_CPT = os.getenv("CHROMA_COLLECTION_CPT", "cpt_2025")
    COLLECTION_DOCUMENTS = os.getenv("CHROMA_COLLECTION_DOCUMENTS", "documents")
    
    # Retrieval settings
    TOP_K = int(os.getenv("CHROMA_TOP_K", "10"))
    SIMILARITY_THRESHOLD = float(os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.7"))
    
    # Distance metric
    DISTANCE_METRIC = "cosine"  # Options: cosine, l2, ip
    
    @classmethod
    def get_collection_name(cls, code_type: str, year: int) -> str:
        """Generate collection name for specific code type and year"""
        return f"{cls.COLLECTION_PREFIX}_{code_type}_{year}"
    
    @classmethod
    def get_all_collections(cls) -> List[str]:
        """Get list of all collection names"""
        return [
            cls.COLLECTION_ICD10CM,
            cls.COLLECTION_ICD10PCS,
            cls.COLLECTION_CPT,
            cls.COLLECTION_DOCUMENTS
        ]

# ============================================================================
# 4. CMS CODE MANAGEMENT (Year-wise)
# ============================================================================
class CMSConfig:
    """CMS code management configuration"""
    CURRENT_YEAR = int(os.getenv("CMS_CURRENT_YEAR", "2025"))
    PREVIOUS_YEARS = [
        int(year.strip()) 
        for year in os.getenv("CMS_PREVIOUS_YEARS", "2024,2023,2022").split(",")
    ]
    
    AUTO_UPDATE = os.getenv("CMS_AUTO_UPDATE", "true").lower() == "true"
    UPDATE_CHECK_INTERVAL = os.getenv("CMS_UPDATE_CHECK_INTERVAL", "weekly")
    
    # CMS data paths
    RAW_DATA_DIR = Path(os.getenv("CMS_RAW_DATA_DIR", str(RAW_DATA_DIR)))
    PROCESSED_DATA_DIR = Path(os.getenv("CMS_PROCESSED_DATA_DIR", str(PROCESSED_DATA_DIR)))
    
    # Code type definitions
    CODE_TYPES = ["icd10cm", "icd10pcs", "cpt"]
    
    @classmethod
    def get_all_years(cls) -> List[int]:
        """Get all years (current + previous)"""
        return [cls.CURRENT_YEAR] + cls.PREVIOUS_YEARS
    
    @classmethod
    def get_processed_file_path(cls, code_type: str, year: int) -> Path:
        """Get path to processed code file"""
        return cls.PROCESSED_DATA_DIR / f"{code_type}_{year}.csv"

# ============================================================================
# 5. EMBEDDING CONFIGURATION
# ============================================================================
class EmbeddingConfig:
    """Embedding configuration"""
    MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    CACHE_ENABLED = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    
    # Embedding generation settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

# ============================================================================
# 6. RETRIEVAL & RAG CONFIGURATION
# ============================================================================
class RAGConfig:
    """RAG (Retrieval Augmented Generation) configuration"""
    
    # Hybrid search weights
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_SEARCH_WEIGHT", "0.7"))
    KEYWORD_WEIGHT = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))
    
    # Re-ranking
    RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() == "true"
    RERANKING_MODEL = os.getenv("RERANKING_MODEL", "gpt-4.1")
    RERANKING_TOP_K = int(os.getenv("RERANKING_TOP_K", "5"))
    
    # Context window
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    CONTEXT_OVERLAP = int(os.getenv("CONTEXT_OVERLAP", "200"))
    
    # Retrieval strategy
    RETRIEVAL_STRATEGY = "hybrid"  # Options: semantic, keyword, hybrid
    
    @classmethod
    def validate_weights(cls) -> bool:
        """Validate that weights sum to 1.0"""
        total = cls.SEMANTIC_WEIGHT + cls.KEYWORD_WEIGHT
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Search weights must sum to 1.0. Got: {total}"
            )
        return True

# ============================================================================
# 7. CODING ENGINE CONFIGURATION
# ============================================================================
class CodingConfig:
    """Medical coding engine configuration"""
    
    # Confidence scoring
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6"))
    HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.85"))
    
    # Code validation
    STRICT_VALIDATION = os.getenv("STRICT_VALIDATION", "true").lower() == "true"
    ALLOW_DEPRECATED_CODES = os.getenv("ALLOW_DEPRECATED_CODES", "false").lower() == "true"
    
    # Suggestion limits
    MAX_SUGGESTIONS_PER_DIAGNOSIS = 5
    MAX_SUGGESTIONS_PER_PROCEDURE = 5
    
    # Code validation rules
    REQUIRE_SPECIFICITY = True
    CHECK_CODE_COMPATIBILITY = True

# ============================================================================
# 8. DOCUMENT PROCESSING CONFIGURATION
# ============================================================================
class DocumentConfig:
    """Document processing configuration"""
    
    # Upload limits
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_FILE_TYPES = os.getenv("ALLOWED_FILE_TYPES", "pdf,csv,xlsx,txt").split(",")
    
    # PDF processing
    PDF_DPI = int(os.getenv("PDF_DPI", "300"))
    PDF_OCR_ENABLED = os.getenv("PDF_OCR_ENABLED", "true").lower() == "true"
    PDF_OCR_LANGUAGE = os.getenv("PDF_OCR_LANGUAGE", "eng")
    
    # CSV processing
    CSV_ENCODING = os.getenv("CSV_ENCODING", "utf-8")
    CSV_DELIMITER = os.getenv("CSV_DELIMITER", ",")
    
    # Processing options
    EXTRACT_TABLES = True
    EXTRACT_IMAGES = False
    PRESERVE_LAYOUT = True

# ============================================================================
# 9. LOGGING CONFIGURATION
# ============================================================================
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / os.getenv("LOG_FILE", "app.log").lstrip("./")
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Audit logging
    AUDIT_LOG_ENABLED = os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true"
    AUDIT_LOG_FILE = LOGS_DIR / os.getenv("AUDIT_LOG_FILE", "audit.log").lstrip("./")
    
    # Performance monitoring
    MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "60"))
    
    # Log format
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# 10. CACHING CONFIGURATION
# ============================================================================
class CacheConfig:
    """Caching configuration"""
    ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_DIR = CACHE_DIR
    TTL = int(os.getenv("CACHE_TTL", "3600"))  # seconds
    MAX_SIZE_MB = int(os.getenv("CACHE_MAX_SIZE_MB", "1000"))
    
    # Cache strategies
    CACHE_EMBEDDINGS = True
    CACHE_RETRIEVALS = True
    CACHE_LLM_RESPONSES = False  # Disabled for code suggestions

# ============================================================================
# 11. API CONFIGURATION (Optional)
# ============================================================================
class APIConfig:
    """API configuration (for future FastAPI deployment)"""
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    WORKERS = int(os.getenv("API_WORKERS", "4"))
    
    # Security
    API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    API_KEY = os.getenv("API_KEY", "")
    RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))

# ============================================================================
# 12. STREAMLIT UI CONFIGURATION
# ============================================================================
class StreamlitConfig:
    """Streamlit UI configuration"""
    SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    SERVER_HEADLESS = os.getenv("STREAMLIT_SERVER_HEADLESS", "false").lower() == "true"
    THEME = os.getenv("STREAMLIT_THEME", "light")
    
    # UI settings
    PAGE_TITLE = "Medical Coding System"
    PAGE_ICON = "🏥"
    LAYOUT = "wide"

# ============================================================================
# 13. SECURITY & COMPLIANCE
# ============================================================================
class SecurityConfig:
    """Security and HIPAA compliance configuration"""
    
    # HIPAA compliance
    HIPAA_MODE = os.getenv("HIPAA_MODE", "true").lower() == "true"
    PHI_ENCRYPTION = os.getenv("PHI_ENCRYPTION", "true").lower() == "true"
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "2555"))  # 7 years
    
    # Anonymization
    ANONYMIZE_PATIENT_DATA = os.getenv("ANONYMIZE_PATIENT_DATA", "true").lower() == "true"
    REDACT_PHI = os.getenv("REDACT_PHI", "true").lower() == "true"
    
    # PHI patterns to redact
    PHI_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{10}\b',              # Phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]

# ============================================================================
# 14. AGENT CONFIGURATION
# ============================================================================
class AgentConfig:
    """Multi-agent system configuration"""
    
    # Agent behavior
    MAX_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
    TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))  # seconds
    VERBOSE = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    RETRY_ATTEMPTS = int(os.getenv("AGENT_RETRY_ATTEMPTS", "3"))
    
    # Agent types
    AGENT_TYPES = ["extraction", "coding", "validation", "orchestrator"]
    
    # Agent coordination
    USE_ORCHESTRATOR = True
    PARALLEL_EXECUTION = False

# ============================================================================
# 15. EXPORT CONFIGURATION
# ============================================================================
class ExportConfig:
    """Export and reporting configuration"""
    
    # Report generation
    DEFAULT_FORMAT = os.getenv("EXPORT_FORMAT", "xlsx")
    INCLUDE_CONFIDENCE_SCORES = os.getenv("INCLUDE_CONFIDENCE_SCORES", "true").lower() == "true"
    INCLUDE_AUDIT_TRAIL = os.getenv("INCLUDE_AUDIT_TRAIL", "true").lower() == "true"
    
    # Excel export
    EXCEL_TEMPLATE_PATH = os.getenv("EXCEL_TEMPLATE_PATH", "./templates/export_template.xlsx")
    
    # Export options
    SUPPORTED_FORMATS = ["xlsx", "csv", "pdf", "json"]

# ============================================================================
# 16. SYSTEM CONFIGURATION
# ============================================================================
class SystemConfig:
    """General system configuration"""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Feature flags
    ENABLE_EXPERIMENTAL = os.getenv("ENABLE_EXPERIMENTAL_FEATURES", "false").lower() == "true"
    ENABLE_TELEMETRY = os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"
    
    # Multi-tenancy
    MULTI_TENANT_ENABLED = os.getenv("MULTI_TENANT_ENABLED", "false").lower() == "true"
    TENANT_ID = os.getenv("TENANT_ID", "default")
    
    # System info
    VERSION = "1.0.0"
    APP_NAME = "Medical Coding System"

# ============================================================================
# 17. VALIDATION & INITIALIZATION
# ============================================================================
def validate_configuration() -> bool:
    """Validate all configuration settings"""
    try:
        # Validate Azure OpenAI
        AzureOpenAIConfig.validate()
        
        # Validate RAG weights
        RAGConfig.validate_weights()
        
        # Validate directories exist
        for directory in [DATA_DIR, LOGS_DIR, VECTOR_DB_DIR]:
            if not directory.exists():
                raise FileNotFoundError(f"Required directory not found: {directory}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# ============================================================================
# 18. CONFIGURATION EXPORT
# ============================================================================
def get_config_summary() -> Dict[str, any]:
    """Get configuration summary for debugging"""
    return {
        "azure_openai": {
            "endpoint": AzureOpenAIConfig.ENDPOINT[:50] + "..." if AzureOpenAIConfig.ENDPOINT else None,
            "model": AzureOpenAIConfig.MODEL,
            "chat_deployment": AzureOpenAIConfig.CHAT_DEPLOYMENT,
        },
        "chromadb": {
            "collections": ChromaDBConfig.get_all_collections(),
            "top_k": ChromaDBConfig.TOP_K,
        },
        "cms": {
            "current_year": CMSConfig.CURRENT_YEAR,
            "code_types": CMSConfig.CODE_TYPES,
        },
        "rag": {
            "semantic_weight": RAGConfig.SEMANTIC_WEIGHT,
            "reranking_enabled": RAGConfig.RERANKING_ENABLED,
        },
        "system": {
            "version": SystemConfig.VERSION,
            "environment": SystemConfig.ENVIRONMENT,
            "debug": SystemConfig.DEBUG,
        }
    }

# Auto-validate on import
# ============================================================================
# 19. MAIN BLOCK - Testing & Validation
# ============================================================================
if __name__ == "__main__":
    """
    Test and validate configuration settings
    Usage: python config/settings.py
    """
    import json
    from datetime import datetime
    
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - CONFIGURATION VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Validate Configuration
    print("🔍 Validating configuration...")
    try:
        is_valid = validate_configuration()
        if is_valid:
            print("✅ Configuration validation: PASSED\n")
        else:
            print("❌ Configuration validation: FAILED\n")
            exit(1)
    except Exception as e:
        print(f"❌ Configuration validation: FAILED - {e}\n")
        exit(1)
    
    # 2. Display Configuration Summary
    print("📋 Configuration Summary:")
    print("-" * 80)
    config_summary = get_config_summary()
    print(json.dumps(config_summary, indent=2))
    print()
    
    # 3. Check Azure OpenAI
    print("🔑 Azure OpenAI Configuration:")
    print("-" * 80)
    print(f"  Endpoint: {AzureOpenAIConfig.ENDPOINT}")
    print(f"  API Key: {'✓ Set' if AzureOpenAIConfig.API_KEY else '✗ Missing'}")
    print(f"  Chat Deployment: {AzureOpenAIConfig.CHAT_DEPLOYMENT}")
    print(f"  Embed Deployment: {AzureOpenAIConfig.EMBED_DEPLOYMENT}")
    print(f"  Model: {AzureOpenAIConfig.MODEL}")
    print(f"  Temperature: {AzureOpenAIConfig.TEMPERATURE}")
    print(f"  Max Tokens: {AzureOpenAIConfig.MAX_TOKENS}")
    print()
    
    # 4. Check ChromaDB Configuration
    print("💾 ChromaDB Configuration:")
    print("-" * 80)
    print(f"  Persist Directory: {ChromaDBConfig.PERSIST_DIRECTORY}")
    print(f"  Collections:")
    for collection in ChromaDBConfig.get_all_collections():
        print(f"    - {collection}")
    print(f"  Top K: {ChromaDBConfig.TOP_K}")
    print(f"  Similarity Threshold: {ChromaDBConfig.SIMILARITY_THRESHOLD}")
    print()
    
    # 5. Check CMS Configuration
    print("📅 CMS Code Management:")
    print("-" * 80)
    print(f"  Current Year: {CMSConfig.CURRENT_YEAR}")
    print(f"  Previous Years: {', '.join(map(str, CMSConfig.PREVIOUS_YEARS))}")
    print(f"  All Years: {', '.join(map(str, CMSConfig.get_all_years()))}")
    print(f"  Code Types: {', '.join(CMSConfig.CODE_TYPES)}")
    print(f"  Auto Update: {CMSConfig.AUTO_UPDATE}")
    print()
    
    # 6. Check RAG Configuration
    print("🔍 RAG Configuration:")
    print("-" * 80)
    print(f"  Semantic Weight: {RAGConfig.SEMANTIC_WEIGHT}")
    print(f"  Keyword Weight: {RAGConfig.KEYWORD_WEIGHT}")
    print(f"  Re-ranking Enabled: {RAGConfig.RERANKING_ENABLED}")
    print(f"  Re-ranking Model: {RAGConfig.RERANKING_MODEL}")
    print(f"  Re-ranking Top K: {RAGConfig.RERANKING_TOP_K}")
    print(f"  Max Context Length: {RAGConfig.MAX_CONTEXT_LENGTH}")
    print()
    
    # 7. Check Directory Structure
    print("📁 Directory Structure:")
    print("-" * 80)
    directories = {
        "Project Root": PROJECT_ROOT,
        "Data Directory": DATA_DIR,
        "Raw Data": RAW_DATA_DIR,
        "Processed Data": PROCESSED_DATA_DIR,
        "Uploads": UPLOADS_DIR,
        "Outputs": OUTPUTS_DIR,
        "Cache": CACHE_DIR,
        "Logs": LOGS_DIR,
        "Vector DB": VECTOR_DB_DIR,
    }
    
    for name, path in directories.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}: {path}")
    print()
    
    # 8. Check File Sizes and Limits
    print("⚙️  System Limits:")
    print("-" * 80)
    print(f"  Max File Size: {DocumentConfig.MAX_FILE_SIZE_MB} MB")
    print(f"  Allowed File Types: {', '.join(DocumentConfig.ALLOWED_FILE_TYPES)}")
    print(f"  Embedding Batch Size: {EmbeddingConfig.BATCH_SIZE}")
    print(f"  Cache Max Size: {CacheConfig.MAX_SIZE_MB} MB")
    print()
    
    # 9. Security & Compliance
    print("🔒 Security & Compliance:")
    print("-" * 80)
    print(f"  HIPAA Mode: {SecurityConfig.HIPAA_MODE}")
    print(f"  PHI Encryption: {SecurityConfig.PHI_ENCRYPTION}")
    print(f"  Anonymize Patient Data: {SecurityConfig.ANONYMIZE_PATIENT_DATA}")
    print(f"  Data Retention: {SecurityConfig.DATA_RETENTION_DAYS} days")
    print()
    
    # 10. Agent Configuration
    print("🤖 Agent Configuration:")
    print("-" * 80)
    print(f"  Max Iterations: {AgentConfig.MAX_ITERATIONS}")
    print(f"  Timeout: {AgentConfig.TIMEOUT} seconds")
    print(f"  Retry Attempts: {AgentConfig.RETRY_ATTEMPTS}")
    print(f"  Verbose: {AgentConfig.VERBOSE}")
    print(f"  Use Orchestrator: {AgentConfig.USE_ORCHESTRATOR}")
    print()
    
    # 11. System Information
    print("ℹ️  System Information:")
    print("-" * 80)
    print(f"  App Name: {SystemConfig.APP_NAME}")
    print(f"  Version: {SystemConfig.VERSION}")
    print(f"  Environment: {SystemConfig.ENVIRONMENT}")
    print(f"  Debug Mode: {SystemConfig.DEBUG}")
    print(f"  Experimental Features: {SystemConfig.ENABLE_EXPERIMENTAL}")
    print()
    
    # 12. Final Status
    print("=" * 80)
    print("✅ CONFIGURATION CHECK COMPLETE")
    print("=" * 80)
    print("\n💡 Tips:")
    print("  - Ensure .env file exists with Azure OpenAI credentials")
    print("  - Run 'python ingestion/cms/fetch_cms_codes.py' to download CMS codes")
    print("  - Run 'streamlit run ui/app.py' to start the UI")
    print()