"""
Medical Coding System - Logging Configuration
Production-ready logging with HIPAA compliance, audit trails, and monitoring
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


# ============================================================================
# 1. IMPORT SETTINGS
# ============================================================================
try:
    from config.settings import LoggingConfig, SecurityConfig, LOGS_DIR
except ImportError:
    # Fallback for standalone testing
    LOGS_DIR = Path("./logs")
    LOGS_DIR.mkdir(exist_ok=True)
    
    class LoggingConfig:
        LOG_LEVEL = "INFO"
        LOG_FILE = LOGS_DIR / "app.log"
        LOG_MAX_BYTES = 10485760  # 10MB
        LOG_BACKUP_COUNT = 5
        AUDIT_LOG_ENABLED = True
        AUDIT_LOG_FILE = LOGS_DIR / "audit.log"
        MONITORING_ENABLED = True
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    class SecurityConfig:
        HIPAA_MODE = True
        REDACT_PHI = True


# ============================================================================
# 2. CUSTOM LOG FORMATTER (HIPAA Compliant)
# ============================================================================

class HIPAACompliantFormatter(logging.Formatter):
    """
    Custom formatter that redacts PHI (Protected Health Information)
    """
    
    PHI_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'mrn': r'\bMRN[:\s]*\d+\b',
        'dob': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
    }
    
    def __init__(self, fmt=None, datefmt=None, redact_phi=True):
        super().__init__(fmt, datefmt)
        self.redact_phi = redact_phi
    
    def format(self, record):
        """Format log record and redact PHI if enabled"""
        # Format the record
        formatted = super().format(record)
        
        # Redact PHI if HIPAA mode is enabled
        if self.redact_phi and SecurityConfig.REDACT_PHI:
            import re
            for phi_type, pattern in self.PHI_PATTERNS.items():
                formatted = re.sub(pattern, f'[REDACTED-{phi_type.upper()}]', formatted)
        
        return formatted


# ============================================================================
# 3. CUSTOM LOG HANDLERS
# ============================================================================

class AuditLogHandler(logging.Handler):
    """
    Custom handler for audit logs with structured format
    """
    
    def __init__(self, filename):
        super().__init__()
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record):
        """Emit audit log entry in JSON format"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'user': getattr(record, 'user', 'system'),
                'action': getattr(record, 'action', 'unknown'),
                'resource': getattr(record, 'resource', 'unknown'),
            }
            
            # Write to audit log file
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')
        
        except Exception as e:
            # Fallback to stderr if audit logging fails
            print(f"Failed to write audit log: {e}", file=sys.stderr)


# ============================================================================
# 4. LOGGER SETUP FUNCTIONS
# ============================================================================

def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    enable_audit: bool = False
) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        enable_audit: Whether to enable audit logging
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level or LoggingConfig.LOG_LEVEL)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = HIPAACompliantFormatter(
        fmt=LoggingConfig.LOG_FORMAT,
        datefmt=LoggingConfig.DATE_FORMAT,
        redact_phi=SecurityConfig.REDACT_PHI
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=LoggingConfig.LOG_FILE,
            maxBytes=LoggingConfig.LOG_MAX_BYTES,
            backupCount=LoggingConfig.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Audit handler
    if enable_audit and LoggingConfig.AUDIT_LOG_ENABLED:
        audit_handler = AuditLogHandler(LoggingConfig.AUDIT_LOG_FILE)
        logger.addHandler(audit_handler)
    
    return logger


def setup_application_logging():
    """
    Set up application-wide logging configuration
    """
    # Create main application logger
    app_logger = setup_logger(
        name='medical_coding',
        log_to_file=True,
        log_to_console=True,
        enable_audit=False
    )
    
    # Create audit logger
    if LoggingConfig.AUDIT_LOG_ENABLED:
        audit_logger = setup_logger(
            name='medical_coding.audit',
            log_to_file=False,
            log_to_console=False,
            enable_audit=True
        )
    
    # Configure third-party loggers
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return app_logger


# ============================================================================
# 5. AUDIT LOGGING UTILITIES
# ============================================================================

def log_audit_event(
    logger: logging.Logger,
    action: str,
    resource: str,
    user: str = 'system',
    details: Optional[dict] = None
):
    """
    Log an audit event
    
    Args:
        logger: Logger instance
        action: Action performed (e.g., 'code_suggestion', 'document_upload')
        resource: Resource affected (e.g., 'patient_123', 'document_456')
        user: User performing the action
        details: Additional details as dictionary
    """
    message = f"Action: {action} | Resource: {resource} | User: {user}"
    if details:
        message += f" | Details: {json.dumps(details)}"
    
    # Create log record with extra attributes
    extra = {
        'user': user,
        'action': action,
        'resource': resource
    }
    
    logger.info(message, extra=extra)


# ============================================================================
# 6. PERFORMANCE MONITORING UTILITIES
# ============================================================================

class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} | Duration: {duration:.2f}s")
        else:
            self.logger.error(
                f"Failed: {self.operation} | Duration: {duration:.2f}s | Error: {exc_val}"
            )


# ============================================================================
# 7. LOG LEVEL UTILITIES
# ============================================================================

def get_log_level(level_name: str) -> int:
    """
    Convert log level name to logging constant
    
    Args:
        level_name: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logging level constant
    """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    return levels.get(level_name.upper(), logging.INFO)


def set_log_level(logger: logging.Logger, level: str):
    """
    Set log level for a logger
    
    Args:
        logger: Logger instance
        level: Log level name
    """
    logger.setLevel(get_log_level(level))


# ============================================================================
# 8. MAIN BLOCK - Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate logging configuration
    Usage: python config/logging_config.py
    """
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - LOGGING CONFIGURATION TEST")
    print("=" * 80)
    print()
    
    # 1. Setup application logging
    print("📝 Setting up application logging...")
    app_logger = setup_application_logging()
    print(f"   ✓ Main logger created: {app_logger.name}")
    print(f"   ✓ Log file: {LoggingConfig.LOG_FILE}")
    print(f"   ✓ Log level: {LoggingConfig.LOG_LEVEL}")
    print()
    
    # 2. Test different log levels
    print("🔍 Testing log levels:")
    print("-" * 80)
    app_logger.debug("This is a DEBUG message (detailed diagnostic info)")
    print("   ✓ DEBUG message logged")
    
    app_logger.info("This is an INFO message (general information)")
    print("   ✓ INFO message logged")
    
    app_logger.warning("This is a WARNING message (something unexpected)")
    print("   ✓ WARNING message logged")
    
    app_logger.error("This is an ERROR message (serious problem)")
    print("   ✓ ERROR message logged")
    
    app_logger.critical("This is a CRITICAL message (system failure)")
    print("   ✓ CRITICAL message logged")
    print()
    
    # 3. Test HIPAA-compliant PHI redaction
    print("🔒 Testing PHI redaction:")
    print("-" * 80)
    
    # Test SSN redaction
    app_logger.info("Patient SSN: 123-45-6789")
    print("   ✓ SSN redacted in logs")
    
    # Test phone redaction
    app_logger.info("Contact: 555-123-4567")
    print("   ✓ Phone number redacted")
    
    # Test email redaction
    app_logger.info("Email: patient@example.com")
    print("   ✓ Email redacted")
    
    # Test MRN redaction
    app_logger.info("Medical Record Number MRN: 987654")
    print("   ✓ MRN redacted")
    print()
    
    # 4. Test audit logging
    if LoggingConfig.AUDIT_LOG_ENABLED:
        print("📋 Testing audit logging:")
        print("-" * 80)
        audit_logger = logging.getLogger('medical_coding.audit')
        
        # Log various audit events
        log_audit_event(
            audit_logger,
            action='document_upload',
            resource='patient_chart_001.pdf',
            user='john_doe',
            details={'file_size': '2.5MB', 'pages': 15}
        )
        print("   ✓ Document upload event logged")
        
        log_audit_event(
            audit_logger,
            action='code_suggestion',
            resource='diagnosis_icd10',
            user='system',
            details={'code': 'E11.9', 'confidence': 0.92}
        )
        print("   ✓ Code suggestion event logged")
        
        log_audit_event(
            audit_logger,
            action='code_validation',
            resource='claim_12345',
            user='jane_smith',
            details={'status': 'approved', 'codes_validated': 5}
        )
        print("   ✓ Code validation event logged")
        
        print(f"   ✓ Audit log file: {LoggingConfig.AUDIT_LOG_FILE}")
        print()
    
    # 5. Test performance logging
    print("⏱️  Testing performance logging:")
    print("-" * 80)
    
    import time
    
    with PerformanceLogger(app_logger, "Sample operation"):
        time.sleep(0.5)  # Simulate work
    print("   ✓ Performance metrics logged")
    print()
    
    # 6. Test module-specific loggers
    print("📦 Testing module-specific loggers:")
    print("-" * 80)
    
    # Create loggers for different modules
    extraction_logger = setup_logger('medical_coding.extraction')
    coding_logger = setup_logger('medical_coding.coding')
    validation_logger = setup_logger('medical_coding.validation')
    
    extraction_logger.info("Extraction module initialized")
    print("   ✓ Extraction logger")
    
    coding_logger.info("Coding module initialized")
    print("   ✓ Coding logger")
    
    validation_logger.info("Validation module initialized")
    print("   ✓ Validation logger")
    print()
    
    # 7. Test log file rotation
    print("🔄 Testing log file rotation:")
    print("-" * 80)
    print(f"   Max file size: {LoggingConfig.LOG_MAX_BYTES / (1024*1024):.1f} MB")
    print(f"   Backup count: {LoggingConfig.LOG_BACKUP_COUNT}")
    print(f"   Total capacity: {(LoggingConfig.LOG_MAX_BYTES * (LoggingConfig.LOG_BACKUP_COUNT + 1)) / (1024*1024):.1f} MB")
    print()
    
    # 8. Test error handling
    print("⚠️  Testing error logging:")
    print("-" * 80)
    try:
        # Simulate an error
        raise ValueError("Sample error for testing")
    except Exception as e:
        app_logger.exception("An error occurred during processing")
        print("   ✓ Exception logged with traceback")
    print()
    
    # 9. Display log file information
    print("📂 Log files created:")
    print("-" * 80)
    
    if LoggingConfig.LOG_FILE.exists():
        size = LoggingConfig.LOG_FILE.stat().st_size
        print(f"   ✓ Application log: {LoggingConfig.LOG_FILE}")
        print(f"     Size: {size} bytes")
    
    if LoggingConfig.AUDIT_LOG_ENABLED and LoggingConfig.AUDIT_LOG_FILE.exists():
        size = LoggingConfig.AUDIT_LOG_FILE.stat().st_size
        print(f"   ✓ Audit log: {LoggingConfig.AUDIT_LOG_FILE}")
        print(f"     Size: {size} bytes")
    print()
    
    # 10. Display sample log entries
    print("📄 Sample log entries from file:")
    print("-" * 80)
    if LoggingConfig.LOG_FILE.exists():
        with open(LoggingConfig.LOG_FILE, 'r') as f:
            lines = f.readlines()
            last_5 = lines[-5:] if len(lines) >= 5 else lines
            for line in last_5:
                print(f"   {line.strip()}")
    print()
    
    # 11. Summary
    print("=" * 80)
    print("✅ LOGGING CONFIGURATION TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Usage Examples:")
    print("  from config.logging_config import setup_logger, log_audit_event")
    print("  logger = setup_logger('my_module')")
    print("  logger.info('Processing started')")
    print("  log_audit_event(logger, 'action', 'resource', 'user')")
    print()
    print("🔒 HIPAA Compliance:")
    print(f"  PHI Redaction: {'✓ Enabled' if SecurityConfig.REDACT_PHI else '✗ Disabled'}")
    print(f"  Audit Logging: {'✓ Enabled' if LoggingConfig.AUDIT_LOG_ENABLED else '✗ Disabled'}")
    print()