"""
Medical Coding System - Logger Utility
Convenient wrapper for application-wide logging with HIPAA compliance
"""
import logging
from typing import Optional
from functools import wraps
import time
from datetime import datetime


# ============================================================================
# 1. IMPORT LOGGING CONFIGURATION
# ============================================================================
try:
    from config.logging_config import (
        setup_logger,
        log_audit_event,
        PerformanceLogger,
        setup_application_logging
    )
except ImportError:
    # Fallback for testing
    def setup_logger(name, **kwargs):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def log_audit_event(logger, action, resource, user='system', details=None):
        logger.info(f"AUDIT: {action} | {resource} | {user}")
    
    class PerformanceLogger:
        def __init__(self, logger, operation):
            self.logger = logger
            self.operation = operation
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    def setup_application_logging():
        return setup_logger('medical_coding')


# ============================================================================
# 2. LOGGER FACTORY
# ============================================================================

class LoggerFactory:
    """Factory for creating and managing loggers"""
    
    _loggers = {}
    _app_logger_initialized = False
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger for a module
        
        Args:
            name: Logger name (typically __name__)
        
        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = setup_logger(name)
        return cls._loggers[name]
    
    @classmethod
    def get_app_logger(cls) -> logging.Logger:
        """Get the main application logger"""
        if not cls._app_logger_initialized:
            setup_application_logging()
            cls._app_logger_initialized = True
        return cls.get_logger('medical_coding')


# ============================================================================
# 3. CONVENIENCE FUNCTIONS
# ============================================================================

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (uses __name__ from caller if None)
    
    Returns:
        Logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
    
    return LoggerFactory.get_logger(name)


def audit_log(action: str, resource: str, user: str = 'system', details: Optional[dict] = None):
    """
    Convenience function for audit logging
    
    Args:
        action: Action performed
        resource: Resource affected
        user: User performing action
        details: Additional details
    
    Example:
        audit_log('code_suggestion', 'patient_123', user='doctor_smith', 
                  details={'code': 'E11.9', 'confidence': 0.92})
    """
    logger = get_logger('medical_coding.audit')
    log_audit_event(logger, action, resource, user, details)


# ============================================================================
# 4. DECORATORS FOR AUTOMATIC LOGGING
# ============================================================================

def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to automatically log function calls
    
    Args:
        logger: Logger instance (creates one if None)
    
    Example:
        @log_function_call()
        def process_document(doc_id):
            # Function logic
            pass
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_performance(logger: Optional[logging.Logger] = None, threshold_seconds: float = 1.0):
    """
    Decorator to log performance metrics
    
    Args:
        logger: Logger instance (creates one if None)
        threshold_seconds: Log warning if execution exceeds this threshold
    
    Example:
        @log_performance(threshold_seconds=2.0)
        def slow_operation():
            # Operation logic
            pass
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > threshold_seconds:
                    logger.warning(
                        f"{func_name} took {duration:.2f}s (threshold: {threshold_seconds}s)"
                    )
                else:
                    logger.info(f"{func_name} completed in {duration:.2f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func_name} failed after {duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


def log_errors(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Decorator to automatically log errors
    
    Args:
        logger: Logger instance (creates one if None)
        reraise: Whether to re-raise the exception after logging
    
    Example:
        @log_errors()
        def risky_operation():
            # Operation that might fail
            pass
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {e}")
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


def audit_operation(action: str, get_resource=None, get_user=None):
    """
    Decorator to automatically log audit events for operations
    
    Args:
        action: Action being performed
        get_resource: Function to extract resource from args/kwargs
        get_user: Function to extract user from args/kwargs
    
    Example:
        @audit_operation('document_upload', 
                        get_resource=lambda args, kwargs: kwargs.get('doc_id'))
        def upload_document(doc_id, content):
            # Upload logic
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract resource and user
            resource = get_resource(args, kwargs) if get_resource else func.__name__
            user = get_user(args, kwargs) if get_user else 'system'
            
            # Log audit event
            audit_log(action, resource, user)
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# 5. CONTEXT MANAGERS
# ============================================================================

class LogContext:
    """
    Context manager for logging operations with automatic timing
    
    Example:
        with LogContext(logger, "Processing document"):
            # Processing logic
            pass
    """
    
    def __init__(self, logger: logging.Logger, operation: str, level: str = 'INFO'):
        self.logger = logger
        self.operation = operation
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.level, 
                f"Completed: {self.operation} in {duration:.2f}s")
        else:
            self.logger.error(
                f"Failed: {self.operation} after {duration:.2f}s - {exc_val}"
            )
        
        return False  # Don't suppress exceptions


# ============================================================================
# 6. STRUCTURED LOGGING HELPERS
# ============================================================================

def log_structured(logger: logging.Logger, level: str, message: str, **kwargs):
    """
    Log with structured data
    
    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **kwargs: Additional structured data
    
    Example:
        log_structured(logger, 'INFO', 'Code suggested',
                      code='E11.9', confidence=0.92, duration=1.5)
    """
    import json
    structured_data = json.dumps(kwargs)
    full_message = f"{message} | Data: {structured_data}"
    logger.log(getattr(logging, level.upper()), full_message)


# ============================================================================
# 7. BATCH LOGGING
# ============================================================================

class BatchLogger:
    """
    Collect log messages and flush in batch
    Useful for performance-critical sections
    """
    
    def __init__(self, logger: logging.Logger, batch_size: int = 100):
        self.logger = logger
        self.batch_size = batch_size
        self.messages = []
    
    def add(self, level: str, message: str):
        """Add message to batch"""
        self.messages.append((level, message))
        
        if len(self.messages) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Flush all messages to logger"""
        for level, message in self.messages:
            self.logger.log(getattr(logging, level.upper()), message)
        self.messages.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.flush()


# ============================================================================
# 8. MAIN BLOCK - Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate logger utilities
    Usage: python utils/logger.py
    """
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - LOGGER UTILITIES TEST")
    print("=" * 80)
    print()
    
    # 1. Test basic logger creation
    print("📝 Testing logger creation:")
    print("-" * 80)
    logger = get_logger('test_module')
    print(f"   ✓ Logger created: {logger.name}")
    logger.info("Test message from logger utility")
    print("   ✓ Test message logged")
    print()
    
    # 2. Test audit logging
    print("📋 Testing audit logging:")
    print("-" * 80)
    audit_log(
        action='document_processing',
        resource='test_document.pdf',
        user='test_user',
        details={'pages': 5, 'size': '2.5MB'}
    )
    print("   ✓ Audit log entry created")
    print()
    
    # 3. Test function call decorator
    print("🔧 Testing function call decorator:")
    print("-" * 80)
    
    @log_function_call()
    def sample_function(x, y):
        """Sample function for testing"""
        return x + y
    
    result = sample_function(10, 20)
    print(f"   ✓ Function executed: result = {result}")
    print()
    
    # 4. Test performance decorator
    print("⏱️  Testing performance decorator:")
    print("-" * 80)
    
    @log_performance(threshold_seconds=0.5)
    def fast_operation():
        """Fast operation (below threshold)"""
        time.sleep(0.2)
        return "fast"
    
    @log_performance(threshold_seconds=0.5)
    def slow_operation():
        """Slow operation (above threshold)"""
        time.sleep(0.7)
        return "slow"
    
    fast_result = fast_operation()
    print(f"   ✓ Fast operation: {fast_result}")
    
    slow_result = slow_operation()
    print(f"   ✓ Slow operation: {slow_result} (warning logged)")
    print()
    
    # 5. Test error decorator
    print("⚠️  Testing error decorator:")
    print("-" * 80)
    
    @log_errors(reraise=False)
    def failing_function():
        """Function that raises an error"""
        raise ValueError("Intentional test error")
    
    result = failing_function()
    print(f"   ✓ Error logged, returned: {result}")
    print()
    
    # 6. Test audit operation decorator
    print("🔍 Testing audit operation decorator:")
    print("-" * 80)
    
    @audit_operation(
        action='code_validation',
        get_resource=lambda args, kwargs: kwargs.get('code_id', 'unknown')
    )
    def validate_code(code_id, code_value):
        """Sample validation function"""
        return f"Validated {code_id}: {code_value}"
    
    result = validate_code(code_id='ICD10_001', code_value='E11.9')
    print(f"   ✓ Audit logged for: {result}")
    print()
    
    # 7. Test LogContext
    print("📦 Testing LogContext:")
    print("-" * 80)
    
    test_logger = get_logger('context_test')
    with LogContext(test_logger, "Sample operation"):
        time.sleep(0.3)
        print("   ✓ Operation in progress...")
    print("   ✓ Context logged start and end")
    print()
    
    # 8. Test PerformanceLogger
    print("🚀 Testing PerformanceLogger:")
    print("-" * 80)
    
    perf_logger = get_logger('performance_test')
    with PerformanceLogger(perf_logger, "Performance test operation"):
        time.sleep(0.4)
        print("   ✓ Performance operation in progress...")
    print("   ✓ Performance metrics logged")
    print()
    
    # 9. Test structured logging
    print("📊 Testing structured logging:")
    print("-" * 80)
    
    struct_logger = get_logger('structured_test')
    log_structured(
        struct_logger,
        'INFO',
        'Code suggestion generated',
        code='E11.9',
        confidence=0.92,
        processing_time=1.5,
        model='gpt-4.1'
    )
    print("   ✓ Structured log entry created")
    print()
    
    # 10. Test batch logging
    print("📚 Testing batch logging:")
    print("-" * 80)
    
    batch_logger_instance = get_logger('batch_test')
    
    with BatchLogger(batch_logger_instance, batch_size=5) as batch:
        for i in range(12):
            batch.add('INFO', f"Batch message {i+1}")
        print(f"   ✓ Added 12 messages (flushed in batches of 5)")
    print("   ✓ All batch messages flushed")
    print()
    
    # 11. Test multiple module loggers
    print("🗂️  Testing multiple module loggers:")
    print("-" * 80)
    
    extraction_logger = get_logger('medical_coding.extraction')
    coding_logger = get_logger('medical_coding.coding')
    validation_logger = get_logger('medical_coding.validation')
    
    extraction_logger.info("Extraction module logger test")
    coding_logger.info("Coding module logger test")
    validation_logger.info("Validation module logger test")
    
    print("   ✓ Extraction logger")
    print("   ✓ Coding logger")
    print("   ✓ Validation logger")
    print()
    
    # 12. Test complex scenario
    print("🎯 Testing complex scenario (chained decorators):")
    print("-" * 80)
    
    @log_function_call()
    @log_performance(threshold_seconds=1.0)
    @log_errors()
    def complex_operation(value):
        """Complex operation with multiple decorators"""
        time.sleep(0.3)
        if value < 0:
            raise ValueError("Value must be positive")
        return value * 2
    
    try:
        result = complex_operation(5)
        print(f"   ✓ Complex operation succeeded: {result}")
    except Exception as e:
        print(f"   ✓ Complex operation error handled: {e}")
    print()
    
    # 13. Summary
    print("=" * 80)
    print("✅ LOGGER UTILITIES TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Usage Examples:")
    print()
    print("  # Basic logging")
    print("  from utils.logger import get_logger")
    print("  logger = get_logger(__name__)")
    print("  logger.info('Processing started')")
    print()
    print("  # Audit logging")
    print("  from utils.logger import audit_log")
    print("  audit_log('document_upload', 'doc_123', user='john')")
    print()
    print("  # Performance logging")
    print("  @log_performance(threshold_seconds=2.0)")
    print("  def my_function():")
    print("      pass")
    print()
    print("  # Error handling")
    print("  @log_errors()")
    print("  def risky_function():")
    print("      pass")
    print()