"""
Medical Coding System - Data Validators
Comprehensive validation utilities for medical codes, documents, and data structures
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import mimetypes


# ============================================================================
# 1. MEDICAL CODE FORMAT VALIDATORS
# ============================================================================

class MedicalCodeValidator:
    """Validators for medical code formats"""
    
    # ICD-10-CM: 3-7 characters (letter + 2 digits + optional decimal + up to 4 chars)
    ICD10CM_PATTERN = re.compile(r'^[A-TV-Z][0-9]{2}(\.[A-Z0-9]{1,4})?$', re.IGNORECASE)
    
    # ICD-10-PCS: Exactly 7 alphanumeric characters
    ICD10PCS_PATTERN = re.compile(r'^[0-9A-HJ-NP-Z]{7}$', re.IGNORECASE)
    
    # CPT: 5 digits, optionally followed by modifiers
    CPT_PATTERN = re.compile(r'^\d{5}$')
    
    # HCPCS: Letter followed by 4 digits
    HCPCS_PATTERN = re.compile(r'^[A-Z]\d{4}$', re.IGNORECASE)
    
    @classmethod
    def validate_icd10cm(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate ICD-10-CM code format
        
        Args:
            code: ICD-10-CM code to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code:
            return False, "Code cannot be empty"
        
        code = code.strip().upper()
        
        if not cls.ICD10CM_PATTERN.match(code):
            return False, f"Invalid ICD-10-CM format: {code}"
        
        # Additional validation rules
        if code[0] == 'U':
            # U codes are for special purposes
            if not code.startswith('U07') and not code.startswith('U09'):
                return False, f"Invalid U code category: {code}"
        
        return True, None
    
    @classmethod
    def validate_icd10pcs(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate ICD-10-PCS code format
        
        Args:
            code: ICD-10-PCS code to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code:
            return False, "Code cannot be empty"
        
        code = code.strip().upper()
        
        if len(code) != 7:
            return False, f"ICD-10-PCS must be exactly 7 characters: {code}"
        
        if not cls.ICD10PCS_PATTERN.match(code):
            return False, f"Invalid ICD-10-PCS format: {code}"
        
        # Check valid character set (excludes I, O to avoid confusion with 1, 0)
        invalid_chars = set('IO') & set(code)
        if invalid_chars:
            return False, f"ICD-10-PCS cannot contain I or O: {code}"
        
        return True, None
    
    @classmethod
    def validate_cpt(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate CPT code format
        
        Args:
            code: CPT code to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code:
            return False, "Code cannot be empty"
        
        code = code.strip()
        
        if not cls.CPT_PATTERN.match(code):
            return False, f"Invalid CPT format (must be 5 digits): {code}"
        
        # Check valid ranges
        code_num = int(code)
        valid_ranges = [
            (10, 99999),  # All CPT codes
        ]
        
        if not any(start <= code_num <= end for start, end in valid_ranges):
            return False, f"CPT code out of valid range: {code}"
        
        return True, None
    
    @classmethod
    def validate_hcpcs(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate HCPCS code format
        
        Args:
            code: HCPCS code to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code:
            return False, "Code cannot be empty"
        
        code = code.strip().upper()
        
        if not cls.HCPCS_PATTERN.match(code):
            return False, f"Invalid HCPCS format: {code}"
        
        return True, None
    
    @classmethod
    def validate_code(cls, code: str, code_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a medical code based on its type
        
        Args:
            code: Medical code to validate
            code_type: Type of code (icd10cm, icd10pcs, cpt, hcpcs)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        validators = {
            'icd10cm': cls.validate_icd10cm,
            'icd10pcs': cls.validate_icd10pcs,
            'cpt': cls.validate_cpt,
            'hcpcs': cls.validate_hcpcs,
        }
        
        code_type_lower = code_type.lower()
        if code_type_lower not in validators:
            return False, f"Unknown code type: {code_type}"
        
        return validators[code_type_lower](code)


# ============================================================================
# 2. DOCUMENT VALIDATORS
# ============================================================================

class DocumentValidator:
    """Validators for uploaded documents"""
    
    ALLOWED_EXTENSIONS = {'.pdf', '.csv', '.xlsx', '.xls', '.txt', '.docx'}
    MAX_FILE_SIZE_MB = 50
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file extension
        
        Args:
            filename: Name of the file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"
        
        file_path = Path(filename)
        extension = file_path.suffix.lower()
        
        if extension not in cls.ALLOWED_EXTENSIONS:
            allowed = ', '.join(cls.ALLOWED_EXTENSIONS)
            return False, f"File type '{extension}' not allowed. Allowed: {allowed}"
        
        return True, None
    
    @classmethod
    def validate_file_size(cls, file_size_bytes: int) -> Tuple[bool, Optional[str]]:
        """
        Validate file size
        
        Args:
            file_size_bytes: File size in bytes
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        max_size = cls.MAX_FILE_SIZE_MB * 1024 * 1024
        
        if file_size_bytes > max_size:
            actual_mb = file_size_bytes / (1024 * 1024)
            return False, f"File size {actual_mb:.1f}MB exceeds maximum {cls.MAX_FILE_SIZE_MB}MB"
        
        if file_size_bytes == 0:
            return False, "File is empty (0 bytes)"
        
        return True, None
    
    @classmethod
    def validate_file(cls, filepath: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate a file (extension, size, existence)
        
        Args:
            filepath: Path to the file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not filepath.exists():
            return False, f"File does not exist: {filepath}"
        
        if not filepath.is_file():
            return False, f"Path is not a file: {filepath}"
        
        # Validate extension
        is_valid, error = cls.validate_file_extension(str(filepath))
        if not is_valid:
            return is_valid, error
        
        # Validate size
        file_size = filepath.stat().st_size
        is_valid, error = cls.validate_file_size(file_size)
        if not is_valid:
            return is_valid, error
        
        return True, None
    
    @classmethod
    def get_mime_type(cls, filepath: Path) -> Optional[str]:
        """Get MIME type of a file"""
        mime_type, _ = mimetypes.guess_type(str(filepath))
        return mime_type


# ============================================================================
# 3. DATA STRUCTURE VALIDATORS
# ============================================================================

class DataValidator:
    """Validators for data structures and formats"""
    
    @staticmethod
    def validate_confidence_score(score: float) -> Tuple[bool, Optional[str]]:
        """
        Validate confidence score is between 0 and 1
        
        Args:
            score: Confidence score
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(score, (int, float)):
            return False, f"Confidence score must be numeric, got {type(score)}"
        
        if not 0.0 <= score <= 1.0:
            return False, f"Confidence score must be between 0 and 1, got {score}"
        
        return True, None
    
    @staticmethod
    def validate_year(year: int) -> Tuple[bool, Optional[str]]:
        """
        Validate year is reasonable
        
        Args:
            year: Year value
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        current_year = datetime.now().year
        
        if not isinstance(year, int):
            return False, f"Year must be an integer, got {type(year)}"
        
        if year < 2015:
            return False, f"Year {year} is too old (minimum 2015)"
        
        if year > current_year + 1:
            return False, f"Year {year} is in the future (maximum {current_year + 1})"
        
        return True, None
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, Optional[str]]:
        """
        Validate email format
        
        Args:
            email: Email address
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        if not email_pattern.match(email):
            return False, f"Invalid email format: {email}"
        
        return True, None
    
    @staticmethod
    def validate_non_empty_string(value: str, field_name: str = "Field") -> Tuple[bool, Optional[str]]:
        """
        Validate string is not empty
        
        Args:
            value: String value to validate
            field_name: Name of the field for error message
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, str):
            return False, f"{field_name} must be a string, got {type(value)}"
        
        if not value.strip():
            return False, f"{field_name} cannot be empty"
        
        return True, None
    
    @staticmethod
    def validate_dict_keys(data: dict, required_keys: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate dictionary has required keys
        
        Args:
            data: Dictionary to validate
            required_keys: List of required keys
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, f"Expected dictionary, got {type(data)}"
        
        missing_keys = set(required_keys) - set(data.keys())
        
        if missing_keys:
            return False, f"Missing required keys: {', '.join(missing_keys)}"
        
        return True, None


# ============================================================================
# 4. PHI/PII DETECTION
# ============================================================================

class PHIDetector:
    """Detect Protected Health Information (PHI) and Personally Identifiable Information (PII)"""
    
    # Patterns for detecting PHI/PII
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    MRN_PATTERN = re.compile(r'\bMRN[:\s]*\d+\b', re.IGNORECASE)
    DOB_PATTERN = re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b')
    ZIP_PATTERN = re.compile(r'\b\d{5}(-\d{4})?\b')
    
    @classmethod
    def detect_phi(cls, text: str) -> Dict[str, List[str]]:
        """
        Detect PHI in text
        
        Args:
            text: Text to scan for PHI
        
        Returns:
            Dictionary of PHI types and matches found
        """
        phi_found = {
            'ssn': cls.SSN_PATTERN.findall(text),
            'phone': cls.PHONE_PATTERN.findall(text),
            'email': cls.EMAIL_PATTERN.findall(text),
            'mrn': cls.MRN_PATTERN.findall(text),
            'dob': cls.DOB_PATTERN.findall(text),
            'zip': cls.ZIP_PATTERN.findall(text),
        }
        
        # Remove empty lists
        return {k: v for k, v in phi_found.items() if v}
    
    @classmethod
    def contains_phi(cls, text: str) -> bool:
        """
        Check if text contains any PHI
        
        Args:
            text: Text to check
        
        Returns:
            True if PHI detected
        """
        phi_found = cls.detect_phi(text)
        return len(phi_found) > 0
    
    @classmethod
    def redact_phi(cls, text: str) -> str:
        """
        Redact PHI from text
        
        Args:
            text: Text containing PHI
        
        Returns:
            Text with PHI redacted
        """
        redacted = text
        
        redacted = cls.SSN_PATTERN.sub('[REDACTED-SSN]', redacted)
        redacted = cls.PHONE_PATTERN.sub('[REDACTED-PHONE]', redacted)
        redacted = cls.EMAIL_PATTERN.sub('[REDACTED-EMAIL]', redacted)
        redacted = cls.MRN_PATTERN.sub('[REDACTED-MRN]', redacted)
        redacted = cls.DOB_PATTERN.sub('[REDACTED-DOB]', redacted)
        # Keep ZIP codes as they're often needed for billing
        
        return redacted


# ============================================================================
# 5. INPUT SANITIZATION
# ============================================================================

class InputSanitizer:
    """Sanitize user inputs"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal
        
        Args:
            filename: Original filename
        
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def sanitize_code(code: str) -> str:
        """
        Sanitize medical code input
        
        Args:
            code: Medical code
        
        Returns:
            Sanitized code
        """
        # Remove whitespace and convert to uppercase
        code = code.strip().upper()
        
        # Remove any non-alphanumeric characters except dots and dashes
        code = re.sub(r'[^A-Z0-9.\-]', '', code)
        
        return code
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """
        Sanitize general text input
        
        Args:
            text: Input text
            max_length: Maximum allowed length
        
        Returns:
            Sanitized text
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Trim whitespace
        text = text.strip()
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
        
        return text


# ============================================================================
# 6. MAIN BLOCK - Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate validators
    Usage: python utils/validators.py
    """
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - VALIDATORS TEST")
    print("=" * 80)
    print()
    
    # 1. Test ICD-10-CM validation
    print("🔍 Testing ICD-10-CM Code Validation:")
    print("-" * 80)
    
    icd10cm_test_cases = [
        ("E11.9", True, "Valid: Type 2 diabetes"),
        ("I10", True, "Valid: Essential hypertension"),
        ("K64.2", True, "Valid: Third degree hemorrhoids"),
        ("J45.909", True, "Valid: Unspecified asthma"),
        ("INVALID", False, "Invalid format"),
        ("12345", False, "Invalid: must start with letter"),
        ("E", False, "Invalid: too short"),
    ]
    
    for code, should_be_valid, description in icd10cm_test_cases:
        is_valid, error = MedicalCodeValidator.validate_icd10cm(code)
        status = "✓" if (is_valid == should_be_valid) else "✗"
        result = "VALID" if is_valid else f"INVALID ({error})"
        print(f"   {status} {code:12} -> {result:40} | {description}")
    print()
    
    # 2. Test ICD-10-PCS validation
    print("🔍 Testing ICD-10-PCS Code Validation:")
    print("-" * 80)
    
    icd10pcs_test_cases = [
        ("0DBJ8ZZ", True, "Valid: 7 characters"),
        ("0BH17EZ", True, "Valid: PCS code"),
        ("12345", False, "Invalid: too short"),
        ("0DBIO8Z", False, "Invalid: contains I or O"),
        ("ABCDEFG", True, "Valid: 7 alphanumeric"),
    ]
    
    for code, should_be_valid, description in icd10pcs_test_cases:
        is_valid, error = MedicalCodeValidator.validate_icd10pcs(code)
        status = "✓" if (is_valid == should_be_valid) else "✗"
        result = "VALID" if is_valid else f"INVALID ({error})"
        print(f"   {status} {code:12} -> {result:40} | {description}")
    print()
    
    # 3. Test CPT validation
    print("🔍 Testing CPT Code Validation:")
    print("-" * 80)
    
    cpt_test_cases = [
        ("99213", True, "Valid: Office visit"),
        ("45378", True, "Valid: Colonoscopy"),
        ("12345", True, "Valid: 5 digits"),
        ("ABC12", False, "Invalid: contains letters"),
        ("123", False, "Invalid: too short"),
    ]
    
    for code, should_be_valid, description in cpt_test_cases:
        is_valid, error = MedicalCodeValidator.validate_cpt(code)
        status = "✓" if (is_valid == should_be_valid) else "✗"
        result = "VALID" if is_valid else f"INVALID ({error})"
        print(f"   {status} {code:12} -> {result:40} | {description}")
    print()
    
    # 4. Test document validation
    print("📄 Testing Document Validation:")
    print("-" * 80)
    
    file_test_cases = [
        ("document.pdf", True, "Valid PDF"),
        ("data.csv", True, "Valid CSV"),
        ("report.xlsx", True, "Valid Excel"),
        ("note.txt", True, "Valid text"),
        ("file.exe", False, "Invalid: executable"),
        ("script.sh", False, "Invalid: shell script"),
    ]
    
    for filename, should_be_valid, description in file_test_cases:
        is_valid, error = DocumentValidator.validate_file_extension(filename)
        status = "✓" if (is_valid == should_be_valid) else "✗"
        result = "VALID" if is_valid else f"INVALID ({error})"
        print(f"   {status} {filename:20} -> {result:40} | {description}")
    print()
    
    # 5. Test file size validation
    print("💾 Testing File Size Validation:")
    print("-" * 80)
    
    size_test_cases = [
        (1024, True, "1 KB - Valid"),
        (1024 * 1024, True, "1 MB - Valid"),
        (10 * 1024 * 1024, True, "10 MB - Valid"),
        (50 * 1024 * 1024, True, "50 MB - Valid (at limit)"),
        (51 * 1024 * 1024, False, "51 MB - Too large"),
        (0, False, "0 bytes - Empty file"),
    ]
    
    for size_bytes, should_be_valid, description in size_test_cases:
        is_valid, error = DocumentValidator.validate_file_size(size_bytes)
        status = "✓" if (is_valid == should_be_valid) else "✗"
        result = "VALID" if is_valid else f"INVALID ({error})"
        print(f"   {status} {description:30} -> {result}")
    print()
    
    # 6. Test confidence score validation
    print("📊 Testing Confidence Score Validation:")
    print("-" * 80)
    
    score_test_cases = [
        (0.0, True, "Minimum score"),
        (0.5, True, "Mid-range score"),
        (1.0, True, "Maximum score"),
        (0.92, True, "Typical score"),
        (-0.1, False, "Below minimum"),
        (1.5, False, "Above maximum"),
    ]
    
    for score, should_be_valid, description in score_test_cases:
        is_valid, error = DataValidator.validate_confidence_score(score)
        status = "✓" if (is_valid == should_be_valid) else "✗"
        result = "VALID" if is_valid else f"INVALID ({error})"
        print(f"   {status} {score:5} -> {result:40} | {description}")
    print()
    
    # 7. Test year validation
    print("📅 Testing Year Validation:")
    print("-" * 80)
    
    current_year = datetime.now().year
    year_test_cases = [
        (2023, True, "Recent year"),
        (2024, True, "Current/recent year"),
        (2025, True, "Current/next year"),
        (2014, False, "Too old"),
        (current_year + 2, False, "Too far in future"),
    ]
    
    for year, should_be_valid, description in year_test_cases:
        is_valid, error = DataValidator.validate_year(year)
        status = "✓" if (is_valid == should_be_valid) else "✗"
        result = "VALID" if is_valid else f"INVALID ({error})"
        print(f"   {status} {year} -> {result:40} | {description}")
    print()
    
    # 8. Test PHI detection
    print("🔒 Testing PHI Detection:")
    print("-" * 80)
    
    phi_test_text = """
    Patient Name: John Doe
    SSN: 123-45-6789
    Phone: 555-123-4567
    Email: john.doe@example.com
    MRN: 987654
    DOB: 01/15/1980
    Address: 12345 Main St, Boston, MA 02101
    """
    
    phi_detected = PHIDetector.detect_phi(phi_test_text)
    print(f"   PHI Elements Detected: {len(phi_detected)}")
    for phi_type, matches in phi_detected.items():
        print(f"      {phi_type.upper()}: {matches}")
    print()
    
    # 9. Test PHI redaction
    print("🔒 Testing PHI Redaction:")
    print("-" * 80)
    
    redacted_text = PHIDetector.redact_phi(phi_test_text)
    print("   Original text (excerpt):")
    print(f"      {phi_test_text.split()[10:20]}")
    print("   Redacted text (excerpt):")
    print(f"      {redacted_text.split()[10:20]}")
    print()
    
    # 10. Test input sanitization
    print("🧹 Testing Input Sanitization:")
    print("-" * 80)
    
    sanitize_test_cases = [
        ("../../etc/passwd", "Dangerous path"),
        ("file<script>.pdf", "Script injection attempt"),
        ("normal_file.csv", "Clean filename"),
        ("  e11.9  ", "Code with whitespace"),
    ]
    
    for input_str, description in sanitize_test_cases:
        if "path" in description.lower() or "file" in description.lower():
            sanitized = InputSanitizer.sanitize_filename(input_str)
        elif "code" in description.lower():
            sanitized = InputSanitizer.sanitize_code(input_str)
        else:
            sanitized = InputSanitizer.sanitize_text(input_str)
        
        print(f"   ✓ {description}")
        print(f"      Input:  '{input_str}'")
        print(f"      Output: '{sanitized}'")
    print()
    
    # 11. Summary
    print("=" * 80)
    print("✅ VALIDATORS TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Usage Examples:")
    print()
    print("  # Validate ICD-10-CM code")
    print("  from utils.validators import MedicalCodeValidator")
    print("  is_valid, error = MedicalCodeValidator.validate_icd10cm('E11.9')")
    print()
    print("  # Validate uploaded file")
    print("  from utils.validators import DocumentValidator")
    print("  is_valid, error = DocumentValidator.validate_file(filepath)")
    print()
    print("  # Detect PHI in text")
    print("  from utils.validators import PHIDetector")
    print("  phi_found = PHIDetector.detect_phi(text)")
    print()
    print("  # Sanitize input")
    print("  from utils.validators import InputSanitizer")
    print("  clean_code = InputSanitizer.sanitize_code(user_input)")
    print()