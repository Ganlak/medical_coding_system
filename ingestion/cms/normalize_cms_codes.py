"""
Medical Coding System - CMS Code Normalizer
Clean, normalize, and validate CMS codes for embedding
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm


# ============================================================================
# 1. IMPORT CONFIGURATION
# ============================================================================
try:
    from config.settings import CMSConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR
    from utils.logger import get_logger
    from utils.validators import MedicalCodeValidator
except ImportError:
    # Fallback for standalone usage
    RAW_DATA_DIR = Path("./data/raw")
    PROCESSED_DATA_DIR = Path("./data/processed")
    
    class CMSConfig:
        CURRENT_YEAR = 2025
        CODE_TYPES = ["icd10cm", "icd10pcs", "cpt"]
    
    class MedicalCodeValidator:
        @classmethod
        def validate_icd10cm(cls, code): return True, None
        @classmethod
        def validate_icd10pcs(cls, code): return True, None
        @classmethod
        def validate_cpt(cls, code): return True, None
    
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    def get_logger(name):
        return MockLogger()


logger = get_logger(__name__)


# ============================================================================
# 2. ICD-10-CM NORMALIZER
# ============================================================================

class ICD10CMNormalizer:
    """Normalize ICD-10-CM codes"""
    
    @staticmethod
    def find_code_file(extract_dir: Path) -> Optional[Path]:
        """Find the main code description file"""
        # Common file patterns
        patterns = [
            "**/icd10cm_codes_*.txt",
            "**/icd10cm_order_*.txt",
            "**/*order*.txt",
            "**/*codes*.txt",
            "**/*.txt"
        ]
        
        for pattern in patterns:
            files = list(extract_dir.glob(pattern))
            if files:
                # Return largest file (usually the main code file)
                return max(files, key=lambda f: f.stat().st_size)
        
        return None
    
    @classmethod
    def parse_code_file(cls, file_path: Path) -> List[Dict]:
        """
        Parse ICD-10-CM code file
        
        Expected format (tab or space-delimited):
        A00.0    Cholera due to Vibrio cholerae 01, biovar cholerae
        """
        codes = []
        
        logger.info(f"Parsing ICD-10-CM file: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                # Try tab-delimited first
                parts = line.split('\t')
                if len(parts) < 2:
                    # Try space-delimited
                    parts = line.split(None, 1)
                
                if len(parts) >= 2:
                    code = parts[0].strip().upper()
                    description = parts[1].strip()
                    
                    # Validate code format
                    is_valid, error = MedicalCodeValidator.validate_icd10cm(code)
                    
                    if is_valid:
                        codes.append({
                            'code': code,
                            'description': description,
                            'code_type': 'icd10cm',
                            'category': code[:3],  # First 3 chars
                            'is_billable': '.' in code,  # Codes with decimals are typically billable
                        })
                    else:
                        logger.warning(f"Line {line_num}: Invalid code format: {code}")
        
        logger.info(f"  Parsed {len(codes)} ICD-10-CM codes")
        return codes
    
    @classmethod
    def normalize(cls, extract_dir: Path) -> pd.DataFrame:
        """
        Normalize ICD-10-CM codes from extracted directory
        
        Args:
            extract_dir: Directory containing extracted files
        
        Returns:
            DataFrame with normalized codes
        """
        code_file = cls.find_code_file(extract_dir)
        
        if not code_file:
            logger.error(f"No code file found in {extract_dir}")
            return pd.DataFrame()
        
        codes = cls.parse_code_file(code_file)
        
        if not codes:
            return pd.DataFrame()
        
        df = pd.DataFrame(codes)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['code'], keep='first')
        
        # Sort by code
        df = df.sort_values('code').reset_index(drop=True)
        
        logger.info(f"  ✓ Normalized {len(df)} unique ICD-10-CM codes")
        
        return df


# ============================================================================
# 3. ICD-10-PCS NORMALIZER
# ============================================================================

class ICD10PCSNormalizer:
    """Normalize ICD-10-PCS codes"""
    
    @staticmethod
    def find_code_file(extract_dir: Path) -> Optional[Path]:
        """Find the main PCS code file"""
        patterns = [
            "**/icd10pcs_codes_*.txt",
            "**/icd10pcs_order_*.txt",
            "**/*order*.txt",
            "**/*codes*.txt",
            "**/*.txt"
        ]
        
        for pattern in patterns:
            files = list(extract_dir.glob(pattern))
            if files:
                return max(files, key=lambda f: f.stat().st_size)
        
        return None
    
    @classmethod
    def parse_code_file(cls, file_path: Path) -> List[Dict]:
        """Parse ICD-10-PCS code file"""
        codes = []
        
        logger.info(f"Parsing ICD-10-PCS file: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                # PCS codes are exactly 7 characters
                parts = line.split(None, 1)
                
                if len(parts) >= 2:
                    code = parts[0].strip().upper()
                    description = parts[1].strip()
                    
                    # Validate
                    is_valid, error = MedicalCodeValidator.validate_icd10pcs(code)
                    
                    if is_valid:
                        codes.append({
                            'code': code,
                            'description': description,
                            'code_type': 'icd10pcs',
                            'section': code[0],  # First character
                            'body_system': code[1] if len(code) > 1 else '',
                        })
                    else:
                        logger.warning(f"Line {line_num}: Invalid PCS code: {code}")
        
        logger.info(f"  Parsed {len(codes)} ICD-10-PCS codes")
        return codes
    
    @classmethod
    def normalize(cls, extract_dir: Path) -> pd.DataFrame:
        """Normalize ICD-10-PCS codes"""
        code_file = cls.find_code_file(extract_dir)
        
        if not code_file:
            logger.error(f"No code file found in {extract_dir}")
            return pd.DataFrame()
        
        codes = cls.parse_code_file(code_file)
        
        if not codes:
            return pd.DataFrame()
        
        df = pd.DataFrame(codes)
        df = df.drop_duplicates(subset=['code'], keep='first')
        df = df.sort_values('code').reset_index(drop=True)
        
        logger.info(f"  ✓ Normalized {len(df)} unique ICD-10-PCS codes")
        
        return df


# ============================================================================
# 4. CPT/HCPCS NORMALIZER
# ============================================================================

# ============================================================================
# 4. CPT/HCPCS NORMALIZER (UPDATED)
# ============================================================================

class CPTNormalizer:
    """Normalize CPT/HCPCS codes"""
    
    @staticmethod
    def find_code_file(extract_dir: Path) -> Optional[Path]:
        """Find CPT/HCPCS code file"""
        patterns = [
            "**/*.xlsx",
            "**/*.csv",
            "**/*.txt",
        ]
        
        for pattern in patterns:
            files = list(extract_dir.glob(pattern))
            if files:
                # Prefer Excel files
                excel_files = [f for f in files if f.suffix in ['.xlsx', '.xls']]
                if excel_files:
                    return excel_files[0]
                return files[0]
        
        return None
    
    @classmethod
    def find_data_start_row(cls, file_path: Path) -> Tuple[int, pd.DataFrame]:
        """
        Find where actual data starts in Excel file
        
        Returns:
            Tuple of (header_row_index, dataframe)
        """
        # Try different header row positions
        for skip_rows in range(0, 20):
            try:
                df = pd.read_excel(file_path, skiprows=skip_rows, nrows=10, dtype=str)
                
                # Check if this looks like a data table
                # Look for columns that might contain codes
                col_names = [str(col).lower() for col in df.columns]
                
                # Skip if only one column or all unnamed
                if len(df.columns) < 2:
                    continue
                
                if all('unnamed' in col for col in col_names):
                    continue
                
                # Check if first row looks like data (not more headers)
                if not df.empty:
                    first_val = str(df.iloc[0, 0]).strip()
                    # Look for code-like values (5 characters, alphanumeric)
                    if len(first_val) == 5 and (first_val.isdigit() or 
                                                (first_val[0].isalpha() and first_val[1:].isdigit())):
                        logger.info(f"  Found data starting at row {skip_rows + 1}")
                        # Read the full dataframe
                        full_df = pd.read_excel(file_path, skiprows=skip_rows, dtype=str)
                        return skip_rows, full_df
            
            except Exception as e:
                continue
        
        # Fallback: try reading without skipping
        logger.warning("  Could not auto-detect data start, trying full file")
        try:
            df = pd.read_excel(file_path, dtype=str)
            return 0, df
        except:
            return -1, pd.DataFrame()
    
    @classmethod
    def parse_excel_file(cls, file_path: Path) -> List[Dict]:
        """Parse CPT codes from Excel file with flexible column detection"""
        codes = []
        
        logger.info(f"Parsing CPT file: {file_path.name}")
        
        try:
            # Find where data starts
            skip_rows, df = cls.find_data_start_row(file_path)
            
            if df.empty:
                logger.error("Could not read Excel file")
                return []
            
            logger.info(f"  Columns found: {list(df.columns)}")
            logger.info(f"  Total rows: {len(df)}")
            
            # Flexible column matching
            code_col = None
            desc_col = None
            
            # Strategy 1: Look for column names
            for col in df.columns:
                col_lower = str(col).lower().strip()
                
                # Code column patterns
                if not code_col:
                    if any(pattern in col_lower for pattern in 
                          ['code', 'cpt', 'hcpcs', 'procedure code']):
                        code_col = col
                        logger.info(f"  Using code column: {col}")
                
                # Description column patterns
                if not desc_col:
                    if any(pattern in col_lower for pattern in 
                          ['description', 'descriptor', 'long', 'short', 'procedure']):
                        desc_col = col
                        logger.info(f"  Using description column: {col}")
            
            # Strategy 2: Use first columns if not found
            if not code_col and len(df.columns) >= 1:
                code_col = df.columns[0]
                logger.warning(f"  Defaulting to first column for codes: {code_col}")
            
            if not desc_col and len(df.columns) >= 2:
                desc_col = df.columns[1]
                logger.warning(f"  Defaulting to second column for descriptions: {desc_col}")
            
            # Strategy 3: Look at data to identify columns
            if not code_col:
                # Find column with 5-character codes
                for col in df.columns:
                    sample = df[col].dropna().head(10)
                    if sample.apply(lambda x: len(str(x).strip()) == 5).mean() > 0.5:
                        code_col = col
                        logger.info(f"  Detected code column from data: {col}")
                        break
            
            if not code_col:
                logger.error(f"Could not identify code column. Columns: {list(df.columns)}")
                logger.error(f"First few rows:\n{df.head()}")
                return []
            
            # Parse codes
            parsed_count = 0
            skipped_count = 0
            
            for idx, row in df.iterrows():
                try:
                    code = str(row[code_col]).strip()
                    description = str(row[desc_col]).strip() if desc_col else ''
                    
                    # Skip invalid entries
                    if not code or code.lower() in ['code', 'cpt', 'hcpcs', 'nan', 'none']:
                        skipped_count += 1
                        continue
                    
                    # Clean code
                    code = re.sub(r'[^0-9A-Z]', '', code.upper())
                    
                    if not code or len(code) != 5:
                        skipped_count += 1
                        continue
                    
                    # Validate and categorize
                    if code.isdigit():
                        is_valid, _ = MedicalCodeValidator.validate_cpt(code)
                        code_type = 'cpt'
                    elif code[0].isalpha() and code[1:].isdigit():
                        code_type = 'hcpcs'
                        is_valid = True
                    else:
                        skipped_count += 1
                        continue
                    
                    if is_valid:
                        codes.append({
                            'code': code,
                            'description': description,
                            'code_type': code_type,
                        })
                        parsed_count += 1
                
                except Exception as e:
                    skipped_count += 1
                    continue
            
            logger.info(f"  Parsed {parsed_count} codes, skipped {skipped_count} rows")
        
        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
        logger.info(f"  Total CPT/HCPCS codes extracted: {len(codes)}")
        return codes
    
    @classmethod
    def normalize(cls, extract_dir: Path) -> pd.DataFrame:
        """Normalize CPT/HCPCS codes"""
        code_file = cls.find_code_file(extract_dir)
        
        if not code_file:
            logger.error(f"No code file found in {extract_dir}")
            return pd.DataFrame()
        
        codes = cls.parse_excel_file(code_file)
        
        if not codes:
            logger.warning("No codes parsed, returning empty DataFrame")
            return pd.DataFrame()
        
        df = pd.DataFrame(codes)
        df = df.drop_duplicates(subset=['code'], keep='first')
        df = df.sort_values('code').reset_index(drop=True)
        
        logger.info(f"  ✓ Normalized {len(df)} unique CPT/HCPCS codes")
        
        return df

# ============================================================================
# 5. MAIN NORMALIZATION ORCHESTRATOR
# ============================================================================

def normalize_codes(
    year: int,
    code_type: str,
    raw_dir: Path,
    output_dir: Path
) -> Optional[Path]:
    """
    Normalize codes for a specific year and type
    
    Args:
        year: Fiscal year
        code_type: Code type (icd10cm, icd10pcs, cpt)
        raw_dir: Raw data directory
        output_dir: Output directory for normalized CSV
    
    Returns:
        Path to normalized CSV file, or None if failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Normalizing {code_type.upper()} codes for {year}")
    logger.info(f"{'='*80}")
    
    # Find extracted directory
    extract_patterns = {
        'icd10cm': [
            raw_dir / 'icd10cm' / f'icd10cm_code_descriptions_{year}',
            raw_dir / 'icd10cm' / f'icd10cm_code_tables_{year}',
        ],
        'icd10pcs': [
            raw_dir / 'icd10pcs' / f'icd10pcs_order_{year}',
            raw_dir / 'icd10pcs' / f'icd10pcs_codes_{year}',
        ],
        'cpt': [
            raw_dir / 'cpt' / f'cpt_hcpcs_codes_{year}',
        ]
    }
    
    extract_dir = None
    for pattern_dir in extract_patterns.get(code_type, []):
        if pattern_dir.exists():
            extract_dir = pattern_dir
            break
    
    if not extract_dir:
        logger.error(f"No extracted directory found for {code_type} {year}")
        return None
    
    logger.info(f"Source: {extract_dir}")
    
    # Normalize based on type
    normalizers = {
        'icd10cm': ICD10CMNormalizer,
        'icd10pcs': ICD10PCSNormalizer,
        'cpt': CPTNormalizer,
    }
    
    normalizer = normalizers.get(code_type)
    if not normalizer:
        logger.error(f"Unknown code type: {code_type}")
        return None
    
    df = normalizer.normalize(extract_dir)
    
    if df.empty:
        logger.error("Normalization produced no results")
        return None
    
    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{code_type}_{year}.csv"
    
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"✓ Saved: {output_file}")
    logger.info(f"  Total codes: {len(df)}")
    
    return output_file


# ============================================================================
# 6. COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Normalize CMS medical codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize all 2025 codes
  python -m ingestion.cms.normalize_cms_codes --year 2025 --codes all
  
  # Normalize only ICD-10-CM
  python -m ingestion.cms.normalize_cms_codes --year 2025 --codes icd10cm
  
  # Normalize multiple years
  python -m ingestion.cms.normalize_cms_codes --year 2025 --codes all
  python -m ingestion.cms.normalize_cms_codes --year 2024 --codes all
        """
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
        help='Code types to normalize'
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        default=RAW_DATA_DIR,
        help=f'Input directory (default: {RAW_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f'Output directory (default: {PROCESSED_DATA_DIR})'
    )
    
    args = parser.parse_args()
    
    # Expand 'all'
    if 'all' in args.codes:
        code_types = ['icd10cm', 'icd10pcs', 'cpt']
    else:
        code_types = args.codes
    
    # Display configuration
    print("=" * 80)
    print("CMS CODE NORMALIZER")
    print("=" * 80)
    print(f"Year: {args.year}")
    print(f"Code Types: {', '.join(code_types)}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 80)
    print()
    
    # Normalize each type
    results = {}
    for code_type in code_types:
        output_file = normalize_codes(
            year=args.year,
            code_type=code_type,
            raw_dir=args.input,
            output_dir=args.output
        )
        results[code_type] = output_file
    
    # Summary
    print()
    print("=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)
    
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"✓ Successful: {success_count}/{len(code_types)}")
    
    for code_type, output_file in results.items():
        if output_file:
            print(f"  ✓ {code_type}: {output_file}")
        else:
            print(f"  ✗ {code_type}: Failed")
    
    print("=" * 80)
    
    return 0 if success_count == len(code_types) else 1


# ============================================================================
# 7. MAIN BLOCK
# ============================================================================

if __name__ == "__main__":
    """
    Test and run CMS code normalizer
    Usage: python ingestion/cms/normalize_cms_codes.py --year 2025 --codes all
    """
    sys.exit(main())