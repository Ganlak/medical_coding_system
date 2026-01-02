"""
Medical Coding System - Year Manager
Manage multiple year versions of CMS codes for year-wise retrieval and comparison
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import json


# ============================================================================
# 1. IMPORT CONFIGURATION
# ============================================================================
try:
    from config.settings import CMSConfig, PROCESSED_DATA_DIR, RAW_DATA_DIR
    from utils.logger import get_logger
except ImportError:
    # Fallback for standalone usage
    PROCESSED_DATA_DIR = Path("./data/processed")
    RAW_DATA_DIR = Path("./data/raw")
    
    class CMSConfig:
        CURRENT_YEAR = 2025
        PREVIOUS_YEARS = [2024, 2023, 2022]
        CODE_TYPES = ["icd10cm", "icd10pcs", "cpt"]
        
        @classmethod
        def get_all_years(cls):
            return [cls.CURRENT_YEAR] + cls.PREVIOUS_YEARS
    
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    def get_logger(name):
        return MockLogger()


logger = get_logger(__name__)


# ============================================================================
# 2. YEAR METADATA MANAGER
# ============================================================================

class YearMetadata:
    """Metadata for a specific year's code set"""
    
    def __init__(self, year: int, code_type: str):
        self.year = year
        self.code_type = code_type
        self.file_path = PROCESSED_DATA_DIR / f"{code_type}_{year}.csv"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load or generate metadata"""
        if not self.file_path.exists():
            return {
                'year': self.year,
                'code_type': self.code_type,
                'exists': False,
                'total_codes': 0,
                'file_size': 0,
                'last_updated': None,
            }
        
        try:
            df = pd.read_csv(self.file_path)
            stat = self.file_path.stat()
            
            return {
                'year': self.year,
                'code_type': self.code_type,
                'exists': True,
                'total_codes': len(df),
                'file_size': stat.st_size,
                'file_path': str(self.file_path),
                'last_updated': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'columns': list(df.columns),
            }
        except Exception as e:
            logger.error(f"Error loading metadata for {self.code_type} {self.year}: {e}")
            return {'year': self.year, 'code_type': self.code_type, 'exists': False}
    
    def is_available(self) -> bool:
        """Check if codes are available"""
        return self.metadata.get('exists', False)
    
    def get_total_codes(self) -> int:
        """Get total number of codes"""
        return self.metadata.get('total_codes', 0)
    
    def get_file_path(self) -> Optional[Path]:
        """Get file path if available"""
        return self.file_path if self.is_available() else None


# ============================================================================
# 3. YEAR MANAGER
# ============================================================================

class YearManager:
    """Manage multiple years of CMS codes"""
    
    def __init__(self):
        self.current_year = CMSConfig.CURRENT_YEAR
        self.all_years = CMSConfig.get_all_years()
        self.code_types = CMSConfig.CODE_TYPES
    
    def get_available_years(self, code_type: str) -> List[int]:
        """
        Get list of years that have data available for a code type
        
        Args:
            code_type: Code type (icd10cm, icd10pcs, cpt)
        
        Returns:
            List of available years
        """
        available = []
        
        for year in self.all_years:
            metadata = YearMetadata(year, code_type)
            if metadata.is_available():
                available.append(year)
        
        return sorted(available, reverse=True)
    
    def get_code_file(self, code_type: str, year: Optional[int] = None) -> Optional[Path]:
        """
        Get code file for specific type and year
        
        Args:
            code_type: Code type
            year: Year (uses current year if None)
        
        Returns:
            Path to code file, or None if not available
        """
        if year is None:
            year = self.current_year
        
        metadata = YearMetadata(year, code_type)
        return metadata.get_file_path()
    
    def load_codes(self, code_type: str, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load codes for specific type and year
        
        Args:
            code_type: Code type
            year: Year (uses current year if None)
        
        Returns:
            DataFrame with codes
        """
        file_path = self.get_code_file(code_type, year)
        
        if not file_path:
            logger.warning(f"No codes available for {code_type} year {year or self.current_year}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} {code_type} codes from {year or self.current_year}")
            return df
        except Exception as e:
            logger.error(f"Error loading codes: {e}")
            return pd.DataFrame()
    
    def get_all_metadata(self) -> Dict[str, Dict[int, Dict]]:
        """
        Get metadata for all code types and years
        
        Returns:
            Nested dictionary: {code_type: {year: metadata}}
        """
        all_metadata = {}
        
        for code_type in self.code_types:
            all_metadata[code_type] = {}
            
            for year in self.all_years:
                metadata = YearMetadata(year, code_type)
                all_metadata[code_type][year] = metadata.metadata
        
        return all_metadata
    
    def compare_years(self, code_type: str, year1: int, year2: int) -> Dict:
        """
        Compare codes between two years
        
        Args:
            code_type: Code type
            year1: First year
            year2: Second year
        
        Returns:
            Dictionary with comparison results
        """
        df1 = self.load_codes(code_type, year1)
        df2 = self.load_codes(code_type, year2)
        
        if df1.empty or df2.empty:
            return {'error': 'One or both years not available'}
        
        codes1 = set(df1['code'])
        codes2 = set(df2['code'])
        
        added = codes2 - codes1
        removed = codes1 - codes2
        common = codes1 & codes2
        
        return {
            'code_type': code_type,
            'year1': year1,
            'year2': year2,
            'year1_total': len(codes1),
            'year2_total': len(codes2),
            'added_in_year2': len(added),
            'removed_from_year1': len(removed),
            'common_codes': len(common),
            'added_codes': sorted(list(added))[:100],  # First 100
            'removed_codes': sorted(list(removed))[:100],  # First 100
        }
    
    def get_code_history(self, code: str, code_type: str) -> List[Dict]:
        """
        Get history of a specific code across years
        
        Args:
            code: Medical code
            code_type: Code type
        
        Returns:
            List of year information where code exists
        """
        history = []
        
        for year in sorted(self.all_years, reverse=True):
            df = self.load_codes(code_type, year)
            
            if not df.empty and code in df['code'].values:
                code_info = df[df['code'] == code].iloc[0].to_dict()
                code_info['year'] = year
                history.append(code_info)
        
        return history
    
    def validate_all_years(self) -> Dict[str, Dict[int, bool]]:
        """
        Validate all years have required data
        
        Returns:
            Dictionary: {code_type: {year: is_valid}}
        """
        validation = {}
        
        for code_type in self.code_types:
            validation[code_type] = {}
            
            for year in self.all_years:
                metadata = YearMetadata(year, code_type)
                validation[code_type][year] = metadata.is_available()
        
        return validation
    
    def get_summary_report(self) -> str:
        """
        Generate summary report of all available years
        
        Returns:
            Formatted string report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CMS CODES - YEAR-WISE SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        all_metadata = self.get_all_metadata()
        
        for code_type in self.code_types:
            report_lines.append(f"\n{code_type.upper()}:")
            report_lines.append("-" * 80)
            
            for year in sorted(self.all_years, reverse=True):
                meta = all_metadata[code_type][year]
                
                if meta.get('exists'):
                    size_mb = meta['file_size'] / (1024 * 1024)
                    status = "✓"
                    info = f"{meta['total_codes']:,} codes | {size_mb:.2f} MB"
                else:
                    status = "✗"
                    info = "Not available"
                
                current = " (CURRENT)" if year == self.current_year else ""
                report_lines.append(f"  {status} {year}{current}: {info}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


# ============================================================================
# 4. COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Manage year-wise CMS medical codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary of all years
  python -m ingestion.cms.year_manager --summary
  
  # List available years for ICD-10-CM
  python -m ingestion.cms.year_manager --available icd10cm
  
  # Compare two years
  python -m ingestion.cms.year_manager --compare icd10cm --years 2025 2024
  
  # Get code history
  python -m ingestion.cms.year_manager --history E11.9 --type icd10cm
  
  # Validate all years
  python -m ingestion.cms.year_manager --validate
        """
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary of all available years'
    )
    
    parser.add_argument(
        '--available',
        type=str,
        choices=['icd10cm', 'icd10pcs', 'cpt'],
        help='Show available years for code type'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        choices=['icd10cm', 'icd10pcs', 'cpt'],
        help='Compare codes between two years'
    )
    
    parser.add_argument(
        '--years',
        nargs=2,
        type=int,
        help='Two years to compare (e.g., 2025 2024)'
    )
    
    parser.add_argument(
        '--history',
        type=str,
        help='Get history of a specific code'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['icd10cm', 'icd10pcs', 'cpt'],
        help='Code type for history lookup'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate all years have required data'
    )
    
    args = parser.parse_args()
    
    manager = YearManager()
    
    # Summary
    if args.summary:
        print(manager.get_summary_report())
        return 0
    
    # Available years
    if args.available:
        years = manager.get_available_years(args.available)
        print(f"\nAvailable years for {args.available.upper()}:")
        for year in years:
            current = " (CURRENT)" if year == manager.current_year else ""
            print(f"  • {year}{current}")
        print()
        return 0
    
    # Compare years
    if args.compare:
        if not args.years or len(args.years) != 2:
            print("Error: --compare requires --years with two values")
            return 1
        
        year1, year2 = args.years
        comparison = manager.compare_years(args.compare, year1, year2)
        
        if 'error' in comparison:
            print(f"Error: {comparison['error']}")
            return 1
        
        print(f"\n{'='*80}")
        print(f"COMPARISON: {args.compare.upper()} {year1} vs {year2}")
        print(f"{'='*80}")
        print(f"\n{year1} total codes: {comparison['year1_total']:,}")
        print(f"{year2} total codes: {comparison['year2_total']:,}")
        print(f"\nAdded in {year2}: {comparison['added_in_year2']:,}")
        print(f"Removed from {year1}: {comparison['removed_from_year1']:,}")
        print(f"Common codes: {comparison['common_codes']:,}")
        
        if comparison['added_codes']:
            print(f"\nSample added codes (first 20):")
            for code in comparison['added_codes'][:20]:
                print(f"  + {code}")
        
        if comparison['removed_codes']:
            print(f"\nSample removed codes (first 20):")
            for code in comparison['removed_codes'][:20]:
                print(f"  - {code}")
        
        print()
        return 0
    
    # Code history
    if args.history:
        if not args.type:
            print("Error: --history requires --type")
            return 1
        
        history = manager.get_code_history(args.history, args.type)
        
        if not history:
            print(f"\nCode {args.history} not found in any year for {args.type}")
            return 1
        
        print(f"\n{'='*80}")
        print(f"CODE HISTORY: {args.history} ({args.type.upper()})")
        print(f"{'='*80}\n")
        
        for entry in history:
            year = entry['year']
            desc = entry.get('description', 'N/A')
            print(f"Year {year}:")
            print(f"  Description: {desc}")
            print()
        
        return 0
    
    # Validate
    if args.validate:
        validation = manager.validate_all_years()
        
        print(f"\n{'='*80}")
        print("VALIDATION REPORT")
        print(f"{'='*80}\n")
        
        all_valid = True
        
        for code_type in manager.code_types:
            print(f"{code_type.upper()}:")
            for year in sorted(manager.all_years, reverse=True):
                is_valid = validation[code_type][year]
                status = "✓" if is_valid else "✗"
                print(f"  {status} {year}")
                
                if not is_valid:
                    all_valid = False
            print()
        
        if all_valid:
            print("✅ All years validated successfully")
        else:
            print("⚠️  Some years are missing data")
        
        print()
        return 0 if all_valid else 1
    
    # Default: show summary
    print(manager.get_summary_report())
    return 0


# ============================================================================
# 5. MAIN BLOCK
# ============================================================================

if __name__ == "__main__":
    """
    Test and run year manager
    Usage: python ingestion/cms/year_manager.py --summary
    """
    sys.exit(main())