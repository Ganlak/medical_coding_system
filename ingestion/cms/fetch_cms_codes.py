"""
Medical Coding System - CMS Code Fetcher
Downloads ICD-10-CM, ICD-10-PCS, and CPT/HCPCS codes from CMS.gov with year-wise management
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from typing import Dict, List, Optional, Tuple
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================================
# 1. IMPORT CONFIGURATION
# ============================================================================
try:
    from config.settings import CMSConfig, RAW_DATA_DIR
    from utils.logger import get_logger
except ImportError:
    # Fallback for standalone usage
    RAW_DATA_DIR = Path("./data/raw")
    
    class CMSConfig:
        CURRENT_YEAR = 2025
        CODE_TYPES = ["icd10cm", "icd10pcs", "cpt"]
    
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    def get_logger(name):
        return MockLogger()


logger = get_logger(__name__)


# ============================================================================
# 2. CMS URLS AND PATTERNS
# ============================================================================

# CMS Website URLs
ICD10_PAGE = "https://www.cms.gov/medicare/coding-billing/icd-10-codes"
CPT_HCPCS_PAGE = "https://www.cms.gov/medicare/regulations-guidance/physician-self-referral/list-cpt-hcpcs-codes"
FILES_BASE = "https://www.cms.gov/files/zip/"
AMA_LICENSE_BASE = "https://www.cms.gov/apps/ama/license.asp"

# CPT/HCPCS URLs by year
CPT_HCPCS_URLS = {
    2025: {
        "file": "/files/zip/list-codes-effective-january-1-2025-published-november-26-2024.zip",
        "pdf": "/files/document/annual-update-list-cpt/hcpcs-codes-effective-january-1-2025.pdf",
    },
    2024: {
        "file": "/files/zip/updated-list-codes-effective-january-1-2024-published-march-1-2024-all-codes-effective-january-1-2024-unless-otherwise-indicated-code-list.zip",
        "pdf": "/files/document/annual-update-list-cpt-hcpcs-codes-effective-january-1-2024.pdf",
    },
    2023: {
        "file": "/files/zip/list-codes-effective-january-1-2023-published-december-1-2022.zip",
        "pdf": "/files/document/annual-update-list-cpt-hcpcs-codes-effective-january-1-2023-published-december-1-2022.pdf",
    },
}

# Fallback URL patterns
FALLBACK_SLUGS = {
    # ICD-10-CM
    "icd10cm_poa": "{year}-poa-exempt-codes.zip",
    "icd10cm_conversion": "{year}-conversion-table.zip",
    "icd10cm_code_descriptions": "{year}-code-descriptions-tabular-order.zip",
    "icd10cm_addendum": "{year}-addendum.zip",
    "icd10cm_code_tables": "{year}-code-tables-tabular-and-index.zip",
    
    # ICD-10-PCS
    "icd10pcs_order": "{year}-icd-10-pcs-order-file-long-and-abbreviated-titles.zip",
    "icd10pcs_codes": "{year}-icd-10-pcs-codes-file.zip",
    "icd10pcs_conversion": "{year}-icd-10-pcs-conversion-table.zip",
    "icd10pcs_tables": "{year}-icd-10-pcs-code-tables-and-index.zip",
    "icd10pcs_addendum": "{year}-icd-10-pcs-addendum.zip",
    
    # CPT/HCPCS
    "cpt_hcpcs_codes": "list-codes-effective-january-1-{year}.zip",
}

# Code type groups
ICD10CM_SET = ["icd10cm_code_descriptions", "icd10cm_code_tables"]
ICD10PCS_SET = ["icd10pcs_order", "icd10pcs_codes"]
CPT_SET = ["cpt_hcpcs_codes"]


# ============================================================================
# 3. UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: Path, description: str = "Downloading") -> bool:
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        output_path: Path to save file
        description: Description for progress bar
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading: {url}")
        
        response = requests.get(url, stream=True, timeout=120, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✓ Downloaded: {output_path.name}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False


def download_with_ama_license(file_path: str, output_path: Path) -> bool:
    """
    Download CPT/HCPCS file through AMA license gateway
    
    Args:
        file_path: CMS file path
        output_path: Local output path
    
    Returns:
        True if successful, False otherwise
    """
    license_url = f"{AMA_LICENSE_BASE}?file={file_path}"
    
    logger.info(f"Handling AMA license for: {file_path}")
    
    try:
        session = requests.Session()
        
        # Attempt direct download (license often auto-accepted)
        download_url = f"https://www.cms.gov{file_path}"
        response = session.get(download_url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading CPT") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✓ Downloaded: {output_path.name}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """
    Extract ZIP file
    
    Args:
        zip_path: Path to ZIP file
        dest_dir: Destination directory
    
    Returns:
        True if successful, False otherwise
    """
    ensure_dir(dest_dir)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        logger.info(f"✓ Extracted: {zip_path.name} -> {dest_dir}")
        return True
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        return False


def scrape_cms_page(page_url: str) -> List[str]:
    """
    Scrape CMS page for ZIP file links
    
    Args:
        page_url: CMS page URL
    
    Returns:
        List of ZIP file URLs
    """
    try:
        response = requests.get(page_url, timeout=60)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        zip_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.zip') and '/files/' in href:
                full_url = urljoin("https://www.cms.gov", href)
                zip_links.append(full_url)
        
        return sorted(set(zip_links))
    
    except Exception as e:
        logger.warning(f"Scraping failed for {page_url}: {e}")
        return []


# ============================================================================
# 4. URL BUILDERS
# ============================================================================

def build_download_urls(year: int, code_types: List[str]) -> Dict[str, str]:
    """
    Build download URLs for requested code types and year
    
    Args:
        year: Fiscal year
        code_types: List of code types to download
    
    Returns:
        Dictionary mapping code_type to download URL
    """
    urls = {}
    
    logger.info(f"Building URLs for year {year}...")
    
    for code_type in code_types:
        # CPT/HCPCS codes
        if code_type == "cpt_hcpcs_codes":
            if year in CPT_HCPCS_URLS:
                urls[code_type] = CPT_HCPCS_URLS[year]["file"]
                logger.info(f"  ✓ {code_type}: Found predefined URL")
            else:
                logger.warning(f"  ⚠ {code_type}: No URL for year {year}")
            continue
        
        # ICD-10 codes - try scraping first
        if code_type.startswith("icd10"):
            scraped_links = scrape_cms_page(ICD10_PAGE)
            
            # Build regex pattern for this code type
            patterns = {
                "icd10cm_code_descriptions": rf"/files/zip/{year}-code-descriptions-tabular-order\.zip$",
                "icd10cm_code_tables": rf"/files/zip/{year}-code-tables-tabular-and-index\.zip$",
                "icd10pcs_order": rf"/files/zip/{year}-icd-10-pcs-order-file.*\.zip$",
                "icd10pcs_codes": rf"/files/zip/{year}-icd-10-pcs-codes-file\.zip$",
            }
            
            if code_type in patterns:
                pattern = re.compile(patterns[code_type], re.IGNORECASE)
                match = next((link for link in scraped_links if pattern.search(link)), None)
                
                if match:
                    urls[code_type] = match
                    logger.info(f"  ✓ {code_type}: Found via scraping")
                    continue
        
        # Fallback to predefined slugs
        if code_type in FALLBACK_SLUGS:
            slug = FALLBACK_SLUGS[code_type].format(year=year)
            urls[code_type] = urljoin(FILES_BASE, slug)
            logger.warning(f"  ⚠ {code_type}: Using fallback URL")
    
    return urls


# ============================================================================
# 5. MAIN FETCH LOGIC
# ============================================================================

def fetch_cms_codes(
    year: int,
    code_types: List[str],
    output_dir: Path,
    extract: bool = True
) -> Tuple[int, List[str]]:
    """
    Fetch CMS codes for specified year and types
    
    Args:
        year: Fiscal year
        code_types: List of code types to fetch
        output_dir: Output directory
        extract: Whether to extract ZIP files
    
    Returns:
        Tuple of (success_count, failed_list)
    """
    ensure_dir(output_dir)
    
    # Build URLs
    urls = build_download_urls(year, code_types)
    
    success_count = 0
    failed = []
    
    # Download each file
    for code_type in code_types:
        if code_type not in urls:
            logger.error(f"✗ {code_type}: No URL available")
            failed.append(code_type)
            continue
        
        url_or_path = urls[code_type]
        
        # Determine output paths
        if code_type.startswith("cpt"):
            code_family = "cpt"
        elif code_type.startswith("icd10cm"):
            code_family = "icd10cm"
        elif code_type.startswith("icd10pcs"):
            code_family = "icd10pcs"
        else:
            code_family = "other"
        
        family_dir = output_dir / code_family
        ensure_dir(family_dir)
        
        zip_path = family_dir / f"{code_type}_{year}.zip"
        extract_dir = family_dir / f"{code_type}_{year}"
        
        # Download
        logger.info(f"\n[{code_type}] Downloading...")
        
        if code_type == "cpt_hcpcs_codes":
            success = download_with_ama_license(url_or_path, zip_path)
        else:
            success = download_file(url_or_path, zip_path, f"Downloading {code_type}")
        
        if not success:
            failed.append(code_type)
            continue
        
        # Extract if requested
        if extract:
            if extract_zip(zip_path, extract_dir):
                success_count += 1
            else:
                failed.append(code_type)
        else:
            success_count += 1
    
    return success_count, failed


# ============================================================================
# 6. COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Fetch CMS medical codes (ICD-10, CPT/HCPCS) by year",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all 2025 codes
  python -m ingestion.cms.fetch_cms_codes --year 2025 --codes all
  
  # Download only ICD-10-CM codes
  python -m ingestion.cms.fetch_cms_codes --year 2025 --codes icd10cm
  
  # Download CPT codes
  python -m ingestion.cms.fetch_cms_codes --year 2025 --codes cpt
  
  # Download to custom directory
  python -m ingestion.cms.fetch_cms_codes --year 2025 --codes all --output ./my_data
        """
    )
    
    parser.add_argument(
        "--year",
        type=int,
        default=CMSConfig.CURRENT_YEAR,
        help=f"Fiscal year (default: {CMSConfig.CURRENT_YEAR})"
    )
    
    parser.add_argument(
        "--codes",
        nargs="+",
        default=["all"],
        choices=["all", "icd10cm", "icd10pcs", "cpt"] + list(FALLBACK_SLUGS.keys()),
        help="Code types to download"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Output directory (default: {RAW_DATA_DIR})"
    )
    
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract ZIP files"
    )
    
    args = parser.parse_args()
    
    # Expand code type groups
    code_types = []
    for code_group in args.codes:
        if code_group == "all":
            code_types.extend(ICD10CM_SET + ICD10PCS_SET + CPT_SET)
        elif code_group == "icd10cm":
            code_types.extend(ICD10CM_SET)
        elif code_group == "icd10pcs":
            code_types.extend(ICD10PCS_SET)
        elif code_group == "cpt":
            code_types.extend(CPT_SET)
        else:
            code_types.append(code_group)
    
    code_types = sorted(set(code_types))
    
    # Display configuration
    print("=" * 80)
    print("CMS MEDICAL CODE FETCHER")
    print("=" * 80)
    print(f"Year: {args.year}")
    print(f"Code Types: {', '.join(code_types)}")
    print(f"Output: {args.output}")
    print(f"Extract: {not args.no_extract}")
    print("=" * 80)
    print()
    
    # Fetch codes
    start_time = time.time()
    
    success_count, failed = fetch_cms_codes(
        year=args.year,
        code_types=code_types,
        output_dir=args.output,
        extract=not args.no_extract
    )
    
    duration = time.time() - start_time
    
    # Summary
    print()
    print("=" * 80)
    print("FETCH COMPLETE")
    print("=" * 80)
    print(f"✓ Successful: {success_count}/{len(code_types)}")
    if failed:
        print(f"✗ Failed: {', '.join(failed)}")
    print(f"⏱ Duration: {duration:.1f}s")
    print(f"📁 Output: {args.output}")
    print("=" * 80)
    
    return 0 if success_count == len(code_types) else 1


# ============================================================================
# 7. MAIN BLOCK
# ============================================================================

if __name__ == "__main__":
    """
    Test and run CMS code fetcher
    Usage: python ingestion/cms/fetch_cms_codes.py --year 2025 --codes all
    """
    sys.exit(main())