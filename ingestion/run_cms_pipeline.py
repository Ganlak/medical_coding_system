"""
Medical Coding System - CMS Pipeline Orchestrator
Single command to run complete CMS ingestion, normalization, and embedding pipeline
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
from typing import List, Dict, Any


# ============================================================================
# 1. IMPORT COMPONENTS
# ============================================================================
from config.settings import CMSConfig
from utils.logger import get_logger

# Import pipeline components
from ingestion.cms.fetch_cms_codes import fetch_cms_codes
from ingestion.cms.normalize_cms_codes import normalize_codes
from ingestion.embeddings.embed_cms_codes import CMSCodeEmbedder

logger = get_logger(__name__)


# ============================================================================
# 2. PIPELINE ORCHESTRATOR
# ============================================================================

class CMSPipelineOrchestrator:
    """
    Orchestrates the complete CMS code pipeline:
    1. Fetch codes from CMS.gov
    2. Normalize and clean codes
    3. Embed codes into ChromaDB
    """
    
    def __init__(self, year: int, code_types: List[str]):
        """
        Initialize pipeline orchestrator
        
        Args:
            year: Fiscal year
            code_types: List of code types to process
        """
        self.year = year
        self.code_types = code_types
        self.results = {
            'fetch': {},
            'normalize': {},
            'embed': {}
        }
        
        logger.info(f"Initialized CMS Pipeline for year {year}")
        logger.info(f"Code types: {', '.join(code_types)}")
    
    def run_fetch(self, extract: bool = True) -> bool:
            """
            Step 1: Fetch CMS codes
            
            Args:
                extract: Whether to extract ZIP files
            
            Returns:
                True if successful
            """
            logger.info("")
            logger.info("=" * 80)
            logger.info("STEP 1/3: FETCHING CMS CODES")
            logger.info("=" * 80)
            
            try:
                from config.settings import RAW_DATA_DIR
                
                # Expand code types to detailed types (same as CLI does)
                detailed_code_types = []
                
                # Import the code type sets from fetch_cms_codes
                from ingestion.cms.fetch_cms_codes import ICD10CM_SET, ICD10PCS_SET, CPT_SET
                
                for code_type in self.code_types:
                    if code_type == 'icd10cm':
                        detailed_code_types.extend(ICD10CM_SET)
                    elif code_type == 'icd10pcs':
                        detailed_code_types.extend(ICD10PCS_SET)
                    elif code_type == 'cpt':
                        detailed_code_types.extend(CPT_SET)
                    else:
                        detailed_code_types.append(code_type)
                
                # Remove duplicates
                detailed_code_types = list(set(detailed_code_types))
                
                logger.info(f"Expanded to detailed types: {detailed_code_types}")
                
                success_count, failed = fetch_cms_codes(
                    year=self.year,
                    code_types=detailed_code_types,
                    output_dir=RAW_DATA_DIR,
                    extract=extract
                )
                
                self.results['fetch'] = {
                    'success_count': success_count,
                    'failed': failed,
                    'total': len(detailed_code_types)
                }
                
                if success_count == len(detailed_code_types):
                    logger.info(f"✓ Fetch complete: {success_count}/{len(detailed_code_types)} succeeded")
                    return True
                else:
                    logger.warning(f"⚠ Fetch partial: {success_count}/{len(detailed_code_types)} succeeded")
                    logger.warning(f"  Failed: {', '.join(failed)}")
                    return False
            
            except Exception as e:
                logger.error(f"✗ Fetch failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    
    def run_normalize(self) -> bool:
        """
        Step 2: Normalize CMS codes
        
        Returns:
            True if successful
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 2/3: NORMALIZING CMS CODES")
        logger.info("=" * 80)
        
        try:
            from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
            
            success_count = 0
            
            for code_type in self.code_types:
                output_file = normalize_codes(
                    year=self.year,
                    code_type=code_type,
                    raw_dir=RAW_DATA_DIR,
                    output_dir=PROCESSED_DATA_DIR
                )
                
                if output_file:
                    success_count += 1
                    self.results['normalize'][code_type] = str(output_file)
            
            if success_count == len(self.code_types):
                logger.info(f"✓ Normalize complete: {success_count}/{len(self.code_types)} succeeded")
                return True
            else:
                logger.warning(f"⚠ Normalize partial: {success_count}/{len(self.code_types)} succeeded")
                return False
        
        except Exception as e:
            logger.error(f"✗ Normalize failed: {e}")
            return False
    
    def run_embed(self, force_recreate: bool = False, batch_size: int = 100) -> bool:
        """
        Step 3: Embed CMS codes into ChromaDB
        
        Args:
            force_recreate: Whether to recreate collections
            batch_size: Embedding batch size
        
        Returns:
            True if successful
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 3/3: EMBEDDING CMS CODES")
        logger.info("=" * 80)
        
        try:
            embedder = CMSCodeEmbedder()
            
            success_count = 0
            total_embedded = 0
            
            for code_type in self.code_types:
                result = embedder.embed_codes(
                    code_type=code_type,
                    year=self.year,
                    force_recreate=force_recreate,
                    batch_size=batch_size
                )
                
                if result['success']:
                    success_count += 1
                    total_embedded += result['embedded_count']
                    self.results['embed'][code_type] = result
            
            if success_count == len(self.code_types):
                logger.info(f"✓ Embed complete: {success_count}/{len(self.code_types)} succeeded")
                logger.info(f"  Total codes embedded: {total_embedded:,}")
                return True
            else:
                logger.warning(f"⚠ Embed partial: {success_count}/{len(self.code_types)} succeeded")
                return False
        
        except Exception as e:
            logger.error(f"✗ Embed failed: {e}")
            return False
    
    def run_full_pipeline(
        self,
        skip_fetch: bool = False,
        skip_normalize: bool = False,
        skip_embed: bool = False,
        force_embed: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Run complete pipeline
        
        Args:
            skip_fetch: Skip fetch step
            skip_normalize: Skip normalize step
            skip_embed: Skip embed step
            force_embed: Force recreate embeddings
            batch_size: Embedding batch size
        
        Returns:
            Pipeline results dictionary
        """
        start_time = time.time()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("CMS CODE PIPELINE - STARTING")
        logger.info("=" * 80)
        logger.info(f"Year: {self.year}")
        logger.info(f"Code Types: {', '.join(self.code_types)}")
        logger.info(f"Steps: {'Fetch' if not skip_fetch else ''} "
                   f"{'Normalize' if not skip_normalize else ''} "
                   f"{'Embed' if not skip_embed else ''}")
        logger.info("=" * 80)
        
        pipeline_success = True
        
        # Step 1: Fetch
        if not skip_fetch:
            fetch_success = self.run_fetch()
            if not fetch_success:
                logger.error("Pipeline stopped: Fetch step failed")
                pipeline_success = False
        
        # Step 2: Normalize (only if fetch succeeded or was skipped)
        if pipeline_success and not skip_normalize:
            normalize_success = self.run_normalize()
            if not normalize_success:
                logger.error("Pipeline stopped: Normalize step failed")
                pipeline_success = False
        
        # Step 3: Embed (only if previous steps succeeded or were skipped)
        if pipeline_success and not skip_embed:
            embed_success = self.run_embed(force_recreate=force_embed, batch_size=batch_size)
            if not embed_success:
                logger.error("Pipeline stopped: Embed step failed")
                pipeline_success = False
        
        duration = time.time() - start_time
        
        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("CMS CODE PIPELINE - SUMMARY")
        logger.info("=" * 80)
        
        if pipeline_success:
            logger.info("✅ Pipeline completed successfully!")
        else:
            logger.error("❌ Pipeline completed with errors")
        
        logger.info(f"Total duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        
        # Detailed results
        if not skip_fetch and 'fetch' in self.results:
            fetch = self.results['fetch']
            logger.info(f"\nFetch: {fetch.get('success_count', 0)}/{fetch.get('total', 0)} succeeded")
        
        if not skip_normalize and 'normalize' in self.results:
            logger.info(f"Normalize: {len(self.results['normalize'])} files created")
        
        if not skip_embed and 'embed' in self.results:
            total_embedded = sum(
                r.get('embedded_count', 0) 
                for r in self.results['embed'].values()
            )
            logger.info(f"Embed: {total_embedded:,} codes embedded")
        
        logger.info("=" * 80)
        
        return {
            'success': pipeline_success,
            'duration': duration,
            'results': self.results
        }


# ============================================================================
# 3. COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run complete CMS code pipeline (fetch, normalize, embed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for 2025
  python ingestion/run_cms_pipeline.py --year 2025 --codes all
  
  # Run only for ICD-10-CM
  python ingestion/run_cms_pipeline.py --year 2025 --codes icd10cm
  
  # Skip fetch (use existing downloaded files)
  python ingestion/run_cms_pipeline.py --year 2025 --codes all --skip-fetch
  
  # Force recreate embeddings
  python ingestion/run_cms_pipeline.py --year 2025 --codes all --force-embed
  
  # Run only normalize and embed
  python ingestion/run_cms_pipeline.py --year 2025 --codes all --skip-fetch
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
        help='Code types to process'
    )
    
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetch step (use existing files)'
    )
    
    parser.add_argument(
        '--skip-normalize',
        action='store_true',
        help='Skip normalize step'
    )
    
    parser.add_argument(
        '--skip-embed',
        action='store_true',
        help='Skip embed step'
    )
    
    parser.add_argument(
        '--force-embed',
        action='store_true',
        help='Force recreate embeddings (delete existing)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Embedding batch size (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Expand 'all'
    if 'all' in args.codes:
        code_types = ['icd10cm', 'icd10pcs', 'cpt']
    else:
        code_types = args.codes
    
    # Initialize orchestrator
    orchestrator = CMSPipelineOrchestrator(
        year=args.year,
        code_types=code_types
    )
    
    # Run pipeline
    result = orchestrator.run_full_pipeline(
        skip_fetch=args.skip_fetch,
        skip_normalize=args.skip_normalize,
        skip_embed=args.skip_embed,
        force_embed=args.force_embed,
        batch_size=args.batch_size
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


# ============================================================================
# 4. MAIN BLOCK
# ============================================================================

if __name__ == "__main__":
    """
    Run CMS pipeline orchestrator
    Usage: python ingestion/run_cms_pipeline.py --year 2025 --codes all
    """
    main()