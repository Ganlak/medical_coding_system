"""
Medical Coding System - Extraction Agent
Extracts diagnoses and procedures from medical documentation using LLM
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Dict, List, Optional, Any


# ============================================================================
# 1. IMPORT DEPENDENCIES
# ============================================================================
from agents.base_agent import PromptBasedAgent
from config.prompts import get_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 2. EXTRACTION AGENT CLASS
# ============================================================================

class ExtractionAgent(PromptBasedAgent):
    """
    Agent specialized in extracting medical entities from clinical documentation
    Extracts diagnoses, procedures, symptoms, and related clinical information
    """
    
    def __init__(
        self,
        agent_name: str = "ExtractionAgent",
        temperature: float = 0.1,  # Low temperature for consistent extraction
        **kwargs
    ):
        """
        Initialize extraction agent
        
        Args:
            agent_name: Name of the agent
            temperature: Sampling temperature (low for consistency)
            **kwargs: Additional arguments for PromptBasedAgent
        """
        super().__init__(
            agent_name=agent_name,
            temperature=temperature,
            **kwargs
        )
        
        logger.info(f"  ✓ {agent_name} ready for medical entity extraction")
    
    def extract_diagnoses(self, document_text: str) -> Dict[str, Any]:
        """
        Extract diagnoses from medical document
        
        Args:
            document_text: Medical document text
        
        Returns:
            Dictionary with extracted diagnoses and metadata
        """
        try:
            # Get diagnosis extraction prompt
            prompt = get_prompt("diagnosis_extraction", document_text=document_text)
        except KeyError:
            # Fallback prompt if template not found
            prompt = f"""Extract all diagnoses from the following medical document.

**Input Document:**
{document_text}

**Instructions:**
1. Identify all diagnoses, conditions, and symptoms mentioned
2. Extract the exact medical terminology used
3. Include both primary and secondary diagnoses
4. Note any qualifiers (acute, chronic, suspected, etc.)

**Output Format (JSON):**
{{
  "diagnoses": [
    {{
      "condition": "exact medical term",
      "qualifier": "acute/chronic/suspected/etc",
      "severity": "mild/moderate/severe if mentioned",
      "context": "relevant clinical context"
    }}
  ]
}}

Extract the diagnoses now:"""
        
        # Process with LLM
        response = self.process_with_prompt(
            user_prompt=prompt,
            parse_json=True,
            response_format={"type": "json_object"}
        )
        
        # Validate and return
        if response.get('parsed'):
            diagnoses = response['parsed'].get('diagnoses', [])
            
            logger.info(f"  ✓ Extracted {len(diagnoses)} diagnoses from document")
            
            return {
                'success': True,
                'diagnoses': diagnoses,
                'count': len(diagnoses),
                'raw_response': response['content'],
                'tokens_used': response['usage']['total_tokens']
            }
        else:
            logger.warning("  ⚠ Failed to parse diagnosis extraction response")
            return {
                'success': False,
                'diagnoses': [],
                'count': 0,
                'error': 'Failed to parse JSON response'
            }
    
    def extract_procedures(self, document_text: str) -> Dict[str, Any]:
        """
        Extract procedures from medical document
        
        Args:
            document_text: Medical document text
        
        Returns:
            Dictionary with extracted procedures and metadata
        """
        try:
            # Get procedure extraction prompt
            prompt = get_prompt("procedure_extraction", document_text=document_text)
        except KeyError:
            # Fallback prompt
            prompt = f"""Extract all procedures from the following medical document.

**Input Document:**
{document_text}

**Instructions:**
1. Identify all procedures, surgeries, and interventions mentioned
2. Extract exact procedural terminology
3. Include anatomical locations and surgical approach
4. Note laterality (left/right/bilateral) if applicable

**Output Format (JSON):**
{{
  "procedures": [
    {{
      "procedure_name": "exact procedural term",
      "anatomical_site": "specific body location",
      "laterality": "left/right/bilateral/none",
      "approach": "open/laparoscopic/endoscopic/etc",
      "context": "relevant clinical details"
    }}
  ]
}}

Extract the procedures now:"""
        
        # Process with LLM
        response = self.process_with_prompt(
            user_prompt=prompt,
            parse_json=True,
            response_format={"type": "json_object"}
        )
        
        # Validate and return
        if response.get('parsed'):
            procedures = response['parsed'].get('procedures', [])
            
            logger.info(f"  ✓ Extracted {len(procedures)} procedures from document")
            
            return {
                'success': True,
                'procedures': procedures,
                'count': len(procedures),
                'raw_response': response['content'],
                'tokens_used': response['usage']['total_tokens']
            }
        else:
            logger.warning("  ⚠ Failed to parse procedure extraction response")
            return {
                'success': False,
                'procedures': [],
                'count': 0,
                'error': 'Failed to parse JSON response'
            }
    
    def extract_all(self, document_text: str) -> Dict[str, Any]:
        """
        Extract both diagnoses and procedures from document
        
        Args:
            document_text: Medical document text
        
        Returns:
            Dictionary with all extracted entities
        """
        logger.info("Starting comprehensive medical entity extraction...")
        
        # Extract diagnoses
        diagnosis_result = self.extract_diagnoses(document_text)
        
        # Extract procedures
        procedure_result = self.extract_procedures(document_text)
        
        # Combine results
        total_tokens = (
            diagnosis_result.get('tokens_used', 0) +
            procedure_result.get('tokens_used', 0)
        )
        
        result = {
            'success': diagnosis_result['success'] and procedure_result['success'],
            'diagnoses': diagnosis_result.get('diagnoses', []),
            'procedures': procedure_result.get('procedures', []),
            'counts': {
                'diagnoses': diagnosis_result.get('count', 0),
                'procedures': procedure_result.get('count', 0),
                'total': diagnosis_result.get('count', 0) + procedure_result.get('count', 0)
            },
            'tokens_used': total_tokens
        }
        
        logger.info(f"  ✓ Extraction complete: "
                   f"{result['counts']['diagnoses']} diagnoses, "
                   f"{result['counts']['procedures']} procedures "
                   f"({total_tokens} tokens)")
        
        return result
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process extraction request
        
        Args:
            input_data: Dictionary with 'document_text' and optional 'extract_type'
        
        Returns:
            Extraction results
        """
        document_text = input_data.get('document_text', '')
        extract_type = input_data.get('extract_type', 'all')  # 'diagnoses', 'procedures', 'all'
        
        if not document_text:
            return {
                'success': False,
                'error': 'No document text provided'
            }
        
        # Route to appropriate extraction method
        if extract_type == 'diagnoses':
            return self.extract_diagnoses(document_text)
        elif extract_type == 'procedures':
            return self.extract_procedures(document_text)
        else:
            return self.extract_all(document_text)


# ============================================================================
# 3. MAIN BLOCK - Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate extraction agent
    Usage: python agents/extraction_agent.py
    """
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - EXTRACTION AGENT TEST")
    print("=" * 80)
    print()
    
    # Initialize extraction agent
    print("🤖 Initializing Extraction Agent...")
    extraction_agent = ExtractionAgent()
    print()
    
    # Sample medical document
    sample_document = """
    PATIENT: John Doe
    DATE: November 5, 2025
    
    CHIEF COMPLAINT: Shortness of breath and chest pain
    
    HISTORY OF PRESENT ILLNESS:
    65-year-old male presents with acute onset of shortness of breath and 
    substernal chest pain for the past 2 hours. Patient has a history of 
    type 2 diabetes mellitus and essential hypertension.
    
    PAST MEDICAL HISTORY:
    - Type 2 diabetes mellitus
    - Essential hypertension
    - Hyperlipidemia
    
    PROCEDURES PERFORMED:
    - Electrocardiogram (ECG)
    - Cardiac catheterization with coronary angiography
    - Percutaneous coronary intervention (PCI) with stent placement in left anterior descending artery
    
    FINDINGS:
    - Acute ST-elevation myocardial infarction (STEMI)
    - 90% stenosis of left anterior descending artery
    - Successful stent placement
    
    DIAGNOSIS:
    1. Acute ST-elevation myocardial infarction (STEMI)
    2. Type 2 diabetes mellitus
    3. Essential hypertension
    4. Hyperlipidemia
    """
    
    # Test 1: Extract diagnoses only
    print("📋 Test 1: Extracting Diagnoses")
    print("-" * 80)
    diagnosis_result = extraction_agent.extract_diagnoses(sample_document)
    
    if diagnosis_result['success']:
        print(f"✓ Extracted {diagnosis_result['count']} diagnoses:")
        for i, diag in enumerate(diagnosis_result['diagnoses'], 1):
            print(f"\n  {i}. {diag.get('condition', 'N/A')}")
            print(f"     Qualifier: {diag.get('qualifier', 'N/A')}")
            print(f"     Severity: {diag.get('severity', 'N/A')}")
            print(f"     Context: {diag.get('context', 'N/A')[:60]}...")
        print(f"\n  Tokens used: {diagnosis_result['tokens_used']}")
    else:
        print(f"✗ Extraction failed: {diagnosis_result.get('error')}")
    print()
    
    # Test 2: Extract procedures only
    print("📋 Test 2: Extracting Procedures")
    print("-" * 80)
    procedure_result = extraction_agent.extract_procedures(sample_document)
    
    if procedure_result['success']:
        print(f"✓ Extracted {procedure_result['count']} procedures:")
        for i, proc in enumerate(procedure_result['procedures'], 1):
            print(f"\n  {i}. {proc.get('procedure_name', 'N/A')}")
            print(f"     Anatomical site: {proc.get('anatomical_site', 'N/A')}")
            print(f"     Laterality: {proc.get('laterality', 'N/A')}")
            print(f"     Approach: {proc.get('approach', 'N/A')}")
            print(f"     Context: {proc.get('context', 'N/A')[:60]}...")
        print(f"\n  Tokens used: {procedure_result['tokens_used']}")
    else:
        print(f"✗ Extraction failed: {procedure_result.get('error')}")
    print()
    
    # Test 3: Extract all entities
    print("📋 Test 3: Extracting All Entities")
    print("-" * 80)
    all_result = extraction_agent.extract_all(sample_document)
    
    if all_result['success']:
        print(f"✓ Total entities extracted:")
        print(f"  • Diagnoses: {all_result['counts']['diagnoses']}")
        print(f"  • Procedures: {all_result['counts']['procedures']}")
        print(f"  • Total: {all_result['counts']['total']}")
        print(f"  • Total tokens: {all_result['tokens_used']}")
    else:
        print(f"✗ Extraction failed")
    print()
    
    # Test 4: Process method
    print("📋 Test 4: Using process() method")
    print("-" * 80)
    
    # Extract only diagnoses via process
    process_result = extraction_agent.process({
        'document_text': sample_document,
        'extract_type': 'diagnoses'
    })
    
    print(f"✓ Process method extracted {process_result['count']} diagnoses")
    print()
    
    # Test 5: Simple document
    print("📋 Test 5: Simple Document")
    print("-" * 80)
    
    simple_doc = """
    Patient presents with fever and cough.
    Diagnosis: Acute bronchitis
    """
    
    simple_result = extraction_agent.extract_all(simple_doc)
    
    if simple_result['success']:
        print(f"✓ Simple document extraction:")
        print(f"  Diagnoses: {len(simple_result['diagnoses'])}")
        if simple_result['diagnoses']:
            print(f"  - {simple_result['diagnoses'][0].get('condition', 'N/A')}")
        print(f"  Procedures: {len(simple_result['procedures'])}")
    print()
    
    # Test 6: Agent statistics
    print("📊 Agent Statistics:")
    print("-" * 80)
    stats = extraction_agent.get_stats()
    print(f"  Total LLM calls: {stats['total_calls']}")
    print(f"  Successful calls: {stats['successful_calls']}")
    print(f"  Total tokens used: {stats['total_tokens']}")
    print(f"  Average tokens per call: {stats['avg_tokens_per_call']:.1f}")
    print(f"  Average time per call: {stats['avg_time_per_call']:.2f}s")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ EXTRACTION AGENT TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Usage Examples:")
    print()
    print("  from agents.extraction_agent import ExtractionAgent")
    print()
    print("  # Initialize agent")
    print("  agent = ExtractionAgent()")
    print()
    print("  # Extract diagnoses")
    print("  result = agent.extract_diagnoses(document_text)")
    print()
    print("  # Extract procedures")
    print("  result = agent.extract_procedures(document_text)")
    print()
    print("  # Extract everything")
    print("  result = agent.extract_all(document_text)")
    print()