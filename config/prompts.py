"""
Medical Coding System - LLM Prompt Templates
Production-ready prompts for extraction, coding, validation, and re-ranking
"""
from typing import Dict, List, Optional


# ============================================================================
# 1. DIAGNOSIS EXTRACTION PROMPTS
# ============================================================================

DIAGNOSIS_EXTRACTION_PROMPT = """You are a medical coding expert specializing in extracting diagnoses from clinical documentation.

**Task:** Extract all diagnoses mentioned in the following medical document.

**Input Document:**
{document_text}

**Instructions:**
1. Identify all diagnoses, conditions, and symptoms mentioned
2. Extract the exact medical terminology used
3. Include both primary and secondary diagnoses
4. Note any qualifiers (acute, chronic, suspected, rule out, etc.)
5. Preserve the clinical context

**Output Format (JSON):**
{{
  "diagnoses": [
    {{
      "condition": "exact medical term from document",
      "qualifier": "acute/chronic/suspected/history of/etc",
      "severity": "mild/moderate/severe if mentioned",
      "context": "relevant clinical context",
      "location_in_text": "exact quote from document"
    }}
  ]
}}

**Important:**
- Extract ONLY what is explicitly stated in the document
- Do NOT infer or assume diagnoses not mentioned
- Include all modifiers and specificity details
- Maintain exact medical terminology

Extract the diagnoses now:"""


# ============================================================================
# 2. PROCEDURE EXTRACTION PROMPTS
# ============================================================================

PROCEDURE_EXTRACTION_PROMPT = """You are a medical coding expert specializing in extracting procedures from clinical documentation.

**Task:** Extract all procedures, treatments, and interventions from the following medical document.

**Input Document:**
{document_text}

**Instructions:**
1. Identify all procedures, surgeries, and interventions mentioned
2. Extract exact procedural terminology used
3. Include anatomical locations and surgical approach (if mentioned)
4. Note laterality (left/right/bilateral) if applicable
5. Capture any modifiers, complications, or additional details

**Output Format (JSON):**
{{
  "procedures": [
    {{
      "procedure_name": "exact procedural term from document",
      "anatomical_site": "specific body location",
      "laterality": "left/right/bilateral/none",
      "approach": "open/laparoscopic/endoscopic/percutaneous/etc",
      "context": "relevant clinical details",
      "location_in_text": "exact quote from document"
    }}
  ]
}}

**Important:**
- Extract ONLY explicitly mentioned procedures
- Include all anatomical and approach details
- Note any complications or additional procedures
- Preserve exact medical terminology

Extract the procedures now:"""


# ============================================================================
# 3. CODE SUGGESTION PROMPTS
# ============================================================================

ICD10_CODE_SUGGESTION_PROMPT = """You are an expert medical coder specializing in ICD-10-CM and ICD-10-PCS coding.

**Task:** Suggest the most appropriate ICD-10 codes for the following medical condition/procedure.

**Medical Entity:**
{medical_entity}

**Additional Context:**
{context}

**Retrieved Candidate Codes:**
{retrieved_codes}

**Instructions:**
1. Analyze the medical entity and its context
2. Review the candidate codes retrieved from the database
3. Select the most specific and appropriate code(s)
4. Consider code specificity, laterality, and additional characters
5. Provide confidence score (0.0 to 1.0) for each suggestion

**Output Format (JSON):**
{{
  "suggestions": [
    {{
      "code": "ICD-10 code",
      "description": "code description",
      "confidence": 0.95,
      "reasoning": "why this code is appropriate",
      "specificity_level": "high/medium/low"
    }}
  ],
  "notes": "any additional coding notes or considerations"
}}

**Coding Rules:**
- Always select the most specific code available
- Consider combination codes when applicable
- Note if additional codes are needed
- Flag if documentation is insufficient

Provide your code suggestions now:"""


CPT_CODE_SUGGESTION_PROMPT = """You are an expert medical coder specializing in CPT (Current Procedural Terminology) coding.

**Task:** Suggest the most appropriate CPT codes for the following procedure.

**Procedure:**
{procedure}

**Additional Context:**
{context}

**Retrieved Candidate Codes:**
{retrieved_codes}

**Instructions:**
1. Analyze the procedure and its clinical context
2. Review the candidate CPT codes from the database
3. Select the most appropriate code(s)
4. Consider modifiers if needed (e.g., -50, -LT, -RT, -59)
5. Provide confidence score for each suggestion

**Output Format (JSON):**
{{
  "suggestions": [
    {{
      "code": "CPT code",
      "description": "procedure description",
      "modifiers": ["modifier codes if applicable"],
      "confidence": 0.90,
      "reasoning": "why this code is appropriate",
      "additional_codes": ["related codes to consider"]
    }}
  ],
  "notes": "any additional coding considerations"
}}

**Coding Rules:**
- Select the code that most accurately describes the procedure
- Include appropriate modifiers
- Consider bundling rules and NCCI edits
- Note if multiple codes are needed

Provide your CPT code suggestions now:"""


# ============================================================================
# 4. CODE VALIDATION PROMPTS
# ============================================================================

CODE_VALIDATION_PROMPT = """You are a medical coding auditor specializing in code validation and compliance.

**Task:** Validate the following medical code assignment against CMS guidelines.

**Assigned Code:**
Code: {code}
Description: {code_description}

**Medical Documentation:**
{documentation}

**Validation Criteria:**
1. Is the code supported by the documentation?
2. Is the code specific enough (no unspecified codes unless necessary)?
3. Are all required characters/digits present?
4. Is laterality correctly coded (if applicable)?
5. Are there any coding rule violations?

**Output Format (JSON):**
{{
  "is_valid": true/false,
  "validation_score": 0.85,
  "issues": [
    {{
      "issue_type": "specificity/documentation/rule_violation/etc",
      "severity": "critical/warning/info",
      "description": "detailed issue description",
      "recommendation": "how to fix the issue"
    }}
  ],
  "alternative_codes": ["suggested better codes if applicable"],
  "compliance_notes": "any CMS compliance considerations"
}}

**Important:**
- Check for documentation support
- Verify code specificity
- Identify any rule violations
- Provide actionable recommendations

Validate the code now:"""


# ============================================================================
# 5. RE-RANKING PROMPTS
# ============================================================================

RERANKING_PROMPT = """You are a medical coding expert performing relevance re-ranking.

**Task:** Re-rank the following candidate codes based on their relevance to the medical entity.

**Medical Entity:**
{medical_entity}

**Clinical Context:**
{context}

**Candidate Codes to Re-rank:**
{candidates}

**Instructions:**
1. Evaluate each candidate code's relevance to the medical entity
2. Consider clinical context and specificity
3. Re-order codes from most to least relevant
4. Assign a relevance score (0.0 to 1.0) to each code

**Output Format (JSON):**
{{
  "reranked_codes": [
    {{
      "code": "medical code",
      "description": "code description",
      "relevance_score": 0.95,
      "reasoning": "why this code ranks at this position"
    }}
  ]
}}

**Re-ranking Criteria:**
- Exact match > Partial match
- Specific code > General code
- Primary diagnosis > Secondary
- Current condition > History of

Re-rank the codes now:"""


# ============================================================================
# 6. SUMMARY GENERATION PROMPTS
# ============================================================================

AUDIT_SUMMARY_PROMPT = """You are a medical coding auditor creating an executive summary.

**Task:** Generate a comprehensive audit summary for the following coding session.

**Coding Session Data:**
{session_data}

**Instructions:**
1. Summarize total diagnoses and procedures coded
2. Highlight high-confidence vs low-confidence codes
3. Note any validation issues or warnings
4. Provide overall accuracy assessment
5. Include actionable recommendations

**Output Format (Markdown):**

# Medical Coding Audit Summary

## Overview
- Total Documents Processed: X
- Total Codes Assigned: X
- Average Confidence Score: X.XX

## Diagnosis Codes (ICD-10-CM)
- Primary Diagnoses: X
- Secondary Diagnoses: X
- High Confidence (>0.85): X
- Needs Review (<0.6): X

## Procedure Codes (CPT/ICD-10-PCS)
- Total Procedures: X
- High Confidence: X
- Needs Review: X

## Validation Results
- Codes Validated: X
- Issues Found: X
  - Critical: X
  - Warnings: X

## Recommendations
1. [Actionable recommendation]
2. [Another recommendation]

## Notes
[Any additional observations]

Generate the audit summary now:"""


# ============================================================================
# 7. ENTITY RECOGNITION PROMPTS
# ============================================================================

MEDICAL_NER_PROMPT = """You are a medical NLP expert performing named entity recognition.

**Task:** Extract all medical entities from the following text.

**Input Text:**
{text}

**Entity Types to Extract:**
- DIAGNOSIS: Diseases, conditions, symptoms
- PROCEDURE: Surgeries, treatments, interventions
- MEDICATION: Drugs, pharmaceuticals
- ANATOMY: Body parts, organs, systems
- LAB_TEST: Laboratory tests, diagnostic tests
- VITAL_SIGN: Blood pressure, temperature, etc.

**Output Format (JSON):**
{{
  "entities": [
    {{
      "text": "exact entity text",
      "type": "entity type",
      "start": start_position,
      "end": end_position,
      "context": "surrounding text for context"
    }}
  ]
}}

Extract all medical entities now:"""


# ============================================================================
# 8. PROMPT TEMPLATES REGISTRY
# ============================================================================

PROMPT_REGISTRY: Dict[str, str] = {
    "diagnosis_extraction": DIAGNOSIS_EXTRACTION_PROMPT,
    "procedure_extraction": PROCEDURE_EXTRACTION_PROMPT,
    "icd10_suggestion": ICD10_CODE_SUGGESTION_PROMPT,
    "cpt_suggestion": CPT_CODE_SUGGESTION_PROMPT,
    "code_validation": CODE_VALIDATION_PROMPT,
    "reranking": RERANKING_PROMPT,
    "audit_summary": AUDIT_SUMMARY_PROMPT,
    "medical_ner": MEDICAL_NER_PROMPT,
}


# ============================================================================
# 9. PROMPT UTILITIES
# ============================================================================

def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a prompt template by name and format it with provided arguments.
    
    Args:
        prompt_name: Name of the prompt template
        **kwargs: Variables to format into the prompt
    
    Returns:
        Formatted prompt string
    
    Raises:
        KeyError: If prompt name not found
        KeyError: If required format variables missing
    """
    if prompt_name not in PROMPT_REGISTRY:
        raise KeyError(f"Prompt '{prompt_name}' not found. Available: {list(PROMPT_REGISTRY.keys())}")
    
    template = PROMPT_REGISTRY[prompt_name]
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing required variable for prompt '{prompt_name}': {e}")


def list_prompts() -> List[str]:
    """List all available prompt templates."""
    return list(PROMPT_REGISTRY.keys())


def get_prompt_variables(prompt_name: str) -> List[str]:
    """
    Extract required variables from a prompt template.
    
    Args:
        prompt_name: Name of the prompt template
    
    Returns:
        List of variable names required by the prompt
    """
    if prompt_name not in PROMPT_REGISTRY:
        raise KeyError(f"Prompt '{prompt_name}' not found")
    
    template = PROMPT_REGISTRY[prompt_name]
    import re
    variables = re.findall(r'\{(\w+)\}', template)
    return list(set(variables))


# ============================================================================
# 10. MAIN BLOCK - Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate prompt templates
    Usage: python config/prompts.py
    """
    print("=" * 80)
    print("MEDICAL CODING SYSTEM - PROMPT TEMPLATES")
    print("=" * 80)
    print()
    
    # 1. List all available prompts
    print("📋 Available Prompt Templates:")
    print("-" * 80)
    for i, prompt_name in enumerate(list_prompts(), 1):
        variables = get_prompt_variables(prompt_name)
        print(f"{i}. {prompt_name}")
        print(f"   Required variables: {', '.join(variables)}")
    print()
    
    # 2. Test Diagnosis Extraction Prompt
    print("🔍 Testing: Diagnosis Extraction Prompt")
    print("-" * 80)
    sample_document = """
    Patient presents with acute chest pain radiating to left arm. 
    History of hypertension and type 2 diabetes mellitus. 
    Suspected myocardial infarction.
    """
    
    diagnosis_prompt = get_prompt(
        "diagnosis_extraction",
        document_text=sample_document.strip()
    )
    print("Sample Input:")
    print(f"  Document: {sample_document.strip()[:100]}...")
    print("\nGenerated Prompt (first 500 chars):")
    print(f"  {diagnosis_prompt[:500]}...")
    print()
    
    # 3. Test Procedure Extraction Prompt
    print("🔍 Testing: Procedure Extraction Prompt")
    print("-" * 80)
    sample_procedure = """
    Performed laparoscopic cholecystectomy via four-port technique.
    Patient underwent right knee arthroscopy with meniscectomy.
    """
    
    procedure_prompt = get_prompt(
        "procedure_extraction",
        document_text=sample_procedure.strip()
    )
    print("Sample Input:")
    print(f"  Document: {sample_procedure.strip()}")
    print("\nGenerated Prompt (first 500 chars):")
    print(f"  {procedure_prompt[:500]}...")
    print()
    
    # 4. Test ICD-10 Code Suggestion Prompt
    print("🔍 Testing: ICD-10 Code Suggestion Prompt")
    print("-" * 80)
    sample_entity = "Type 2 diabetes mellitus with diabetic nephropathy"
    sample_context = "Patient has chronic kidney disease stage 3"
    sample_codes = """
    E11.21 - Type 2 diabetes mellitus with diabetic nephropathy
    E11.22 - Type 2 diabetes mellitus with diabetic chronic kidney disease
    E11.9 - Type 2 diabetes mellitus without complications
    """
    
    icd10_prompt = get_prompt(
        "icd10_suggestion",
        medical_entity=sample_entity,
        context=sample_context,
        retrieved_codes=sample_codes.strip()
    )
    print("Sample Input:")
    print(f"  Entity: {sample_entity}")
    print(f"  Context: {sample_context}")
    print("\nGenerated Prompt (first 600 chars):")
    print(f"  {icd10_prompt[:600]}...")
    print()
    
    # 5. Test Code Validation Prompt
    print("🔍 Testing: Code Validation Prompt")
    print("-" * 80)
    validation_prompt = get_prompt(
        "code_validation",
        code="I10",
        code_description="Essential (primary) hypertension",
        documentation="Patient has elevated blood pressure. BP: 150/95."
    )
    print("Sample Input:")
    print(f"  Code: I10")
    print(f"  Documentation: Patient has elevated blood pressure...")
    print("\nGenerated Prompt (first 500 chars):")
    print(f"  {validation_prompt[:500]}...")
    print()
    
    # 6. Test Re-ranking Prompt
    print("🔍 Testing: Re-ranking Prompt")
    print("-" * 80)
    sample_candidates = """
    1. K64.9 - Unspecified hemorrhoids
    2. K64.0 - First degree hemorrhoids
    3. K64.1 - Second degree hemorrhoids
    4. K64.2 - Third degree hemorrhoids
    """
    
    rerank_prompt = get_prompt(
        "reranking",
        medical_entity="Grade III internal hemorrhoids",
        context="Patient presents with prolapsing hemorrhoids requiring manual reduction",
        candidates=sample_candidates.strip()
    )
    print("Sample Input:")
    print(f"  Entity: Grade III internal hemorrhoids")
    print(f"  Candidates: 4 hemorrhoid codes")
    print("\nGenerated Prompt (first 500 chars):")
    print(f"  {rerank_prompt[:500]}...")
    print()
    
    # 7. Test Variable Extraction
    print("🔧 Testing: Variable Extraction")
    print("-" * 80)
    for prompt_name in ["diagnosis_extraction", "icd10_suggestion", "code_validation"]:
        variables = get_prompt_variables(prompt_name)
        print(f"  {prompt_name}: {variables}")
    print()
    
    # 8. Error Handling Test
    print("⚠️  Testing: Error Handling")
    print("-" * 80)
    try:
        # Test missing prompt
        get_prompt("nonexistent_prompt")
    except KeyError as e:
        print(f"  ✓ Correctly caught missing prompt: {e}")
    
    try:
        # Test missing variable
        get_prompt("diagnosis_extraction")
    except KeyError as e:
        print(f"  ✓ Correctly caught missing variable: {e}")
    print()
    
    # 9. Summary
    print("=" * 80)
    print("✅ PROMPT TEMPLATE TESTING COMPLETE")
    print("=" * 80)
    print(f"\nTotal Prompts: {len(PROMPT_REGISTRY)}")
    print("\n💡 Usage Examples:")
    print("  from config.prompts import get_prompt")
    print("  prompt = get_prompt('diagnosis_extraction', document_text='...')")
    print()