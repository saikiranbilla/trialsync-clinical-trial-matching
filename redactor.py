"""
PHI Redaction Module for TrialSync
===================================

This module provides HIPAA-compliant de-identification of clinical text
using BERT-based Named Entity Recognition (NER) models.

Key Features:
- Context-aware PHI detection using transformer models
- Preserves clinical data (conditions, medications) while redacting identifiers
- Confidence-based human-in-the-loop flagging
- Audit trail logging for 21 CFR Part 11 compliance

"""

import re
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RedactedEntity:
    """Represents a single redacted PHI entity with audit metadata."""
    token: str
    original_value: str
    entity_type: str
    confidence: float
    start_position: int
    end_position: int
    model_version: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    flagged_for_review: bool = False


@dataclass
class RedactionResult:
    """Complete result of PHI redaction with audit trail."""
    original_text_hash: str  # SHA-256 hash, not the actual text
    redacted_text: str
    entities_redacted: list[RedactedEntity]
    entities_flagged: list[RedactedEntity]
    clinical_entities_preserved: list[dict]
    model_used: str
    processing_timestamp: str
    confidence_threshold: float


class ClinicalPHIRedactor:
    """
    HIPAA-compliant PHI redaction using BERT-based NER.
    
    This class implements context-aware de-identification that:
    - Detects PHI (names, dates, locations, IDs) using clinical NER models
    - Preserves clinical data (conditions, medications, procedures)
    - Provides confidence scores for human-in-the-loop review
    - Maintains audit trails for regulatory compliance
    
    Example:
        >>> redactor = ClinicalPHIRedactor()
        >>> result = redactor.redact("John Smith, 45, has diabetes")
        >>> print(result.redacted_text)
        "[NAME_001], [AGE_001], has diabetes"
    """
    
    # PHI categories to redact (HIPAA Safe Harbor identifiers)
    PHI_CATEGORIES = {
        'NAME', 'PATIENT', 'DOCTOR', 'USERNAME',
        'DATE', 'AGE', 'DOB',
        'LOCATION', 'HOSPITAL', 'ORGANIZATION', 'CITY', 'STATE', 'COUNTRY',
        'STREET', 'ZIP', 'ADDRESS',
        'PHONE', 'FAX', 'EMAIL', 'URL',
        'ID', 'SSN', 'MRN', 'ACCOUNT', 'LICENSE', 'VEHICLE', 'DEVICE',
        'BIOMETRIC', 'PHOTO',
        'PER',  # Generic person tag from some models
        'LOC',  # Generic location tag
        'ORG',  # Generic organization tag
    }
    
    # Clinical categories to PRESERVE (not PHI)
    CLINICAL_CATEGORIES = {
        'CONDITION', 'DISEASE', 'PROBLEM', 'DIAGNOSIS',
        'MEDICATION', 'DRUG', 'TREATMENT',
        'PROCEDURE', 'TEST', 'SURGERY',
        'LAB', 'VITAL', 'MEASUREMENT',
        'ANATOMY', 'BODY_PART',
        'DOSAGE', 'FREQUENCY', 'DURATION',
    }
    
    # Supported model configurations
    SUPPORTED_MODELS = {
        'deid_roberta': 'obi/deid_roberta_i2b2',
        'clinical_bert': 'emilyalsentzer/Bio_ClinicalBERT',
        'biobert': 'dmis-lab/biobert-v1.1',
        'default': 'dslim/bert-base-NER',  # Fallback for demo
    }
    
    def __init__(
        self,
        model_name: str = 'default',
        confidence_threshold: float = 0.85,
        device: int = -1  # -1 for CPU, 0+ for GPU
    ):
        """
        Initialize the PHI redactor with specified model.
        
        Args:
            model_name: Key from SUPPORTED_MODELS or HuggingFace model path
            confidence_threshold: Minimum confidence for auto-redaction (0.0-1.0)
            device: Device for inference (-1=CPU, 0+=GPU index)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            self.model_path = self.SUPPORTED_MODELS[model_name]
        else:
            self.model_path = model_name
            
        self.model_version = f"{self.model_path}@{datetime.utcnow().strftime('%Y%m%d')}"
        
        # Initialize model (lazy loading)
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        
        logger.info(f"Initialized ClinicalPHIRedactor with model: {self.model_path}")
    
    def _load_pipeline(self):
        """Lazy load the NER pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading NER model: {self.model_path}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self._model = AutoModelForTokenClassification.from_pretrained(self.model_path)
                self._pipeline = pipeline(
                    "ner",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    aggregation_strategy="simple",
                    device=self.device
                )
                logger.info("NER model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_path}: {e}")
                raise
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type labels across different models."""
        # Remove B-, I- prefixes (BIO tagging)
        normalized = re.sub(r'^[BI]-', '', entity_type.upper())
        
        # Map common variations
        mappings = {
            'PER': 'NAME',
            'PERSON': 'NAME',
            'LOC': 'LOCATION',
            'GPE': 'LOCATION',  # Geo-Political Entity
            'ORG': 'ORGANIZATION',
            'MISC': 'OTHER',
        }
        
        return mappings.get(normalized, normalized)
    
    def _is_phi_category(self, entity_type: str) -> bool:
        """Check if entity type is a PHI category requiring redaction."""
        normalized = self._normalize_entity_type(entity_type)
        return normalized in self.PHI_CATEGORIES
    
    def _is_clinical_category(self, entity_type: str) -> bool:
        """Check if entity type is clinical data to preserve."""
        normalized = self._normalize_entity_type(entity_type)
        return normalized in self.CLINICAL_CATEGORIES
    
    def _generate_token(self, entity_type: str, counter: int) -> str:
        """Generate a redaction token for an entity."""
        normalized = self._normalize_entity_type(entity_type)
        return f"[{normalized}_{counter:03d}]"
    
    def _hash_text(self, text: str) -> str:
        """Generate SHA-256 hash of text for audit trail (not storing actual PHI)."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def redact(self, text: str) -> RedactionResult:
        """
        Redact PHI from clinical text while preserving clinical information.
        
        This method:
        1. Runs NER to detect all entities
        2. Classifies entities as PHI (redact) or clinical (preserve)
        3. Replaces PHI with tokens
        4. Flags low-confidence detections for human review
        5. Creates audit trail
        
        Args:
            text: Raw clinical text potentially containing PHI
            
        Returns:
            RedactionResult with redacted text, token mappings, and audit info
        """
        self._load_pipeline()
        
        # Run NER
        entities = self._pipeline(text)
        
        # Separate PHI from clinical entities
        phi_entities = []
        clinical_entities = []
        
        for entity in entities:
            entity_type = self._normalize_entity_type(entity['entity_group'])
            
            if self._is_phi_category(entity['entity_group']):
                phi_entities.append(entity)
            elif self._is_clinical_category(entity['entity_group']):
                clinical_entities.append({
                    'text': entity['word'],
                    'type': entity_type,
                    'confidence': entity['score']
                })
        
        # Sort PHI entities by position (reverse for replacement)
        phi_entities_sorted = sorted(phi_entities, key=lambda x: x['start'], reverse=True)
        
        # Perform redaction
        redacted_text = text
        entities_redacted = []
        entities_flagged = []
        counters = {}
        
        for entity in phi_entities_sorted:
            entity_type = self._normalize_entity_type(entity['entity_group'])
            
            # Generate token
            if entity_type not in counters:
                counters[entity_type] = 0
            counters[entity_type] += 1
            token = self._generate_token(entity_type, counters[entity_type])
            
            # Create redacted entity record
            redacted_entity = RedactedEntity(
                token=token,
                original_value=entity['word'],
                entity_type=entity_type,
                confidence=entity['score'],
                start_position=entity['start'],
                end_position=entity['end'],
                model_version=self.model_version,
                flagged_for_review=entity['score'] < self.confidence_threshold
            )
            
            # Categorize based on confidence
            if entity['score'] >= self.confidence_threshold:
                entities_redacted.append(redacted_entity)
                # Perform replacement
                redacted_text = (
                    redacted_text[:entity['start']] +
                    token +
                    redacted_text[entity['end']:]
                )
            else:
                # Flag for human review, still redact but mark it
                entities_flagged.append(redacted_entity)
                redacted_text = (
                    redacted_text[:entity['start']] +
                    f"{token}[REVIEW]" +
                    redacted_text[entity['end']:]
                )
                logger.warning(
                    f"Low confidence ({entity['score']:.2f}) for entity: "
                    f"'{entity['word']}' as {entity_type}"
                )
        
        # Create result with audit trail
        result = RedactionResult(
            original_text_hash=self._hash_text(text),
            redacted_text=redacted_text,
            entities_redacted=entities_redacted,
            entities_flagged=entities_flagged,
            clinical_entities_preserved=clinical_entities,
            model_used=self.model_version,
            processing_timestamp=datetime.utcnow().isoformat(),
            confidence_threshold=self.confidence_threshold
        )
        
        logger.info(
            f"Redaction complete: {len(entities_redacted)} entities redacted, "
            f"{len(entities_flagged)} flagged for review, "
            f"{len(clinical_entities)} clinical entities preserved"
        )
        
        return result
    
    def get_token_map(self, result: RedactionResult) -> dict:
        """
        Extract secure token mapping from redaction result.
        
        WARNING: This mapping contains original PHI values.
        Store encrypted with appropriate access controls.
        
        Args:
            result: RedactionResult from redact() method
            
        Returns:
            Dictionary mapping tokens to original values
        """
        token_map = {}
        
        for entity in result.entities_redacted + result.entities_flagged:
            token_map[entity.token] = {
                'original': entity.original_value,
                'type': entity.entity_type,
                'confidence': entity.confidence,
                'flagged': entity.flagged_for_review
            }
        
        return token_map
    
    def to_audit_log(self, result: RedactionResult) -> dict:
        """
        Generate audit log entry for 21 CFR Part 11 compliance.
        
        This log does NOT contain original PHI values, only metadata
        about the redaction process for regulatory auditing.
        
        Args:
            result: RedactionResult from redact() method
            
        Returns:
            Audit log dictionary safe for long-term storage
        """
        return {
            'timestamp': result.processing_timestamp,
            'original_text_hash': result.original_text_hash,
            'model_used': result.model_used,
            'confidence_threshold': result.confidence_threshold,
            'entities_redacted_count': len(result.entities_redacted),
            'entities_flagged_count': len(result.entities_flagged),
            'clinical_entities_preserved_count': len(result.clinical_entities_preserved),
            'entity_types_redacted': list(set(e.entity_type for e in result.entities_redacted)),
            'flagged_entity_details': [
                {
                    'token': e.token,
                    'type': e.entity_type,
                    'confidence': e.confidence
                }
                for e in result.entities_flagged
            ]
        }


class TokenVault:
    """
    Secure storage for PHI token mappings.
    
    In production, this would integrate with:
    - AWS KMS / Azure Key Vault for encryption
    - Salesforce Shield for field-level encryption
    - HSM for key management
    
    This implementation provides the interface pattern.
    """
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize token vault.
        
        Args:
            encryption_key: Encryption key (in production, from KMS)
        """
        self._storage = {}  # In production: encrypted database
        self._encryption_key = encryption_key
        logger.info("TokenVault initialized (demo mode - use KMS in production)")
    
    def store(self, session_id: str, token_map: dict) -> str:
        """
        Store token mapping securely.
        
        Args:
            session_id: Unique identifier for the redaction session
            token_map: Mapping from tokens to original PHI
            
        Returns:
            Storage key for retrieval
        """
        storage_key = hashlib.sha256(
            f"{session_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # In production: encrypt before storage
        self._storage[storage_key] = {
            'session_id': session_id,
            'token_map': token_map,
            'created_at': datetime.utcnow().isoformat(),
            'accessed_count': 0
        }
        
        logger.info(f"Token map stored with key: {storage_key}")
        return storage_key
    
    def retrieve(self, storage_key: str) -> Optional[dict]:
        """
        Retrieve token mapping (with access logging).
        
        Args:
            storage_key: Key returned from store()
            
        Returns:
            Token mapping or None if not found
        """
        if storage_key not in self._storage:
            logger.warning(f"Token map not found: {storage_key}")
            return None
        
        entry = self._storage[storage_key]
        entry['accessed_count'] += 1
        entry['last_accessed'] = datetime.utcnow().isoformat()
        
        logger.info(f"Token map retrieved: {storage_key} (access #{entry['accessed_count']})")
        return entry['token_map']
    
    def delete(self, storage_key: str) -> bool:
        """
        Securely delete token mapping.
        
        Args:
            storage_key: Key to delete
            
        Returns:
            True if deleted, False if not found
        """
        if storage_key in self._storage:
            del self._storage[storage_key]
            logger.info(f"Token map deleted: {storage_key}")
            return True
        return False


# Convenience function for simple usage
def redact_phi(text: str, confidence_threshold: float = 0.85) -> tuple[str, dict]:
    """
    Simple interface for PHI redaction.
    
    Args:
        text: Clinical text to redact
        confidence_threshold: Minimum confidence for auto-redaction
        
    Returns:
        Tuple of (redacted_text, token_map)
        
    Example:
        >>> redacted, tokens = redact_phi("John Smith has diabetes")
        >>> print(redacted)
        "[NAME_001] has diabetes"
    """
    redactor = ClinicalPHIRedactor(confidence_threshold=confidence_threshold)
    result = redactor.redact(text)
    token_map = redactor.get_token_map(result)
    return result.redacted_text, token_map


if __name__ == "__main__":
    # Demo usage
    sample_text = """
    John Smith, a 45-year-old male from Chicago, presented to 
    Mayo Clinic on March 15, 2024. He has Type 2 diabetes mellitus 
    and is currently taking metformin 500mg twice daily. 
    Contact number: 555-123-4567. MRN: 12345678.
    """
    
    print("=" * 60)
    print("PHI REDACTION DEMO")
    print("=" * 60)
    print(f"\nOriginal text:\n{sample_text}")
    
    # Initialize redactor
    redactor = ClinicalPHIRedactor(
        model_name='default',  # Use 'deid_roberta' in production
        confidence_threshold=0.85
    )
    
    # Perform redaction
    result = redactor.redact(sample_text)
    
    print(f"\nRedacted text:\n{result.redacted_text}")
    
    print("\n" + "=" * 60)
    print("TOKEN MAPPING (Store securely!)")
    print("=" * 60)
    token_map = redactor.get_token_map(result)
    for token, info in token_map.items():
        flag = " [NEEDS REVIEW]" if info['flagged'] else ""
        print(f"  {token}: '{info['original']}' ({info['type']}, conf: {info['confidence']:.2f}){flag}")
    
    print("\n" + "=" * 60)
    print("CLINICAL DATA PRESERVED")
    print("=" * 60)
    for entity in result.clinical_entities_preserved:
        print(f"  {entity['text']} ({entity['type']}, conf: {entity['confidence']:.2f})")
    
    print("\n" + "=" * 60)
    print("AUDIT LOG (Safe for long-term storage)")
    print("=" * 60)
    audit_log = redactor.to_audit_log(result)
    print(json.dumps(audit_log, indent=2))
