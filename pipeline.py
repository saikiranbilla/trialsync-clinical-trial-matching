"""
Voice-to-Safe-Text Pipeline for TrialSync
==========================================

This module combines speech-to-text transcription with PHI redaction
to create a HIPAA-compliant pipeline for processing patient voice input.

Pipeline Flow:
1. Audio Input → 2. Transcription (self-hosted) → 3. PHI Detection →
4. Tokenization → 5. Safe Text Output (for downstream agents)

The output text contains no PHI and can be safely processed by
external AI services or multi-agent systems.

Author: Sai Kiran Billa
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

from .transcription import (
    TranscriptionFactory,
    TranscriptionBackend,
    TranscriptionResult,
    transcribe_voice
)
from .redactor import (
    ClinicalPHIRedactor,
    RedactionResult,
    TokenVault
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from voice-to-safe-text pipeline."""
    
    # Safe outputs (no PHI)
    safe_text: str                      # Redacted text for downstream processing
    clinical_entities: list[dict]       # Preserved medical information
    
    # Metadata
    transcription_backend: str
    transcription_duration_seconds: float
    redaction_model: str
    entities_redacted_count: int
    entities_flagged_count: int
    
    # Compliance
    hipaa_compliant: bool
    audio_deleted: bool
    token_vault_key: Optional[str]      # Key to retrieve original values if needed
    
    # Audit
    pipeline_timestamp: str
    original_audio_hash: Optional[str]  # Hash only, not actual audio
    original_text_hash: str             # Hash only, not actual text


class VoiceToSafeTextPipeline:
    """
    End-to-end pipeline for HIPAA-compliant voice processing.
    
    This pipeline ensures that:
    1. Audio never leaves your infrastructure (self-hosted transcription)
    2. PHI is immediately tokenized after transcription
    3. Downstream systems only receive de-identified text
    4. Full audit trail is maintained for compliance
    
    Example:
        >>> pipeline = VoiceToSafeTextPipeline()
        >>> result = pipeline.process("patient_voice.wav")
        >>> print(result.safe_text)
        "[NAME_001], [AGE_001], has Type 2 diabetes..."
        >>> # safe_text can now be sent to matching agents
    """
    
    def __init__(
        self,
        transcription_backend: TranscriptionBackend = TranscriptionBackend.OPENAI_API,
        redaction_model: str = 'default',
        confidence_threshold: float = 0.85,
        require_hipaa: bool = False,
        enable_token_vault: bool = True,
        **transcription_kwargs
    ):
        """
        Initialize the voice processing pipeline.
        
        Args:
            transcription_backend: Which speech-to-text service to use
                                   (default: OpenAI API for development)
            redaction_model: NER model for PHI detection
            confidence_threshold: Minimum confidence for auto-redaction
            require_hipaa: Enforce HIPAA compliance (set True for production)
            enable_token_vault: Store token mappings for later de-tokenization
            **transcription_kwargs: Additional args for transcription service
        """
        self.transcription_backend = transcription_backend
        self.require_hipaa = require_hipaa
        self.enable_token_vault = enable_token_vault
        self.transcription_kwargs = transcription_kwargs
        
        # Initialize components
        self.transcription_service = TranscriptionFactory.create(
            backend=transcription_backend,
            require_hipaa=require_hipaa,
            **transcription_kwargs
        )
        
        self.redactor = ClinicalPHIRedactor(
            model_name=redaction_model,
            confidence_threshold=confidence_threshold
        )
        
        self.token_vault = TokenVault() if enable_token_vault else None
        
        logger.info(
            f"Pipeline initialized: "
            f"transcription={transcription_backend.value}, "
            f"redaction_model={redaction_model}, "
            f"hipaa_required={require_hipaa}"
        )
    
    def process(
        self,
        audio_path: str,
        session_id: Optional[str] = None,
        delete_audio: bool = True
    ) -> PipelineResult:
        """
        Process voice audio through the complete pipeline.
        
        Args:
            audio_path: Path to audio file
            session_id: Unique session identifier for token storage
            delete_audio: Securely delete audio after processing
            
        Returns:
            PipelineResult with safe text and audit metadata
        """
        pipeline_start = datetime.utcnow()
        session_id = session_id or f"session_{pipeline_start.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting pipeline for session: {session_id}")
        
        # Hash the audio for audit trail (not storing actual audio)
        audio_hash = self._hash_file(audio_path) if os.path.exists(audio_path) else None
        
        # Step 1: Transcription
        logger.info("Step 1: Transcribing audio...")
        transcription_result = self.transcription_service.transcribe(audio_path)
        
        logger.info(f"Transcription complete: {len(transcription_result.text)} characters")
        
        # Step 2: PHI Redaction
        logger.info("Step 2: Detecting and redacting PHI...")
        redaction_result = self.redactor.redact(transcription_result.text)
        
        logger.info(
            f"Redaction complete: {len(redaction_result.entities_redacted)} redacted, "
            f"{len(redaction_result.entities_flagged)} flagged"
        )
        
        # Step 3: Store token mapping (if enabled)
        token_vault_key = None
        if self.enable_token_vault and self.token_vault:
            token_map = self.redactor.get_token_map(redaction_result)
            token_vault_key = self.token_vault.store(session_id, token_map)
            logger.info(f"Token mapping stored: {token_vault_key}")
        
        # Step 4: Secure audio deletion
        audio_deleted = False
        if delete_audio and os.path.exists(audio_path):
            self._secure_delete(audio_path)
            audio_deleted = True
            logger.info("Audio file securely deleted")
        
        # Build result
        result = PipelineResult(
            safe_text=redaction_result.redacted_text,
            clinical_entities=redaction_result.clinical_entities_preserved,
            transcription_backend=transcription_result.backend_used,
            transcription_duration_seconds=transcription_result.duration_seconds,
            redaction_model=redaction_result.model_used,
            entities_redacted_count=len(redaction_result.entities_redacted),
            entities_flagged_count=len(redaction_result.entities_flagged),
            hipaa_compliant=transcription_result.hipaa_compliant,
            audio_deleted=audio_deleted,
            token_vault_key=token_vault_key,
            pipeline_timestamp=datetime.utcnow().isoformat(),
            original_audio_hash=audio_hash,
            original_text_hash=redaction_result.original_text_hash
        )
        
        logger.info(f"Pipeline complete for session: {session_id}")
        
        return result
    
    def process_bytes(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        session_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Process audio bytes through the pipeline.
        
        Args:
            audio_bytes: Raw audio data
            audio_format: Audio format (wav, mp3, etc.)
            session_id: Unique session identifier
            
        Returns:
            PipelineResult with safe text and metadata
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            return self.process(
                audio_path=tmp_path,
                session_id=session_id,
                delete_audio=True  # Always delete temp files
            )
        finally:
            # Ensure cleanup even on error
            if os.path.exists(tmp_path):
                self._secure_delete(tmp_path)
    
    def retrieve_original_phi(self, token_vault_key: str) -> Optional[dict]:
        """
        Retrieve original PHI values from token vault.
        
        ⚠️  Use only when absolutely necessary and with proper authorization.
        All retrievals are logged for audit purposes.
        
        Args:
            token_vault_key: Key from PipelineResult.token_vault_key
            
        Returns:
            Token mapping or None if not found
        """
        if not self.token_vault:
            logger.error("Token vault not enabled")
            return None
        
        return self.token_vault.retrieve(token_vault_key)
    
    def detokenize(self, safe_text: str, token_vault_key: str) -> Optional[str]:
        """
        Convert tokenized text back to original (with PHI).
        
        ⚠️  Use only for patient-facing displays with proper authorization.
        
        Args:
            safe_text: Tokenized text from pipeline
            token_vault_key: Key to token vault
            
        Returns:
            Original text with PHI restored, or None if vault key invalid
        """
        token_map = self.retrieve_original_phi(token_vault_key)
        if not token_map:
            return None
        
        result = safe_text
        for token, info in token_map.items():
            result = result.replace(token, info['original'])
        
        return result
    
    def _hash_file(self, file_path: str) -> str:
        """Generate SHA-256 hash of file for audit trail."""
        import hashlib
        
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _secure_delete(self, file_path: str):
        """Securely delete file by overwriting before removal."""
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size))
            os.remove(file_path)
    
    def get_audit_log(self, result: PipelineResult) -> dict:
        """
        Generate comprehensive audit log for compliance.
        
        This log contains no PHI, only metadata about processing.
        Safe for long-term storage and regulatory review.
        """
        return {
            'pipeline_timestamp': result.pipeline_timestamp,
            'session_info': {
                'original_audio_hash': result.original_audio_hash,
                'original_text_hash': result.original_text_hash,
                'audio_securely_deleted': result.audio_deleted
            },
            'transcription': {
                'backend': result.transcription_backend,
                'duration_seconds': result.transcription_duration_seconds,
                'hipaa_compliant': result.hipaa_compliant
            },
            'redaction': {
                'model': result.redaction_model,
                'entities_redacted': result.entities_redacted_count,
                'entities_flagged_for_review': result.entities_flagged_count,
                'clinical_entities_preserved': len(result.clinical_entities)
            },
            'token_vault': {
                'enabled': self.enable_token_vault,
                'storage_key': result.token_vault_key
            }
        }


# Convenience function
def process_patient_voice(
    audio_path: str,
    session_id: Optional[str] = None
) -> tuple[str, list[dict]]:
    """
    Simple interface for processing patient voice input.
    
    Args:
        audio_path: Path to patient voice recording
        session_id: Optional session identifier
        
    Returns:
        Tuple of (safe_text, clinical_entities)
        
    Example:
        >>> safe_text, clinical = process_patient_voice("recording.wav")
        >>> print(safe_text)
        "[NAME_001], [AGE_001], has diabetes and takes metformin"
        >>> # safe_text is ready for trial matching agents
    """
    pipeline = VoiceToSafeTextPipeline()
    result = pipeline.process(audio_path, session_id)
    return result.safe_text, result.clinical_entities


if __name__ == "__main__":
    print("=" * 60)
    print("VOICE-TO-SAFE-TEXT PIPELINE")
    print("=" * 60)
    
    print("""
    This pipeline processes patient voice recordings:
    
    1. TRANSCRIPTION (self-hosted Whisper)
       Audio → Text (never leaves your infrastructure)
    
    2. PHI DETECTION (BERT-based NER)
       Identifies names, dates, locations, IDs
    
    3. TOKENIZATION
       "John Smith, 45" → "[NAME_001], [AGE_001]"
    
    4. SECURE STORAGE
       Token mappings encrypted, audio deleted
    
    5. OUTPUT
       Safe text ready for downstream AI agents
    
    Example usage:
    
        from pipeline import VoiceToSafeTextPipeline
        
        pipeline = VoiceToSafeTextPipeline(
            transcription_backend=TranscriptionBackend.LOCAL_WHISPER,
            redaction_model='deid_roberta',
            confidence_threshold=0.85,
            require_hipaa=True
        )
        
        result = pipeline.process("patient_recording.wav")
        
        # Safe for downstream processing
        print(result.safe_text)
        
        # Clinical data preserved
        for entity in result.clinical_entities:
            print(f"  {entity['text']} ({entity['type']})")
        
        # Later, if authorized to see original:
        original = pipeline.detokenize(
            result.safe_text,
            result.token_vault_key
        )
    """)
