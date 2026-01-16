"""
Speech-to-Text Transcription Module for TrialSync
==================================================

This module provides voice transcription with multiple backend options.

CURRENT IMPLEMENTATION (Demo/Development):
- OpenAI Whisper API: Fast, accurate, easy to set up
- Used for project demonstration and development

FOR HIPAA-COMPLIANT PRODUCTION:
- Local Whisper: Self-hosted, most secure, no data leaves infrastructure
- AWS Transcribe Medical: HIPAA-eligible with signed BAA
- Google Cloud Speech: HIPAA-eligible with signed BAA

Note: OpenAI does not currently offer a BAA, so the Whisper API should
NOT be used with real patient data. For production healthcare systems,
use self-hosted Whisper or cloud services with signed BAAs.

Author: Sai Kiran Billa
"""

import os
import tempfile
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, BinaryIO
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionBackend(Enum):
    """Available transcription backends with HIPAA compliance status."""
    LOCAL_WHISPER = "local_whisper"      # HIPAA: ✅ Self-hosted
    AWS_TRANSCRIBE = "aws_transcribe"    # HIPAA: ✅ With BAA
    GOOGLE_SPEECH = "google_speech"      # HIPAA: ✅ With BAA
    OPENAI_API = "openai_api"            # HIPAA: ❌ No BAA (dev only)


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription with metadata."""
    text: str
    language: str
    confidence: float
    duration_seconds: float
    backend_used: str
    timestamp: str
    hipaa_compliant: bool
    audio_deleted: bool = True  # Audio should be deleted after transcription


class TranscriptionService(ABC):
    """Abstract base class for transcription backends."""
    
    @property
    @abstractmethod
    def is_hipaa_compliant(self) -> bool:
        """Whether this backend can be HIPAA compliant."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file to text."""
        pass
    
    @abstractmethod
    def transcribe_bytes(self, audio_bytes: bytes, format: str = "wav") -> TranscriptionResult:
        """Transcribe audio bytes to text."""
        pass


class LocalWhisperService(TranscriptionService):
    """
    Self-hosted Whisper transcription - HIPAA compliant.
    
    Audio never leaves your infrastructure. This is the most secure
    option for PHI-containing voice data.
    
    Requires: pip install openai-whisper torch
    
    For production deployment:
    - Use faster-whisper for 4x speed improvement
    - Deploy on GPU instances (AWS p3, GCP T4/A100)
    - Can also deploy via Vertex AI or SageMaker
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize local Whisper model.
        
        Args:
            model_size: tiny, base, small, medium, large, large-v3
            device: cpu or cuda
        """
        self.model_size = model_size
        self.device = device
        self._model = None
        
        logger.info(f"Initializing LocalWhisperService (model: {model_size}, device: {device})")
    
    @property
    def is_hipaa_compliant(self) -> bool:
        return True  # Self-hosted, PHI never leaves infrastructure
    
    def _load_model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_size}")
                self._model = whisper.load_model(self.model_size, device=self.device)
                logger.info("Whisper model loaded successfully")
            except ImportError:
                raise ImportError(
                    "whisper not installed. Run: pip install openai-whisper torch"
                )
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio file using local Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResult with text and metadata
        """
        self._load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        start_time = datetime.utcnow()
        
        result = self._model.transcribe(audio_path)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "en"),
            confidence=1.0,  # Whisper doesn't provide per-transcription confidence
            duration_seconds=duration,
            backend_used=f"local_whisper_{self.model_size}",
            timestamp=datetime.utcnow().isoformat(),
            hipaa_compliant=True
        )
    
    def transcribe_bytes(self, audio_bytes: bytes, format: str = "wav") -> TranscriptionResult:
        """Transcribe audio bytes by writing to temp file."""
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = self.transcribe(tmp.name)
            # Temp file auto-deleted
        return result


class AWSTranscribeMedicalService(TranscriptionService):
    """
    AWS Transcribe Medical - HIPAA compliant with BAA.
    
    Requires:
    - AWS account with BAA signed
    - Appropriate IAM permissions
    - pip install boto3
    
    Note: This is a stub implementation. Production would use
    actual AWS SDK calls with proper error handling.
    """
    
    def __init__(self, region: str = "us-east-1", medical_specialty: str = "PRIMARYCARE"):
        """
        Initialize AWS Transcribe Medical.
        
        Args:
            region: AWS region
            medical_specialty: PRIMARYCARE, CARDIOLOGY, NEUROLOGY, etc.
        """
        self.region = region
        self.medical_specialty = medical_specialty
        
        logger.info(f"Initializing AWSTranscribeMedicalService (region: {region})")
    
    @property
    def is_hipaa_compliant(self) -> bool:
        return True  # With signed BAA
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe using AWS Transcribe Medical.
        
        In production, this would:
        1. Upload audio to S3 (encrypted)
        2. Start transcription job
        3. Poll for completion
        4. Retrieve and return results
        5. Delete audio from S3
        """
        # Stub implementation
        logger.warning("AWS Transcribe Medical: Using stub implementation")
        
        return TranscriptionResult(
            text="[AWS Transcribe Medical would process this audio]",
            language="en-US",
            confidence=0.95,
            duration_seconds=0.0,
            backend_used="aws_transcribe_medical",
            timestamp=datetime.utcnow().isoformat(),
            hipaa_compliant=True
        )
    
    def transcribe_bytes(self, audio_bytes: bytes, format: str = "wav") -> TranscriptionResult:
        """Transcribe audio bytes."""
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.transcribe(tmp.name)


class OpenAIWhisperAPIService(TranscriptionService):
    """
    OpenAI Whisper API - Primary transcription service for this project.
    
    This is the DEFAULT backend used in TrialSync for:
    - Fast and accurate transcription
    - Easy setup and development
    - Project demonstrations
    
    ⚠️  IMPORTANT FOR PRODUCTION:
    OpenAI does not currently offer a BAA (Business Associate Agreement).
    For HIPAA-compliant production systems with real patient data, use:
    - LocalWhisperService (self-hosted)
    - AWSTranscribeMedicalService (with signed BAA)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        """
        Initialize OpenAI Whisper API client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Whisper model to use (default: whisper-1)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        logger.info(f"Initialized OpenAI Whisper API (model: {model})")
    
    @property
    def is_hipaa_compliant(self) -> bool:
        return False  # No BAA available from OpenAI
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio file using OpenAI Whisper API.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        client = self._get_client()
        
        logger.info(f"Transcribing audio: {audio_path}")
        start_time = datetime.utcnow()
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="text"
            )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Handle response (can be string or object depending on response_format)
        text = transcript if isinstance(transcript, str) else transcript.text
        
        logger.info(f"Transcription complete: {len(text)} characters in {duration:.2f}s")
        
        return TranscriptionResult(
            text=text.strip(),
            language="en",  # Whisper auto-detects but we default to English
            confidence=1.0,  # Whisper doesn't provide confidence scores
            duration_seconds=duration,
            backend_used=f"openai_whisper_api_{self.model}",
            timestamp=datetime.utcnow().isoformat(),
            hipaa_compliant=False  # Important: OpenAI has no BAA
        )
    
    def transcribe_bytes(self, audio_bytes: bytes, format: str = "wav") -> TranscriptionResult:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Raw audio data
            format: Audio format extension (wav, mp3, m4a, etc.)
            
        Returns:
            TranscriptionResult with transcribed text
        """
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.transcribe(tmp.name)


class TranscriptionFactory:
    """
    Factory for creating transcription service instances.
    
    Default: OpenAI Whisper API (for development/demo)
    Production: Set require_hipaa=True to enforce compliant backends
    """
    
    @staticmethod
    def create(
        backend: TranscriptionBackend = TranscriptionBackend.OPENAI_API,
        require_hipaa: bool = False,
        **kwargs
    ) -> TranscriptionService:
        """
        Create transcription service instance.
        
        Args:
            backend: Which backend to use (default: OpenAI API)
            require_hipaa: If True, raises error for non-compliant backends
            **kwargs: Backend-specific arguments
            
        Returns:
            TranscriptionService instance
            
        Example:
            # Development (default)
            service = TranscriptionFactory.create()
            
            # Production (HIPAA required)
            service = TranscriptionFactory.create(
                backend=TranscriptionBackend.LOCAL_WHISPER,
                require_hipaa=True
            )
        """
        services = {
            TranscriptionBackend.LOCAL_WHISPER: LocalWhisperService,
            TranscriptionBackend.AWS_TRANSCRIBE: AWSTranscribeMedicalService,
            TranscriptionBackend.OPENAI_API: OpenAIWhisperAPIService,
        }
        
        service_class = services.get(backend)
        if not service_class:
            raise ValueError(f"Unknown backend: {backend}")
        
        service = service_class(**kwargs)
        
        if require_hipaa and not service.is_hipaa_compliant:
            raise ValueError(
                f"Backend {backend.value} is not HIPAA compliant. "
                f"For production with PHI, use LOCAL_WHISPER or AWS_TRANSCRIBE. "
                f"Set require_hipaa=False only for development with synthetic data."
            )
        
        return service


def transcribe_voice(
    audio_path: str,
    backend: TranscriptionBackend = TranscriptionBackend.OPENAI_API,
    require_hipaa: bool = False,
    delete_audio: bool = True,
    **kwargs
) -> TranscriptionResult:
    """
    Convenience function for voice transcription.
    
    Uses OpenAI Whisper API by default for easy development.
    Set require_hipaa=True and use LOCAL_WHISPER for production.
    
    Args:
        audio_path: Path to audio file
        backend: Which transcription backend to use (default: OpenAI API)
        require_hipaa: Enforce HIPAA compliance (default: False for dev)
        delete_audio: Securely delete audio after transcription
        **kwargs: Backend-specific arguments
        
    Returns:
        TranscriptionResult with text and metadata
        
    Example:
        # Development (uses OpenAI API)
        >>> result = transcribe_voice("recording.wav")
        >>> print(result.text)
        "I'm 45 years old and I have diabetes..."
        
        # Production (HIPAA compliant)
        >>> result = transcribe_voice(
        ...     "recording.wav",
        ...     backend=TranscriptionBackend.LOCAL_WHISPER,
        ...     require_hipaa=True
        ... )
    """
    service = TranscriptionFactory.create(
        backend=backend,
        require_hipaa=require_hipaa,
        **kwargs
    )
    
    result = service.transcribe(audio_path)
    
    if delete_audio and os.path.exists(audio_path):
        # Secure deletion - overwrite before delete
        file_size = os.path.getsize(audio_path)
        with open(audio_path, "wb") as f:
            f.write(os.urandom(file_size))
        os.remove(audio_path)
        result.audio_deleted = True
        logger.info(f"Audio file securely deleted: {audio_path}")
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("TRANSCRIPTION SERVICE DEMO")
    print("=" * 60)
    
    print("\nAvailable backends:")
    for backend in TranscriptionBackend:
        print(f"  - {backend.value}")
    
    print("\nHIPAA Compliance Status:")
    print("  ❌ OPENAI_API     - No BAA (used in this project for demo)")
    print("  ✅ LOCAL_WHISPER  - Self-hosted, most secure (for production)")
    print("  ✅ AWS_TRANSCRIBE - With signed BAA (for production)")
    print("  ✅ GOOGLE_SPEECH  - With signed BAA (for production)")
    
    print("\nCurrent Project Setup:")
    print("  This project uses OpenAI Whisper API for easy development.")
    print("  For HIPAA-compliant production, switch to LOCAL_WHISPER.")
    
    print("\nExample usage:")
    print("""
    from transcription import transcribe_voice, TranscriptionBackend
    
    # Default: OpenAI Whisper API (for development/demo)
    result = transcribe_voice("recording.wav")
    print(result.text)
    
    # Production: Self-hosted Whisper (HIPAA compliant)
    result = transcribe_voice(
        "patient_audio.wav",
        backend=TranscriptionBackend.LOCAL_WHISPER,
        require_hipaa=True
    )
    """)
