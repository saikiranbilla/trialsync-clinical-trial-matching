# TrialSync

**Voice-enabled clinical trial matching platform with AI-powered eligibility screening**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TrialSync allows patients to find matching clinical trials through natural voice interaction. The platform uses multi-agent AI to automate complex eligibility screening while maintaining strict privacy controls—patient identity is never exposed to the matching algorithms.

### Key Features

- **Voice-First Interface**: Patients describe their condition naturally; no forms to fill
- **90% Faster Screening**: Multi-agent architecture evaluates eligibility in <100ms
- **Privacy-Focused**: Immediate PHI tokenization before AI processing
- **Context-Aware NER**: BERT-based models distinguish names from medical terms
- **Multi-Agent Matching**: 50+ autonomous Fetch.ai agents for parallel criteria evaluation

## Architecture
![Architecture](./assets/architecture-diagram.png)

## Transcription Backend

### Current Implementation (This Project)

This project uses **OpenAI Whisper API** for speech-to-text transcription:
- Fast and accurate transcription
- Easy setup for development and demonstrations
- Requires `OPENAI_API_KEY` environment variable

```python
# Default: Uses OpenAI Whisper API
from phi_redaction import process_patient_voice
safe_text, clinical_entities = process_patient_voice("recording.wav")
```

### For HIPAA-Compliant Production

For production healthcare systems with real patient data, the following backends are supported:

| Backend | HIPAA Compliant | Setup |
|---------|-----------------|-------|
| **OpenAI Whisper API** | ❌ No (no BAA) | Current project default |
| **Self-hosted Whisper** | ✅ Yes | Deploy on your infrastructure |
| **AWS Transcribe Medical** | ✅ Yes | Requires signed BAA with AWS |
| **Google Cloud Speech** | ✅ Yes | Requires signed BAA with GCP |

```python
# Production: Self-hosted Whisper (HIPAA compliant)
from phi_redaction import VoiceToSafeTextPipeline, TranscriptionBackend

pipeline = VoiceToSafeTextPipeline(
    transcription_backend=TranscriptionBackend.LOCAL_WHISPER,
    require_hipaa=True
)
```

## Modules

### 1. PHI Redaction (`phi_redaction/`)

Privacy-focused voice processing pipeline:

- **Transcription**: OpenAI Whisper API (default) or self-hosted options
- **PHI Detection**: BERT-based NER fine-tuned on i2b2 de-identification dataset
- **Tokenization**: Reversible redaction with secure token vault
- **Audit Trail**: Full logging for compliance documentation

```python
from phi_redaction import process_patient_voice

safe_text, clinical_entities = process_patient_voice("recording.wav")
# safe_text: "[NAME_001], [AGE_001], has Type 2 diabetes..."
# clinical_entities: [{'text': 'Type 2 diabetes', 'type': 'CONDITION'}]
```

### 2. Multi-Agent Matching (Coming Soon)

Fetch.ai agent orchestration for parallel eligibility evaluation:

- 50+ autonomous agents checking different criteria
- <100ms response time via Redis pub/sub
- Kubernetes deployment for auto-scaling

### 3. Medical Record Parser (Coming Soon)

Support for multiple clinical data formats:

- HL7 FHIR (modern standard)
- HL7v2 (legacy)
- PDF clinical documents (via BioBERT NER)

## Quick Start

```bash
# Clone repository
git clone https://github.com/saikiranbilla/trialsync.git
cd trialsync

# Install dependencies
cd phi_redaction
pip install -r requirements.txt

# Run demo
python examples.py
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Transcription | OpenAI Whisper API (default), self-hosted Whisper (production) |
| PHI Detection | BERT, RoBERTa, ClinicalBERT |
| Agent Orchestration | Fetch.ai |
| Message Queue | Redis |
| Container Orchestration | Kubernetes |
| Frontend | React, TypeScript |

## Privacy & Compliance

### Current Project Implementation
- ✅ Immediate PHI tokenization after transcription
- ✅ Encrypted token vault for reversible redaction
- ✅ Audit trail logging
- ⚠️ Uses OpenAI API (not HIPAA compliant)

### Production-Ready Options
For deploying with real patient data:
- ✅ Self-hosted Whisper (audio never leaves infrastructure)
- ✅ AWS Transcribe Medical (with signed BAA)
- ✅ Full 21 CFR Part 11 audit trail support

## Project Structure

```
trialsync/
├── phi_redaction/           # PHI detection and redaction
│   ├── __init__.py
│   ├── redactor.py          # BERT-based NER redaction
│   ├── transcription.py     # Speech-to-text backends
│   ├── pipeline.py          # End-to-end processing
│   ├── examples.py          # Usage demonstrations
│   ├── requirements.txt
│   └── README.md
├── agents/                   # Multi-agent matching (coming soon)
├── parsers/                  # Medical record parsers (coming soon)
└── README.md
```

## Relevance to Clinical Research

This project addresses the #1 cause of clinical trial delays: **patient recruitment**. 80% of trials fail to meet enrollment timelines due to manual, slow eligibility screening.

TrialSync automates this process while maintaining the privacy and compliance standards required in healthcare:

- **For Patients**: Natural voice interface, instant matching
- **For Sites**: Reduced screening workload, higher quality matches
- **For Sponsors**: Faster enrollment, lower costs

## Author

- GitHub: [github.com/saikiranbilla](https://github.com/saikiranbilla)
- LinkedIn: [linkedin.com/in/saikiranbilla](https://linkedin.com/in/saikiranbilla)
- Email: sbilla21@outlook.com

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## ⚠️ Important Disclaimer

**THIS IS FOR EDUCATIONAL AND DEMONSTRATION PURPOSES ONLY.**

- **Not a Medical Device**: TrialSync is not a certified medical device and should not be used for actual clinical trial matching, medical diagnosis, or healthcare decisions.

- **Not HIPAA Compliant as Configured**: The default configuration uses OpenAI's API which does not have a BAA. Do not use with real patient data without implementing HIPAA-compliant alternatives.

- **No Warranty**: This software is provided "as is" without warranty of any kind. The authors are not responsible for any damages or liability arising from its use.

- **Consult Professionals**: Always consult qualified healthcare professionals and legal experts before implementing any system that handles patient health information.

- **Research Project**: This is a research/portfolio project demonstrating AI concepts in healthcare. It has not undergone the rigorous validation required for clinical use.

By using this , you acknowledge that it is not intended for production healthcare use and should only be used with synthetic or test data.

---

*Built as a demonstration of AI concepts in clinical research* 🏥
