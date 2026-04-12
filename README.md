# Deep-Guard Agent

An Interactive Audio-Visual Agent for Deepfake Detection and Cognitive Intervention.

**Team**: Qiming Li · Yiting Wang · Yawen Ou

## Overview

Deep-Guard Agent is a collaborative human-AI system that treats deepfake detection as a joint task. Instead of providing opaque "fake/real" scores, it focuses on **audio-visual inconsistencies** as the detection core and provides **transparent reasoning** to foster critical thinking.

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Detection Backend                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Visual-Lip   │  │    Audio-    │  │  Cross-Modal │  │
│  │   Encoder     │  │ Articulatory │  │    Fusion    │  │
│  │  (MediaPipe)  │  │   Encoder    │  │ (Transformer)│  │
│  │              │  │  (Wav2Vec2)  │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └─────────────────┼─────────────────┘          │
│                           ▼                             │
│              Discrepancy Scores + Heatmaps              │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     LLM Reasoner                        │
│         Synthesize → Explain → Ground                   │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 Human-Agent Interface                    │
│     Overlay  │  Advice  │  Legal Report Export          │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Visual-Lip Encoder** — Frame-by-frame tracking of labial movements using MediaPipe Face Mesh. Captures subtle shape changes for bilabial sounds ('b', 'p', 'm') that AI synthesis routinely distorts.

2. **Audio-Articulatory Encoder** — Self-supervised module (Wav2Vec2) that maps audio signals to predicted articulatory positions of the vocal tract. Rooted in physical acoustics, not pixel patterns.

3. **Cross-Modal Fusion** — Transformer-based comparator aligning visual lip features with predicted articulatory positions frame-by-frame, generating discrepancy heatmaps.

4. **LLM Reasoner** — Aggregates detection signals into plain-language explanations grounded in physical evidence.

5. **Human-Agent Interface** — Gradio-based web UI with real-time overlays, context-specific guidance, and exportable forensic reports.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/deepguard-agent.git
cd deepguard-agent

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.10+
- FFmpeg (for audio/video processing)
- OpenAI API key (for the LLM Reasoner)

```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Web Interface

```bash
deepguard
# or
python -m deepguard.app
```

Then open http://localhost:7860 in your browser.

### Programmatic Usage

```python
from deepguard.detection import VisualLipEncoder, AudioArticulatoryEncoder, CrossModalFusion
from deepguard.reasoning import LLMReasoner
from deepguard.interface import ReportGenerator

# Initialize components
visual = VisualLipEncoder()
audio = AudioArticulatoryEncoder()
fusion = CrossModalFusion()
reasoner = LLMReasoner()
reporter = ReportGenerator()

# Run detection
lip_features = visual.process_video("suspect_video.mp4")
audio_features = audio.process("suspect_video.mp4")
fusion_result = fusion.analyze(audio_features.embeddings, visual_matrix)

# Generate report
analysis = reasoner.analyze(fusion_result)
print(reporter.to_text(fusion_result, analysis))
```

## Configuration

Edit `configs/default.yaml` to customize detection parameters, LLM settings, and interface options.

## Testing

```bash
pytest
```

## Project Structure

```
deepguard/
├── detection/
│   ├── visual_encoder.py    # Visual-Lip Encoder (MediaPipe)
│   ├── audio_encoder.py     # Audio-Articulatory Encoder (Wav2Vec2)
│   └── fusion.py            # Cross-Modal Fusion
├── reasoning/
│   └── llm_reasoner.py      # LLM-powered analysis
├── interface/
│   ├── overlay.py           # Video annotation overlay
│   └── report.py            # Forensic report generation
├── utils/
│   └── video.py             # Video processing utilities
└── app.py                   # Gradio web interface
```

## Research Background

This project addresses three gaps:

- **Technological Gap**: Most detectors are unimodal. Articulatory detection is rooted in physics — vocal-tract movements obey physical laws that AI cannot easily fake.
- **Cognitive Gap**: Explainable AI (XAI) interrupts "System 1" intuitive processing and forces "System 2" analytical evaluation.
- **Ethical Necessity**: Providing victims with detectable, traceable technical proof for digital advocacy and legal support.

## References

- Wang & Huang (2024) — ART-AVDF: Articulatory Representation Learning
- Abercrombie et al. (2024) — Taxonomy of AI Harms
- AV-HuBERT — Audio-Visual Hidden Unit BERT

## License

MIT
