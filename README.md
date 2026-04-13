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

1. **Visual-Lip Encoder** — Frame-by-frame tracking of labial movements using MediaPipe FaceLandmarker (Tasks API). Extracts lip landmarks, shape descriptors (aspect ratio, openness, asymmetry, eccentricity, corner angle, bilabial closure), and generates 256-dimensional feature vectors per frame.

2. **Audio-Articulatory Encoder** — Self-supervised module (Wav2Vec2 `facebook/wav2vec2-base-960h`) that maps audio signals to articulatory representations at 50fps. Uses FFmpeg for robust audio extraction. Optionally integrates Montreal Forced Aligner (MFA) for phoneme-level alignment.

3. **Cross-Modal Fusion** — Transformer-based temporal attention module that projects audio and visual features into a shared space, computes frame-level discrepancy scores combining learned classification and cosine similarity, and generates per-frame heatmaps.

4. **LLM Reasoner** — Multi-provider support (Groq/OpenAI/Anthropic). Aggregates detection signals into structured JSON analysis with verdict, evidence, harm categorization, and recommended actions. Falls back gracefully when LLM is unavailable.

5. **Human-Agent Interface** — Gradio web UI with annotated video overlays (face bounding boxes, lip contours, verdict badges), inline HTML forensic reports with interactive discrepancy timeline charts, and downloadable JSON reports.

## Installation

```bash
# Clone the repository
git clone https://github.com/qimingl4/IS596-deepfake.git
cd IS596-deepfake

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.10+
- FFmpeg (for audio extraction from video)
- [Groq API key](https://console.groq.com/) (for the LLM Reasoner)

```bash
# Create a .env file in the project root
echo "GROQ_API_KEY=your-key-here" > .env
```

The system uses Groq's LLaMA 3.3 70B model by default for fast, free LLM reasoning. You can also configure OpenAI or Anthropic providers in `configs/default.yaml`.

## Usage

### Web Interface

```bash
python -m deepguard.app
```

Then open http://localhost:7860 in your browser.

1. **Upload** a video containing a human face with speech
2. Click **Analyze Video**
3. View the annotated video with per-frame detection overlays
4. Read the inline forensic report (verdict, evidence, timeline chart, flagged frames)
5. Download the JSON report for further analysis

### Programmatic Usage

```python
from deepguard.detection import VisualLipEncoder, AudioArticulatoryEncoder, CrossModalFusion
from deepguard.reasoning import LLMReasoner
from deepguard.interface import ReportGenerator

# Initialize components
visual = VisualLipEncoder()
audio = AudioArticulatoryEncoder()
fusion = CrossModalFusion(visual_dim=256)
reasoner = LLMReasoner(provider="groq", model="llama-3.3-70b-versatile")
reporter = ReportGenerator()

# Run detection
lip_features = visual.process_video("suspect_video.mp4")
audio_features = audio.process("suspect_video.mp4")

import numpy as np
visual_matrix = np.array([
    visual.landmarks_to_feature_vector(f.landmarks, f.descriptors)
    for f in lip_features
])

fusion_result = fusion.analyze(
    audio_features=audio_features.embeddings,
    visual_features=visual_matrix,
    audio_fps=audio_features.frame_rate,
    video_fps=25.0,
)

# Generate report
analysis = reasoner.analyze(fusion_result)
print(reporter.to_text(fusion_result, analysis))
```

## Configuration

Edit `configs/default.yaml` to customize:

| Section | Key Settings |
|---------|-------------|
| `detection.visual` | Face mesh confidence, temporal smoothing |
| `detection.audio` | Wav2Vec2 model, sample rate, MFA toggle |
| `detection.fusion` | Discrepancy threshold, checkpoint path |
| `reasoning` | LLM provider (`groq`/`openai`/`anthropic`), model, temperature |
| `app` | Host, port, max video duration, temp directory |

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
