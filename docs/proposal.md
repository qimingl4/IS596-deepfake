# Deep-Guard Agent: An Interactive Audio-Visual Intelligent System for Deepfake Detection and Cognitive Intervention

**Course**: IS596 — Intelligent Systems  
**Team**: Qiming Li, Yiting Wang, Yawen Ou  
**Date**: April 2026

---

## 1. Research Questions

This project investigates the following research questions:

**RQ1**: How can articulatory representation learning — grounded in the physics of human speech production — be leveraged to detect audio-visual inconsistencies in deepfake videos more robustly than pixel-level or spectral-level approaches?

**RQ2**: Can an LLM-powered reasoning agent translate opaque detection scores into transparent, evidence-grounded explanations that help non-expert users critically evaluate potentially manipulated media?

**RQ3**: To what extent does providing structured forensic evidence (e.g., phoneme-level mismatch analysis, frame-level discrepancy timelines) shift users from intuitive "System 1" acceptance toward analytical "System 2" evaluation of synthetic media?

---

## 2. Introduction

The rapid advancement of generative AI has made deepfake creation accessible to anyone with a consumer GPU and an internet connection. Tools such as DeepFaceLab, FaceSwap, and more recently diffusion-based face synthesis models can produce hyper-realistic face-swapped and lip-synced videos that are increasingly indistinguishable from authentic footage to the human eye (Tolosana et al., 2020; Mirsky & Lee, 2021). The consequences are severe and far-reaching: political deepfakes have been deployed to manipulate elections and undermine democratic discourse (Chesney & Citron, 2019); non-consensual intimate imagery (NCII) disproportionately harms women and marginalized communities (Ajder et al., 2019); and financial fraud through voice/face impersonation has resulted in losses exceeding $25 million in documented cases (Chen et al., 2024).

Despite growing awareness, current detection solutions suffer from three fundamental gaps:

1. **Technological Gap**: Most deployed detectors are unimodal — they analyze either visual artifacts (e.g., blending boundaries, unnatural blinking) or audio anomalies (e.g., spectral inconsistencies) in isolation. However, the most convincing deepfakes, particularly lip-sync manipulations, require cross-modal analysis where visual lip movements are compared against the corresponding audio signal. Articulatory phonetics tells us that human speech production follows strict physical constraints: bilabial consonants (/b/, /p/, /m/) require complete lip closure, and current synthesis methods routinely violate these constraints in ways that are measurable but invisible to casual observers (Wang & Huang, 2024).

2. **Cognitive Gap**: Even when detection systems correctly identify manipulated content, presenting results as opaque confidence scores (e.g., "87% Fake") fails to support informed decision-making. Research in Dual Process Theory (Kahneman, 2011) demonstrates that humans default to fast, intuitive "System 1" processing when consuming media. Without structured, evidence-based explanations that engage "System 2" analytical thinking, users either over-trust or dismiss detection results without genuine understanding.

3. **Ethical and Legal Gap**: Victims of deepfake abuse — particularly NCII victims — need more than a binary classification. They require forensic reports with traceable evidence chains (timestamped frames, cryptographic hashes, phoneme-level analysis) that can support platform moderation appeals, legal proceedings, and digital advocacy (Abercrombie et al., 2024).

Deep-Guard Agent addresses all three gaps through an integrated system that combines physics-grounded audio-visual detection, LLM-powered explainable reasoning, and a human-centered interface designed to foster critical media literacy.

---

## 3. Related Work / Literature Review

### 3.1 Deepfake Detection Methods

Early deepfake detection relied on visual artifact analysis — detecting blending inconsistencies, unnatural eye blinking, or spectral artifacts in face-swapped images (Li & Lyu, 2019; Afchar et al., 2018). While effective against first-generation face swaps, these methods have proven brittle against newer synthesis techniques that have largely eliminated visible artifacts.

**Audio-visual approaches** represent a more robust detection paradigm. The core insight is that lip-sync deepfakes must coordinate two independently generated signals (visual face manipulation and audio synthesis), and inconsistencies between them are physically grounded rather than artifact-dependent. SyncNet (Chung & Zisserman, 2017) pioneered audio-visual synchronization analysis, while subsequent work extended this to temporal consistency modeling. Haliassos et al. (2021) proposed Lips Don't Lie, demonstrating that irregular mouth movements detectable through lipreading models serve as reliable deepfake indicators.

**AV-HuBERT** (Shi et al., 2022) introduced a self-supervised framework for learning joint audio-visual representations through masked prediction. By training on large-scale unlabeled video, AV-HuBERT captures fine-grained correlations between speech acoustics and facial movements that are extremely difficult for synthesis methods to replicate. This approach demonstrated that self-supervised representations encode rich phonetic information about the relationship between mouth shapes and speech sounds.

Most directly relevant to our work, **ART-AVDF** (Wang & Huang, 2024) introduced articulatory representation learning for audio-visual deepfake detection. Rather than operating on raw pixel features, ART-AVDF maps both audio and visual signals to articulatory feature spaces that describe the physical configuration of the vocal tract during speech. This physics-grounded representation is fundamentally more robust than learned statistical correlations because it exploits physical laws that generative models cannot easily circumvent — the vocal tract is a physical system whose dynamics are governed by biomechanical constraints.

### 3.2 Self-Supervised Speech Representations

**Wav2Vec 2.0** (Baevski et al., 2020) demonstrated that self-supervised pre-training on unlabeled speech data can learn rich acoustic representations. The model's hidden states encode information about phonetic content, speaker characteristics, and articulatory configurations — even without explicit supervision. When fine-tuned or probed on downstream tasks, Wav2Vec2 representations have been shown to capture articulatory features that correlate with vocal tract configurations (Cho et al., 2023). This makes Wav2Vec2 an effective audio encoder for our system: its learned representations implicitly encode the physical speech production process, enabling comparison against visual lip movement features.

### 3.3 Explainable AI for Media Forensics

A critical limitation of existing deepfake detectors is their opacity. CNNs and transformer-based classifiers produce confidence scores without explaining *what* makes a video suspicious or *why* a particular segment was flagged (Arrieta et al., 2020). This "black box" problem is particularly problematic in the deepfake domain, where false positives can unfairly damage reputations and false negatives can leave harmful content unchallenged.

Recent work has explored several directions for explainable deepfake detection: attention visualization to highlight manipulated regions (Wodajo & Atnafu, 2021), temporal saliency maps showing which frames contribute most to the detection decision (Haliassos et al., 2022), and natural language explanations generated by multimodal LLMs (Jia et al., 2024). Our work extends this line by using an LLM not merely as a post-hoc explainer but as an active reasoning agent that synthesizes multiple detection signals into structured forensic analyses grounded in physical evidence.

### 3.4 Cognitive Intervention and Dual Process Theory

Kahneman's (2011) Dual Process Theory distinguishes between fast, automatic "System 1" processing and slow, deliberative "System 2" reasoning. Media consumption — especially on social platforms — overwhelmingly engages System 1: users scroll, react, and share without critical evaluation (Pennycook & Rand, 2019). Deepfakes exploit this cognitive vulnerability by presenting content that "looks right" at the surface level.

Research on cognitive intervention for misinformation demonstrates that *nudges* — structured prompts that encourage users to consider accuracy before sharing — can significantly reduce misinformation spread (Pennycook et al., 2021). "Prebunking" approaches that inoculate users with examples of manipulation techniques have shown promise in building resilience (Roozenbeek & van der Linden, 2022). Our system operationalizes these insights by providing not just detection results but structured forensic evidence — phoneme-level mismatch analysis, frame-by-frame discrepancy timelines, and contextualized harm assessments — that transform passive consumption into active evaluation.

### 3.5 Harm Taxonomy and Ethical Frameworks

Abercrombie et al. (2024) developed a comprehensive taxonomy of harms arising from AI-generated media, distinguishing between content-level harms (what is depicted), distribution-level harms (how it spreads), and systemic harms (long-term effects on trust and information ecosystems). Our system incorporates this taxonomy through automated harm categorization (political manipulation, NCII, financial fraud, general synthetic media) and generates tailored recommended actions for each category.

The ethical design of detection tools must also address the dual-use concern: detection systems can be weaponized to suppress legitimate creative expression or used as censorship tools (Paris & Donovan, 2019). We mitigate this by designing Deep-Guard Agent as a transparency tool that provides evidence for human judgment rather than an automated content moderation system that makes final decisions.

---

## 4. Study Rationale

Existing deepfake detection systems face a fundamental paradox: as generative models improve, artifact-based detection becomes increasingly unreliable, while human perception becomes increasingly deceived. This creates an urgent need for detection approaches that:

1. **Are grounded in physical invariants** rather than statistical artifacts that can be eliminated through better synthesis. The physics of human speech production — the biomechanics of lip closure for bilabial sounds, the aerodynamics of fricative production, the temporal coordination of articulators — represents a detection signal space that is not easily circumvented because it reflects constraints of the physical world, not limitations of current technology.

2. **Explain rather than classify**. A 92% confidence score is meaningless to a victim seeking platform recourse, a journalist verifying a source, or a voter evaluating a political video. These users need to understand *what specific evidence* supports the detection result and *how reliable* that evidence is.

3. **Integrate detection with intervention**. Detection alone is insufficient if users cannot act on the results. Deep-Guard Agent bridges the gap between technical detection and practical action by generating platform-ready forensic reports, providing harm-contextualized recommendations, and fostering the critical thinking skills that build long-term media resilience.

This project is further motivated by the observation that the most impactful deepfakes — political misinformation, NCII, financial fraud — are precisely those where the combination of technical detection, explainable reasoning, and actionable forensics is most urgently needed.

---

## 5. Conceptual Model and Approach of the Intelligent System

### 5.1 Design Philosophy

Deep-Guard Agent is designed as a **collaborative human-AI system** rather than a fully automated detector. The system's architecture reflects three core design principles:

- **Physics over Pixels**: Detection is rooted in articulatory phonetics — physical properties of speech production that deepfake generators cannot easily violate.
- **Transparency over Accuracy**: Rather than optimizing solely for classification accuracy, the system prioritizes generating interpretable, evidence-grounded explanations that enable informed human judgment.
- **Intervention over Classification**: The system aims not just to detect deepfakes but to actively support cognitive intervention — helping users develop critical evaluation skills through structured forensic analysis.

### 5.2 Conceptual Architecture

The system follows a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Physics-Grounded Detection                        │
│                                                             │
│  Visual-Lip Encoder ─────┐                                  │
│  (MediaPipe FaceLandmarker)│──► Cross-Modal Fusion           │
│                           │    (Transformer Temporal         │
│  Audio-Articulatory ──────┘     Attention)                   │
│  Encoder (Wav2Vec2)             ──► Discrepancy Scores       │
│                                      + Heatmaps             │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: LLM-Powered Reasoning                             │
│                                                             │
│  Detection Signals ──► Structured Context ──► LLM Analysis   │
│  (scores, flagged     (formatted for LLM    (JSON-structured │
│   frames, phonemes)    consumption)           verdict,        │
│                                               evidence,       │
│                                               recommendations)│
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Human-Agent Interface (Cognitive Intervention)     │
│                                                             │
│  Annotated Video ─── Inline Forensic Report ─── JSON Export  │
│  (per-frame overlays) (verdict, evidence,       (traceable    │
│                        timeline chart,           forensic      │
│                        flagged frames)           data)         │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Cognitive Intervention Model

The interface is designed to interrupt System 1 processing and activate System 2 evaluation through a structured information flow:

1. **Attention Capture**: The annotated video with per-frame colored overlays (green for authentic, red for flagged frames) immediately signals that the content requires scrutiny.
2. **Evidence Presentation**: The forensic report provides specific, concrete evidence — "Frame 45 (1.8s): bilabial phoneme /p/ detected in audio but lip closure absent in visual" — rather than abstract scores.
3. **Contextualized Harm Assessment**: Automated harm categorization (political, NCII, financial fraud) with tailored recommended actions connects detection to actionable next steps.
4. **Forensic Traceability**: SHA-256 file hashing, timestamped frame references, and exportable JSON reports create an evidence chain suitable for platform appeals or legal proceedings.

---

## 6. Technical Explanation of the System

### 6.1 Visual-Lip Encoder

The visual encoder uses **MediaPipe FaceLandmarker** (Tasks API) to extract 478-point 3D face landmarks from each video frame. From these, we isolate lip-region landmarks (22 outer + 22 inner lip contour points) and compute a set of physics-informed shape descriptors:

| Descriptor | Definition | Detection Relevance |
|-----------|-----------|-------------------|
| **Aspect Ratio** | Height/width of lip bounding box | Captures mouth opening shape |
| **Openness** | Euclidean distance between upper and lower lip midpoints | Critical for bilabial detection |
| **Asymmetry** | Absolute difference in left vs. right lip height | Exposes synthesis artifacts |
| **Eccentricity** | PCA-based shape elongation measure | Captures unnatural lip shapes |
| **Corner Angle** | Angle between lip corners relative to center | Detects impossible corner positions |
| **Bilabial Closure** | Lip openness normalized by face width | Key phoneme-specific signal |

These descriptors, combined with flattened normalized landmark coordinates, produce a **256-dimensional feature vector** per frame. Temporal smoothing (exponential moving average) is optionally applied to reduce frame-to-frame jitter from landmark detection noise.

### 6.2 Audio-Articulatory Encoder

The audio encoder extracts articulatory representations using **Wav2Vec2** (`facebook/wav2vec2-base-960h`), a self-supervised speech model pre-trained on 960 hours of unlabeled English speech.

Audio is extracted from video using **FFmpeg** (for robust codec support), resampled to 16kHz mono, and processed through the Wav2Vec2 model to produce **768-dimensional frame-level embeddings** at 50fps (one frame per 20ms of audio). These embeddings have been shown to encode articulatory information — the physical configuration of the vocal tract during speech — without explicit articulatory supervision (Cho et al., 2023).

Optionally, the **Montreal Forced Aligner (MFA)** can be used to produce phoneme-level temporal alignment, enabling explicit detection of bilabial phoneme mismatches (i.e., audio segments where /b/, /p/, or /m/ sounds are detected but visual lip closure is absent).

### 6.3 Cross-Modal Fusion

The fusion module aligns audio and visual feature sequences to a common temporal resolution through linear interpolation, then processes them through a **Transformer-based architecture**:

1. **Modality Projection**: Separate projection networks (Linear → GELU → LayerNorm → Linear → LayerNorm) map audio (768d) and visual (256d) features into a shared 256-dimensional embedding space.

2. **Temporal Attention**: A 2-layer, 4-head Transformer encoder captures cross-frame temporal dependencies — essential for detecting phoneme transition violations where a lip shape must change at a specific rate to match the corresponding audio.

3. **Discrepancy Scoring**: Two complementary signals are combined:
   - **Learned discrepancy head** (Linear → GELU → Linear → Sigmoid): A trained binary classifier that predicts frame-level mismatch probability from concatenated audio-visual projections.
   - **Cosine discrepancy**: `(1 - cosine_similarity) / 2` between projected audio and visual features, providing a geometry-based mismatch signal.
   - **Final score**: `0.7 × learned_score + 0.3 × cosine_discrepancy`

Frames exceeding the configurable threshold (default: 0.65) are flagged, and the overall deepfake probability is computed as the mean of all frame-level scores.

### 6.4 LLM Reasoning Agent

Detection scores are aggregated into a structured context document containing: overall score, top-10 highest-scoring frames, flagged frame count, cosine similarity statistics, and optional phoneme alignment data. This context is submitted to a **large language model** (default: LLaMA 3.3 70B via Groq API) with a carefully designed system prompt that instructs the model to:

1. **Synthesize** all detection signals into a coherent analysis
2. **Explain** findings in plain language with specific physical evidence
3. **Ground** conclusions in acoustic and articulatory science

The LLM returns a structured JSON response containing: summary, verdict (authentic/suspicious/likely\_fake), evidence points, harm category, recommended actions, phoneme analysis, and confidence reasoning. A robust fallback mechanism ensures the system remains functional even when the LLM is unavailable.

### 6.5 Human-Agent Interface

The web interface (built with **Gradio 6**) presents results through three coordinated views:

- **Annotated Video**: Per-frame overlays showing face bounding boxes, lip landmark contours, verdict badges, and red-bordered flagged frames.
- **Inline Forensic Report**: A dark-themed HTML report rendered directly in the browser, featuring the verdict badge, AI-generated summary, evidence list, interactive discrepancy timeline chart (Canvas-based), flagged frame table with timestamps, and detection metadata.
- **Exportable JSON Report**: Machine-readable forensic data with SHA-256 file hashing for evidence chain integrity.

---

## 7. Data Use / Management Plan

### 7.1 Input Data

Deep-Guard Agent processes **user-uploaded video files** (MP4, AVI, MOV, MKV) containing human faces with speech. The system does not collect, store, or transmit user data beyond the immediate analysis session. All processing occurs locally, and temporary files (extracted audio, annotated video, reports) are stored in a configurable temporary directory (`/tmp/deepguard/`) and are not persisted across sessions.

### 7.2 Pre-Trained Models

The system relies on the following pre-trained models, all of which are publicly available and used under their respective open-source licenses:

| Model | Source | License | Purpose |
|-------|--------|---------|---------|
| MediaPipe FaceLandmarker | Google | Apache 2.0 | 3D face landmark extraction |
| Wav2Vec2-base-960h | Meta/Facebook | MIT | Audio feature extraction |
| LLaMA 3.3 70B (via Groq) | Meta | Llama 3.3 Community License | LLM reasoning |

### 7.3 Evaluation Data (Planned)

For system evaluation, we plan to use the following publicly available deepfake detection benchmarks:

- **FakeAVCeleb** (Khalid et al., 2021): A multi-modal deepfake dataset with both audio and visual manipulations, containing real and fake celebrity videos.
- **DFDC** (Dolhansky et al., 2020): The DeepFake Detection Challenge dataset from Facebook, containing over 100,000 video clips with known ground truth labels.
- **ASVspoof 2019** (Todisco et al., 2019): For evaluating audio-only detection capabilities.

### 7.4 Privacy and Ethical Considerations

- **No data retention**: User-uploaded videos are processed in-memory and via temporary files that are overwritten on each analysis. No persistent database stores user content.
- **No biometric data collection**: While the system processes facial landmarks, these are used only for real-time detection and are not stored or linked to identities.
- **Harm-aware design**: The harm categorization system (political, NCII, financial fraud, general) is designed to connect victims with appropriate resources rather than to enable surveillance or censorship.
- **Transparency by design**: All detection signals and reasoning are exposed to the user. The system never makes opaque automated decisions — it provides evidence for human judgment.
- **Dual-use mitigation**: The system is designed as an analysis tool, not a content moderation system. It does not automatically flag, remove, or report content. The decision to act on detection results remains with the human user.

### 7.5 Data Pipeline

```
User Video ──► FFmpeg (audio extraction) ──► Wav2Vec2 (embeddings)
     │                                              │
     └──► MediaPipe (landmarks) ──► Feature Vector  │
                                        │           │
                                        └─── Fusion ┘
                                              │
                                     Discrepancy Scores
                                              │
                                    LLM Reasoning (Groq API)
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              Annotated Video    Forensic Reports
                              (local temp)     (JSON + inline HTML)
```

All data flows are local except the LLM API call (Groq), which receives only aggregated numerical scores and metadata — never raw video, audio, or facial data — ensuring that sensitive biometric information never leaves the user's machine.

---

## References

Abercrombie, G., Curry, A. C., Dinkar, T., & Rieser, V. (2024). A taxonomy of harms from AI-generated media. *Proceedings of the ACM Conference on Fairness, Accountability, and Transparency*.

Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a compact facial video forgery detection network. *IEEE International Workshop on Information Forensics and Security (WIFS)*.

Ajder, H., Patrini, G., Cavalli, F., & Cullen, L. (2019). The state of deepfakes: Landscape, threats, and impact. *Deeptrace Labs Report*.

Arrieta, A. B., et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82–115.

Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.

Chen, Y., et al. (2024). Deepfake-enabled financial fraud: Emerging threats and countermeasures. *IEEE Security & Privacy*.

Chesney, R., & Citron, D. K. (2019). Deep fakes: A looming challenge for privacy, democracy, and national security. *California Law Review*, 107, 1753–1820.

Cho, S., et al. (2023). Evidence of articulatory information in self-supervised speech representations. *Proceedings of Interspeech*.

Chung, J. S., & Zisserman, A. (2017). Out of time: Automated lip sync in the wild. *Asian Conference on Computer Vision (ACCV)*.

Dolhansky, B., et al. (2020). The DeepFake Detection Challenge (DFDC) dataset. *arXiv preprint arXiv:2006.07397*.

Haliassos, A., Vougioukas, K., Petridis, S., & Pantic, M. (2021). Lips don't lie: A generalisable and robust approach to face forgery detection. *CVPR*.

Haliassos, A., Mira, R., Petridis, S., & Pantic, M. (2022). Leveraging real talking faces via self-supervision for robust forgery detection. *CVPR*.

Jia, S., et al. (2024). Can ChatGPT detect deepfakes? A study of using multimodal large language models for media forensics. *IEEE/CVF Winter Conference on Applications of Computer Vision*.

Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

Khalid, H., Tariq, S., Kim, M., & Woo, S. S. (2021). FakeAVCeleb: A novel audio-video multimodal deepfake dataset. *NeurIPS Datasets and Benchmarks Track*.

Li, Y., & Lyu, S. (2019). Exposing deepfake videos by detecting face warping artifacts. *CVPR Workshops*.

Mirsky, Y., & Lee, W. (2021). The creation and detection of deepfakes: A survey. *ACM Computing Surveys*, 54(1), 1–41.

Paris, B., & Donovan, J. (2019). Deepfakes and cheap fakes. *Data & Society Report*.

Pennycook, G., & Rand, D. G. (2019). Lazy, not biased: Susceptibility to partisan fake news is better explained by lack of reasoning than by motivated reasoning. *Cognition*, 188, 39–50.

Pennycook, G., Epstein, Z., Mosleh, M., Arechar, A. A., Eckles, D., & Rand, D. G. (2021). Shifting attention to accuracy can reduce misinformation online. *Nature*, 592, 590–595.

Roozenbeek, J., & van der Linden, S. (2022). How to combat health misinformation: A psychological approach. *American Journal of Tropical Medicine and Hygiene*, 107(2), 14–17.

Shi, B., Hsu, W.-N., Lakhotia, K., & Mohamed, A. (2022). Learning audio-visual speech representation by masked multimodal cluster prediction. *ICLR*.

Todisco, M., et al. (2019). ASVspoof 2019: Future horizons in spoofed and fake audio detection. *Proceedings of Interspeech*.

Tolosana, R., Vera-Rodriguez, R., Fierrez, J., Morales, A., & Ortega-Garcia, J. (2020). Deepfakes and beyond: A survey of face manipulation and fake detection. *Information Fusion*, 64, 131–148.

Wang, Y., & Huang, J. (2024). ART-AVDF: Articulatory representation learning for audio-visual deepfake detection. *IEEE Transactions on Information Forensics and Security*.

Wodajo, D., & Atnafu, S. (2021). Deepfake video detection using convolutional vision transformer. *arXiv preprint arXiv:2102.11126*.
