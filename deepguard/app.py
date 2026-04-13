"""Deep-Guard Agent: Gradio-based interactive web interface.

Provides a user-friendly interface for uploading videos, running deepfake
detection, viewing annotated results, and downloading forensic reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
import numpy as np
import yaml

from deepguard.detection.audio_encoder import WAV2VEC2_FRAME_RATE
from deepguard.detection.visual_encoder import VisualLipEncoder, VISUAL_FEATURE_DIM
from deepguard.detection.audio_encoder import AudioArticulatoryEncoder
from deepguard.detection.fusion import CrossModalFusion
from deepguard.reasoning.llm_reasoner import LLMReasoner
from deepguard.interface.overlay import VideoOverlay
from deepguard.interface.report import ReportGenerator
from deepguard.utils.video import get_video_info

logger = logging.getLogger(__name__)

# ── Custom CSS for a polished dark-themed UI ──────────────────────────────────
CUSTOM_CSS = """
/* Global */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header area */
#header-row {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 32px 36px !important;
    margin-bottom: 20px;
    border: 1px solid #334155;
}
#header-row * {
    color: #f1f5f9 !important;
}
#header-row p {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
}

/* Cards */
.card-panel {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
}

/* Primary button */
#analyze-btn {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 12px 0 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
#analyze-btn:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
}

/* Inline HTML report */
#report-html {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Download buttons row */
#download-row .file-preview {
    border-radius: 8px !important;
}

/* Status / verdict badges */
.verdict-authentic { color: #16a34a; font-weight: 700; }
.verdict-suspicious { color: #d97706; font-weight: 700; }
.verdict-likely-fake { color: #dc2626; font-weight: 700; }
"""


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text())
    return {}


class DeepGuardPipeline:
    """End-to-end deepfake detection pipeline."""

    def __init__(self, config: dict | None = None):
        config = config or load_config()
        det_cfg = config.get("detection", {})
        reason_cfg = config.get("reasoning", {})
        app_cfg = config.get("app", {})

        self.visual_encoder = VisualLipEncoder(
            min_detection_confidence=det_cfg.get("visual", {}).get(
                "face_mesh_confidence", 0.5
            ),
        )
        self.audio_encoder = AudioArticulatoryEncoder(
            model_name=det_cfg.get("audio", {}).get(
                "model_name", "facebook/wav2vec2-base-960h"
            ),
            use_mfa=det_cfg.get("audio", {}).get("use_mfa", False),
        )
        self.fusion = CrossModalFusion(
            visual_dim=VISUAL_FEATURE_DIM,
            discrepancy_threshold=det_cfg.get("fusion", {}).get(
                "discrepancy_threshold", 0.65
            ),
            checkpoint_path=det_cfg.get("fusion", {}).get("checkpoint_path"),
        )
        self.reasoner = LLMReasoner(
            provider=reason_cfg.get("provider", "groq"),
            model=reason_cfg.get("model", "llama-3.3-70b-versatile"),
            base_url=reason_cfg.get("base_url"),
            temperature=reason_cfg.get("temperature", 0.3),
        )
        self.overlay = VideoOverlay()
        self.report_gen = ReportGenerator()
        self.temp_dir = Path(app_cfg.get("temp_dir", "/tmp/deepguard"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, video_path: str) -> tuple[str, str, str]:
        """Run the full detection pipeline on a video.

        Returns:
            (annotated_video_path, report_html_content, report_json_path)
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_info = get_video_info(video_path)
        if video_info.frame_count == 0:
            raise ValueError(f"Video has no frames: {video_path}")

        logger.info(
            "Analyzing: %s (%.1fs, %.0ffps, %d frames)",
            video_path, video_info.duration_sec, video_info.fps, video_info.frame_count,
        )

        # Step 1: Extract visual and audio features
        logger.info("Extracting visual lip features...")
        lip_features = self.visual_encoder.process_video(video_path)

        logger.info("Extracting audio features...")
        audio_features = self.audio_encoder.process(video_path)

        if not lip_features:
            raise ValueError(
                "No face detected in any frame. Please upload a video with "
                "a clearly visible human face."
            )

        visual_matrix = np.array([
            self.visual_encoder.landmarks_to_feature_vector(f.landmarks, f.descriptors)
            for f in lip_features
        ])

        logger.info(
            "Features: visual=%s, audio=%s",
            visual_matrix.shape, audio_features.embeddings.shape,
        )

        # Step 2: Cross-modal fusion
        logger.info("Running cross-modal fusion analysis...")

        fusion_result = self.fusion.analyze(
            audio_features=audio_features.embeddings,
            visual_features=visual_matrix,
            audio_fps=audio_features.frame_rate,
            video_fps=video_info.fps,
        )

        # Step 3: Phoneme context
        phoneme_info = None
        if audio_features.phoneme_segments:
            bilabial = [s for s in audio_features.phoneme_segments if s.is_bilabial]
            phoneme_info = (
                f"Total phonemes: {len(audio_features.phoneme_segments)}\n"
                f"Bilabial phonemes: {len(bilabial)}\n"
            )

        # Step 4: LLM reasoning
        logger.info("Generating AI analysis...")

        analysis = self.reasoner.analyze(fusion_result, phoneme_info)

        # Step 5: Generate outputs
        logger.info("Rendering annotated video and reports...")

        annotated_path = str(self.temp_dir / "annotated_output.mp4")
        self.overlay.render_video(video_path, annotated_path, fusion_result, lip_features)

        json_path = str(self.temp_dir / "report.json")
        self.report_gen.to_json(fusion_result, analysis, json_path, source_file=video_path)

        html_content = self.report_gen.to_html_embed(
            fusion_result, analysis, source_file=video_path,
        )

        logger.info("Analysis complete!")

        return annotated_path, html_content, json_path


# ── Gradio Interface ──────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio web interface."""
    pipeline = None

    def initialize_pipeline():
        nonlocal pipeline
        if pipeline is None:
            pipeline = DeepGuardPipeline()

    def process_video(video_file):
        logger.info("process_video called with: %s (type: %s)", video_file, type(video_file))
        if video_file is None:
            return None, "<p style='color:#94a3b8;'>Please upload a video file first.</p>", None

        try:
            initialize_pipeline()
            logger.info("Starting analysis for: %s", video_file)
            annotated, html_content, json_path = pipeline.analyze(video_file)
            logger.info("Analysis complete. Annotated: %s", annotated)
            return annotated, html_content, json_path

        except ValueError as e:
            logger.warning("Validation error: %s", e)
            return None, f"<p style='color:#ef4444;'>Error: {e}</p>", None
        except Exception as e:
            logger.exception("Pipeline error")
            err_msg = str(e) or type(e).__name__
            return None, f"<p style='color:#ef4444;'>An error occurred: {err_msg}</p>", None

    with gr.Blocks(title="Deep-Guard Agent") as app:

        # ── Header ──
        with gr.Row(elem_id="header-row"):
            gr.Markdown(
                "# 🛡 Deep-Guard Agent\n"
                "**Audio-Visual Deepfake Detection & Cognitive Intervention**\n\n"
                "Upload a video to detect deepfake artifacts. The system analyzes "
                "lip-movement / speech consistency using articulatory representation "
                "learning — detecting mismatches that are physically impossible in "
                "authentic recordings."
            )

        # ── Input / Output Row ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                analyze_btn = gr.Button(
                    "▶  Analyze Video",
                    variant="primary",
                    size="lg",
                    elem_id="analyze-btn",
                )

            with gr.Column(scale=1):
                video_output = gr.Video(label="Annotated Result")

        # ── Inline HTML Report ──
        with gr.Row():
            report_html = gr.HTML(
                value="<p style='color:#94a3b8; text-align:center; padding:2rem;'>"
                      "Upload a video and click Analyze to see the forensic report here.</p>",
                elem_id="report-html",
            )

        # ── JSON Download ──
        with gr.Row(elem_id="download-row"):
            json_output = gr.File(label="Download JSON Report")

        # ── Wiring ──
        analyze_btn.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[video_output, report_html, json_output],
        )

    return app


def _load_dotenv():
    """Load .env file if present."""
    import os
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main():
    """Launch the Deep-Guard Agent web interface."""
    _load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = create_app()
    config = load_config()
    app_cfg = config.get("app", {})
    temp_dir = app_cfg.get("temp_dir", "/tmp/deepguard")
    app.launch(
        server_name=app_cfg.get("host", "0.0.0.0"),
        server_port=app_cfg.get("port", 7860),
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        show_error=True,
        allowed_paths=[temp_dir],
    )


if __name__ == "__main__":
    main()
