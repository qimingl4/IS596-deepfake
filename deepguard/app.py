"""Deep-Guard Agent: Gradio-based interactive web interface.

Provides a user-friendly interface for uploading videos, running deepfake
detection, viewing annotated results, and downloading forensic reports.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr
import yaml

from deepguard.detection.visual_encoder import VisualLipEncoder
from deepguard.detection.audio_encoder import AudioArticulatoryEncoder
from deepguard.detection.fusion import CrossModalFusion
from deepguard.reasoning.llm_reasoner import LLMReasoner
from deepguard.interface.overlay import VideoOverlay
from deepguard.interface.report import ReportGenerator
from deepguard.utils.video import get_video_info

import numpy as np


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

        self.visual_encoder = VisualLipEncoder(
            min_detection_confidence=det_cfg.get("visual", {}).get("face_mesh_confidence", 0.5),
        )
        self.audio_encoder = AudioArticulatoryEncoder(
            model_name=det_cfg.get("audio", {}).get("model_name", "facebook/wav2vec2-base-960h"),
        )
        self.fusion = CrossModalFusion(
            discrepancy_threshold=det_cfg.get("fusion", {}).get("discrepancy_threshold", 0.65),
        )
        self.reasoner = LLMReasoner(
            model=reason_cfg.get("model", "gpt-4o"),
            temperature=reason_cfg.get("temperature", 0.3),
        )
        self.overlay = VideoOverlay()
        self.report_gen = ReportGenerator()

    def analyze(self, video_path: str) -> tuple[str, str, str]:
        """Run the full detection pipeline on a video.

        Returns:
            (annotated_video_path, report_text, report_json_path)
        """
        video_info = get_video_info(video_path)

        # Step 1: Extract visual lip features
        lip_features = self.visual_encoder.process_video(video_path)
        visual_matrix = np.array([f.landmarks.flatten()[:128] for f in lip_features])

        # Step 2: Extract audio-articulatory features
        audio_features = self.audio_encoder.process(video_path)

        # Step 3: Cross-modal fusion
        fusion_result = self.fusion.analyze(
            audio_features=audio_features.embeddings,
            visual_features=visual_matrix,
            video_fps=video_info.fps,
        )

        # Step 4: LLM reasoning
        analysis = self.reasoner.analyze(fusion_result)

        # Step 5: Generate outputs
        tmp_dir = Path("/tmp/deepguard")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        annotated_path = str(tmp_dir / "annotated_output.mp4")
        self.overlay.render_video(video_path, annotated_path, fusion_result)

        json_path = str(tmp_dir / "report.json")
        self.report_gen.to_json(fusion_result, analysis, json_path, source_file=video_path)

        report_text = self.report_gen.to_text(fusion_result, analysis, source_file=video_path)

        return annotated_path, report_text, json_path


def create_app() -> gr.Blocks:
    """Create the Gradio web interface."""
    pipeline = None

    def initialize_pipeline():
        nonlocal pipeline
        if pipeline is None:
            pipeline = DeepGuardPipeline()

    def process_video(video_file):
        if video_file is None:
            return None, "Please upload a video file.", None
        initialize_pipeline()
        annotated, report_text, json_path = pipeline.analyze(video_file)
        return annotated, report_text, json_path

    with gr.Blocks(
        title="Deep-Guard Agent",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Deep-Guard Agent\n"
            "**An Interactive Audio-Visual Agent for Deepfake Detection and Cognitive Intervention**\n\n"
            "Upload a video to analyze it for potential deepfake artifacts. The system examines "
            "audio-visual consistency using articulatory representation learning."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Annotated Result")

        with gr.Row():
            report_output = gr.Textbox(
                label="Analysis Report",
                lines=15,
                show_copy_button=True,
            )

        with gr.Row():
            json_output = gr.File(label="Download JSON Report")

        analyze_btn.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[video_output, report_output, json_output],
        )

    return app


def main():
    """Launch the Deep-Guard Agent web interface."""
    app = create_app()
    config = load_config()
    app_cfg = config.get("app", {})
    app.launch(
        server_name=app_cfg.get("host", "0.0.0.0"),
        server_port=app_cfg.get("port", 7860),
    )


if __name__ == "__main__":
    main()
