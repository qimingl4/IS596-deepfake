"""LLM Reasoner: synthesizes detection signals into human-readable explanations.

Aggregates heatmaps, mismatch scores, and metadata into a unified analysis context.
Translates technical data into plain-language reports grounded in physical and
acoustic evidence from the detection layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from deepguard.detection.fusion import FusionResult


SYSTEM_PROMPT = """\
You are Deep-Guard Agent, an expert deepfake forensic analyst. Your role is to interpret
audio-visual detection results and explain them to non-technical users.

You will receive structured detection data including:
- Overall deepfake probability score
- Per-frame discrepancy scores between lip movements and audio
- Flagged frames where audio-visual mismatch exceeds the threshold
- Metadata about the analysis

Your responsibilities:
1. SYNTHESIZE: Aggregate all detection signals into a coherent analysis.
2. EXPLAIN: Translate technical findings into plain language. Instead of just saying
   "90% Fake", explain WHICH phonemes mismatch and WHY.
3. GROUND: Base all conclusions on physical and acoustic evidence. Never speculate
   beyond what the data supports.

Classify the content into one of these harm categories when applicable:
- Political deepfake (misinformation/manipulation)
- Non-consensual intimate imagery (NCII)
- Financial fraud (impersonation for financial gain)
- General synthetic media (entertainment/satire)

Provide context-specific guidance based on the harm category.
"""


@dataclass
class AnalysisReport:
    """Structured analysis report from the LLM reasoner."""

    summary: str                    # Plain-language summary
    verdict: str                    # "authentic" | "suspicious" | "likely_fake"
    confidence: float               # 0.0 to 1.0
    evidence: list[str]             # Key pieces of evidence
    harm_category: str | None       # Detected harm category
    recommended_actions: list[str]  # Suggested next steps
    raw_response: str               # Full LLM response


class LLMReasoner:
    """Uses an LLM to reason about detection results and generate explanations."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)

    def _build_detection_context(self, fusion_result: FusionResult) -> str:
        """Build a structured text representation of detection results for the LLM."""
        meta = fusion_result.metadata
        lines = [
            "## Detection Results",
            f"- Overall deepfake score: {fusion_result.overall_score:.3f}",
            f"- Frames analyzed: {meta.get('num_frames_analyzed', 'N/A')}",
            f"- Frames flagged (above threshold {meta.get('threshold', 0.65)}): "
            f"{meta.get('num_flagged_frames', 0)}",
            f"- Flagged ratio: {meta.get('flagged_ratio', 0):.1%}",
            "",
            "## Frame-Level Discrepancy Scores (top 10 highest)",
        ]

        # Include the top mismatch frames
        scores = fusion_result.discrepancy_scores
        if len(scores) > 0:
            top_indices = scores.argsort()[-10:][::-1]
            for idx in top_indices:
                lines.append(f"  Frame {idx}: {scores[idx]:.3f}")

        return "\n".join(lines)

    def analyze(self, fusion_result: FusionResult) -> AnalysisReport:
        """Generate a human-readable analysis report from detection results.

        Args:
            fusion_result: Output from the CrossModalFusion module.

        Returns:
            AnalysisReport with summary, verdict, and recommended actions.
        """
        context = self._build_detection_context(fusion_result)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Analyze the following deepfake detection results and provide:\n"
                    f"1. A plain-language summary\n"
                    f"2. Your verdict (authentic / suspicious / likely_fake)\n"
                    f"3. Key evidence supporting your verdict\n"
                    f"4. Harm category if applicable\n"
                    f"5. Recommended actions for the user\n\n"
                    f"{context}"
                )},
            ],
        )

        raw = response.choices[0].message.content or ""

        # Derive verdict from overall score (LLM response is supplementary)
        score = fusion_result.overall_score
        if score < 0.3:
            verdict = "authentic"
        elif score < 0.6:
            verdict = "suspicious"
        else:
            verdict = "likely_fake"

        return AnalysisReport(
            summary=raw,
            verdict=verdict,
            confidence=score,
            evidence=[
                f"Overall mismatch score: {score:.3f}",
                f"Flagged {fusion_result.metadata.get('num_flagged_frames', 0)} "
                f"of {fusion_result.metadata.get('num_frames_analyzed', 0)} frames",
            ],
            harm_category=None,  # Parsed from LLM output in production
            recommended_actions=[],
            raw_response=raw,
        )
