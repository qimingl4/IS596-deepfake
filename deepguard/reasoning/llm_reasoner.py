"""LLM Reasoner: synthesizes detection signals into human-readable explanations.

Aggregates heatmaps, mismatch scores, and metadata into a unified analysis context.
Translates technical data into plain-language reports grounded in physical and
acoustic evidence from the detection layer.

Supports OpenAI and Anthropic providers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from deepguard.detection.fusion import FusionResult

logger = logging.getLogger(__name__)


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
- political: Political deepfake (misinformation/manipulation)
- ncii: Non-consensual intimate imagery (NCII)
- financial_fraud: Financial fraud (impersonation for financial gain)
- general: General synthetic media (entertainment/satire)

You MUST respond in valid JSON with this exact structure:
{
  "summary": "Plain-language explanation of findings",
  "verdict": "authentic | suspicious | likely_fake",
  "evidence": ["evidence point 1", "evidence point 2", ...],
  "harm_category": "political | ncii | financial_fraud | general | null",
  "recommended_actions": ["action 1", "action 2", ...],
  "phoneme_analysis": "Description of which phonemes show mismatch patterns",
  "confidence_reasoning": "Why the confidence level is what it is"
}
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
    phoneme_analysis: str           # Phoneme-level mismatch description
    confidence_reasoning: str       # Why this confidence level
    raw_response: str               # Full LLM response


_PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
    },
}


def _create_client(provider: str, api_key: str | None = None, base_url: str | None = None):
    """Create the appropriate LLM client based on provider."""
    import os

    defaults = _PROVIDER_DEFAULTS.get(provider, {})

    # Resolve API key: explicit arg → env var
    if not api_key:
        env_key = defaults.get("env_key", "")
        api_key = os.environ.get(env_key)

    # Resolve base URL: explicit arg → provider default
    if not base_url:
        base_url = defaults.get("base_url")

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. "
                "Install with: pip install anthropic"
            )
    else:
        # OpenAI-compatible providers (OpenAI, Groq, Together, etc.)
        from openai import OpenAI
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)


class LLMReasoner:
    """Uses an LLM to reason about detection results and generate explanations."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = _create_client(provider, api_key, base_url)

    def _build_detection_context(
        self, fusion_result: FusionResult, phoneme_info: str | None = None
    ) -> str:
        """Build a structured text representation of detection results for the LLM."""
        meta = fusion_result.metadata
        lines = [
            "## Detection Results",
            f"- Overall deepfake score: {fusion_result.overall_score:.3f}",
            f"- Frames analyzed: {meta.get('num_frames_analyzed', 'N/A')}",
            f"- Frames flagged (above threshold {meta.get('threshold', 0.65)}): "
            f"{meta.get('num_flagged_frames', 0)}",
            f"- Flagged ratio: {meta.get('flagged_ratio', 0):.1%}",
            f"- Mean cosine similarity: {meta.get('mean_cosine_similarity', 'N/A')}",
            "",
            "## Frame-Level Discrepancy Scores (top 10 highest)",
        ]

        scores = fusion_result.discrepancy_scores
        if len(scores) > 0:
            top_k = min(10, len(scores))
            top_indices = scores.argsort()[-top_k:][::-1]
            for idx in top_indices:
                lines.append(f"  Frame {idx}: {scores[idx]:.3f}")

        if phoneme_info:
            lines.extend(["", "## Phoneme Analysis", phoneme_info])

        return "\n".join(lines)

    def _call_llm(self, system: str, user: str) -> str:
        """Call the LLM and return the response text."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""

    def _parse_response(self, raw: str, fusion_result: FusionResult) -> AnalysisReport:
        """Parse the LLM JSON response into an AnalysisReport."""
        score = fusion_result.overall_score

        # Score-based verdict as fallback
        if score < 0.3:
            default_verdict = "authentic"
        elif score < 0.6:
            default_verdict = "suspicious"
        else:
            default_verdict = "likely_fake"

        try:
            # Try to extract JSON from the response
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON block in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        verdict = data.get("verdict", default_verdict)
        if verdict not in ("authentic", "suspicious", "likely_fake"):
            verdict = default_verdict

        harm_category = data.get("harm_category")
        if harm_category not in ("political", "ncii", "financial_fraud", "general", None):
            harm_category = None

        evidence = data.get("evidence", [])
        if not evidence:
            evidence = [
                f"Overall mismatch score: {score:.3f}",
                f"Flagged {fusion_result.metadata.get('num_flagged_frames', 0)} "
                f"of {fusion_result.metadata.get('num_frames_analyzed', 0)} frames",
            ]

        return AnalysisReport(
            summary=data.get("summary", raw),
            verdict=verdict,
            confidence=score,
            evidence=evidence,
            harm_category=harm_category,
            recommended_actions=data.get("recommended_actions", []),
            phoneme_analysis=data.get("phoneme_analysis", ""),
            confidence_reasoning=data.get("confidence_reasoning", ""),
            raw_response=raw,
        )

    def analyze(
        self,
        fusion_result: FusionResult,
        phoneme_info: str | None = None,
    ) -> AnalysisReport:
        """Generate a human-readable analysis report from detection results.

        Args:
            fusion_result: Output from the CrossModalFusion module.
            phoneme_info: Optional phoneme alignment info string.

        Returns:
            AnalysisReport with summary, verdict, and recommended actions.
        """
        context = self._build_detection_context(fusion_result, phoneme_info)

        user_prompt = (
            "Analyze the following deepfake detection results. "
            "Respond ONLY with valid JSON matching the required structure.\n\n"
            f"{context}"
        )

        try:
            raw = self._call_llm(SYSTEM_PROMPT, user_prompt)
            return self._parse_response(raw, fusion_result)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            # Return a fallback report based on detection scores alone
            score = fusion_result.overall_score
            if score < 0.3:
                verdict = "authentic"
            elif score < 0.6:
                verdict = "suspicious"
            else:
                verdict = "likely_fake"

            return AnalysisReport(
                summary=(
                    f"Detection analysis completed with overall score {score:.3f}. "
                    f"LLM reasoning unavailable ({e})."
                ),
                verdict=verdict,
                confidence=score,
                evidence=[
                    f"Overall mismatch score: {score:.3f}",
                    f"Flagged {fusion_result.metadata.get('num_flagged_frames', 0)} "
                    f"of {fusion_result.metadata.get('num_frames_analyzed', 0)} frames",
                ],
                harm_category=None,
                recommended_actions=["Review flagged frames manually"],
                phoneme_analysis="",
                confidence_reasoning="Based on audio-visual discrepancy scores only.",
                raw_response=f"Error: {e}",
            )
