# Factory for creating video analyzers based on configuration

import os
from typing import Protocol, Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")


class VideoAnalyzer(Protocol):
    """Protocol defining the interface for video analyzers."""

    def analyze_video_chunk(self, video_path: str, chunk_info: Dict) -> str:
        """Analyze a single video chunk."""
        ...

    def combine_analyses(self, analyses: List[str], original_video_path: str) -> str:
        """Combine multiple chunk analyses into a single comprehensive analysis."""
        ...

    def analyze_single_video(self, video_path: str) -> str:
        """Analyze a single video file (when no chunking is needed)."""
        ...


def create_analyzer(
    analyzer_type: str = None,
    prompt_type: str = None,
    require_json_keyframes: bool = None,
) -> VideoAnalyzer:
    """
    Create appropriate video analyzer based on configuration.

    Args:
        analyzer_type: 'gemini' or 'openrouter' (defaults to env variable ANALYZER_TYPE)
        prompt_type: Prompt type for analysis
        require_json_keyframes: Whether to require JSON keyframes output

    Returns:
        VideoAnalyzer instance (either GeminiAnalyzer or OpenRouterAnalyzer)

    Raises:
        ValueError: If analyzer_type is not supported
    """
    if analyzer_type is None:
        analyzer_type = os.getenv("ANALYZER_TYPE", "gemini").strip().lower()

    if analyzer_type == "gemini":
        from gemini_analyzer import GeminiAnalyzer
        return GeminiAnalyzer(
            prompt_type=prompt_type,
            require_json_keyframes=require_json_keyframes,
        )
    elif analyzer_type == "openrouter":
        from openrouter_analyzer import OpenRouterAnalyzer
        return OpenRouterAnalyzer(
            prompt_type=prompt_type,
            require_json_keyframes=require_json_keyframes,
        )
    else:
        raise ValueError(
            f"Unknown analyzer type: '{analyzer_type}'. "
            f"Supported types: 'gemini', 'openrouter'"
        )


def get_analyzer_info() -> Dict[str, str]:
    """
    Get information about current analyzer configuration.

    Returns:
        Dictionary with analyzer type and model name
    """
    analyzer_type = os.getenv("ANALYZER_TYPE", "gemini").strip().lower()

    if analyzer_type == "gemini":
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3-pro-preview")
        return {
            "type": "gemini",
            "model": model_name,
            "description": "Google Gemini via Vertex AI",
        }
    elif analyzer_type == "openrouter":
        model_name = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-3-pro-preview")
        return {
            "type": "openrouter",
            "model": model_name,
            "description": "OpenRouter API",
        }
    else:
        return {
            "type": analyzer_type,
            "model": "unknown",
            "description": f"Unknown analyzer type: {analyzer_type}",
        }
