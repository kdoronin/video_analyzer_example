# OpenRouter API analyzer for video content analysis

import os
import time
import random
import base64
import mimetypes
import httpx
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv("config.env")


class OpenRouterAnalyzer:
    """Handles video analysis using OpenRouter API."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        prompt_type: str = None,
        require_json_keyframes: bool = None,
    ):
        """
        Initialize OpenRouter analyzer.

        Args:
            api_key: OpenRouter API key (defaults to env variable)
            model_name: Model name (defaults to env variable)
            prompt_type: One of [general, lecture, meeting, presentation, tutorial, marketing, language_lesson, interview]
            require_json_keyframes: When True, ask model to ensure KeyFrames are emitted as JSON
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-3-pro-preview")

        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided via OPENROUTER_API_KEY env variable or parameter")

        # Initialize OpenAI client with OpenRouter base URL
        # Extended timeout for long video processing (up to 60 min videos)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=httpx.Timeout(
                timeout=1800.0,    # 30 min total timeout
                connect=60.0,      # 60 sec connect timeout
                read=1800.0,       # 30 min read timeout (waiting for response)
                write=300.0,       # 5 min write timeout (uploading video)
            ),
        )

        # Prompt selection
        self.prompt_type = (prompt_type or os.getenv("PROMPT_TYPE", "general")).strip().lower()
        self.require_json_keyframes = (
            require_json_keyframes
            if require_json_keyframes is not None
            else os.getenv("REQUIRE_JSON_KEYFRAMES", "false").strip().lower() in ("1", "true", "yes")
        )

        # Load prompts from XML files
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        # Prompt templates map
        self.prompt_map = {
            "general": "chunk_analysis_prompt.xml",
            "lecture": "chunk_analysis_lecture.xml",
            "meeting": "chunk_analysis_meeting.xml",
            "presentation": "chunk_analysis_presentation.xml",
            "tutorial": "chunk_analysis_tutorial.xml",
            "marketing": "chunk_analysis_marketing.xml",
            "language_lesson": "chunk_analysis_language_lesson.xml",
            "interview": "chunk_analysis_interview.xml",
        }
        selected_prompt_file = self.prompt_map.get(self.prompt_type, self.prompt_map["general"])
        self.chunk_prompt_template = self._load_xml_prompt(selected_prompt_file)
        self.combine_prompt_template = self._load_xml_prompt("combine_analysis_prompt.xml")

    def _load_xml_prompt(self, filename: str) -> str:
        """
        Load XML prompt template from file.

        Args:
            filename: Name of the XML prompt file

        Returns:
            Raw XML content as string
        """
        prompt_path = os.path.join(self.prompts_dir, filename)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _format_prompt(self, template: str, **params) -> str:
        """
        Format XML prompt template with parameters.

        Args:
            template: XML template string
            **params: Parameters to substitute in template

        Returns:
            Formatted prompt string
        """
        formatted = template
        for key, value in params.items():
            formatted = formatted.replace(f"{{{key}}}", str(value))
        return formatted

    def _retry_with_backoff(self, func, *args, max_retries: int = 5, **kwargs):
        """
        Execute function with exponential backoff retry on rate limit errors.

        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Function result
        """
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "rate" in error_str or "429" in error_str

                if attempt == max_retries:
                    print(f"Max retries exceeded ({max_retries})")
                    raise e

                if is_rate_limit:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0.5, 1.5)
                    delay = base_delay * jitter
                    print(f"Rate limit hit. Waiting {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise e

    def _encode_video_to_base64(self, video_path: str) -> tuple[str, str]:
        """
        Encode video file to base64.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (base64_data, mime_type)
        """
        mime_type = mimetypes.guess_type(video_path)[0]
        if not mime_type:
            mime_type = "video/mp4"

        with open(video_path, "rb") as video_file:
            video_data = video_file.read()

        base64_data = base64.b64encode(video_data).decode("utf-8")
        return base64_data, mime_type

    def _generate_chunk_prompt(self, chunk_info: Dict) -> str:
        """
        Generate appropriate prompt for video chunk analysis using XML template.

        Args:
            chunk_info: Dictionary with chunk information

        Returns:
            Prompt string for analysis
        """
        base_prompt = self._format_prompt(
            self.chunk_prompt_template,
            chunk_number=chunk_info['index'] + 1,
            total_chunks=chunk_info['total_chunks'],
            start_time_minutes=f"{chunk_info.get('start_time_minutes', 0):.1f}",
            end_time_minutes=f"{chunk_info.get('end_time_minutes', chunk_info['duration']/60):.1f}",
            duration_minutes=f"{chunk_info['duration']/60:.1f}"
        )
        if self.require_json_keyframes:
            postfix_path = os.path.join(self.prompts_dir, "common_keyframes_postfix.xml")
            if os.path.exists(postfix_path):
                with open(postfix_path, "r", encoding="utf-8") as pf:
                    base_prompt += "\n\n" + pf.read().strip()
            else:
                print("Warning: common_keyframes_postfix.xml not found; proceeding without JSON KeyFrames postfix")
        return base_prompt

    def analyze_video_chunk(self, video_path: str, chunk_info: Dict) -> str:
        """
        Analyze a single video chunk.

        Args:
            video_path: Path to the video chunk
            chunk_info: Dictionary with chunk information

        Returns:
            Analysis text from OpenRouter
        """
        print(f"Analyzing chunk {chunk_info['index']+1}/{chunk_info['total_chunks']}: {os.path.basename(video_path)}")

        # Encode video to base64
        base64_data, mime_type = self._encode_video_to_base64(video_path)

        # Generate appropriate prompt
        prompt = self._generate_chunk_prompt(chunk_info)

        print(f"Using prompt: {prompt[:100]}...")
        print(f"Using model: {self.model_name}")

        # Build message with video content
        data_url = f"data:{mime_type};base64,{base64_data}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": data_url}
                    }
                ]
            }
        ]

        def _generate_content():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content

        return self._retry_with_backoff(_generate_content)

    def combine_analyses(self, analyses: List[str], original_video_path: str) -> str:
        """
        Combine multiple chunk analyses into a single comprehensive analysis.

        Args:
            analyses: List of analysis texts for each chunk
            original_video_path: Path to the original video file

        Returns:
            Combined analysis text
        """
        print("Combining all chunk analyses into unified description...")

        # Prepare chunk analyses text
        chunk_analyses_text = ""
        for i, analysis in enumerate(analyses, 1):
            chunk_analyses_text += f"=== PART {i} ===\n{analysis}\n\n"

        # Format the prompt with chunk analyses
        combination_prompt = self._format_prompt(
            self.combine_prompt_template,
            chunk_analyses=chunk_analyses_text
        )

        messages = [
            {
                "role": "user",
                "content": combination_prompt
            }
        ]

        def _generate_combined_content():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content

        return self._retry_with_backoff(_generate_combined_content)

    def analyze_single_video(self, video_path: str) -> str:
        """
        Analyze a single video file (when no chunking is needed).

        Args:
            video_path: Path to the video file

        Returns:
            Analysis text from OpenRouter
        """
        print(f"Analyzing single video: {os.path.basename(video_path)}")

        # Encode video to base64
        base64_data, mime_type = self._encode_video_to_base64(video_path)

        # Use XML prompt template for single video (chunk 1 of 1)
        prompt = self._format_prompt(
            self.chunk_prompt_template,
            chunk_number=1,
            total_chunks=1,
            start_time_minutes="0.0",
            end_time_minutes="0.0",
            duration_minutes="0.0"
        )

        data_url = f"data:{mime_type};base64,{base64_data}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": data_url}
                    }
                ]
            }
        ]

        def _generate_single_content():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content

        return self._retry_with_backoff(_generate_single_content)
