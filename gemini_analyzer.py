# Gemini API analyzer for video content analysis

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import mimetypes
import os
import time
import random
from typing import Dict, List
from google.api_core.exceptions import TooManyRequests, ResourceExhausted
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

# Load environment variables
load_dotenv("config.env")


class GeminiAnalyzer:
    """Handles video analysis using Google Gemini API."""
    
    def __init__(self, project_id: str = None, location: str = None, model_name: str = None, prompt_type: str = None, require_json_keyframes: bool = None):
        """
        Initialize Gemini analyzer.
        
        Args:
            project_id: Google Cloud project ID (defaults to env variable)
            location: Vertex AI location (defaults to env variable)
            model_name: Gemini model name (defaults to env variable)
            prompt_type: One of [general, lecture, meeting, presentation, tutorial, marketing, language_lesson, interview]
            require_json_keyframes: When True, ask model to ensure KeyFrames are emitted as JSON
        """
        # Use environment variables as defaults
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.location = location or os.getenv("VERTEX_AI_LOCATION", "global")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
        
        if not self.project_id:
            raise ValueError("Google Cloud Project ID must be provided via GOOGLE_CLOUD_PROJECT_ID env variable or parameter")
        
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)
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
            "voiceover": "chunk_analysis_voiceover.xml",
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
            except (TooManyRequests, ResourceExhausted) as e:
                if attempt == max_retries:
                    print(f"❌ Превышено максимальное количество попыток ({max_retries})")
                    raise e
                
                # Calculate delay with exponential backoff + jitter
                base_delay = 2 ** attempt  # 2, 4, 8, 16, 32 seconds
                jitter = random.uniform(0.5, 1.5)  # Random factor to avoid thundering herd
                delay = base_delay * jitter
                
                print(f"⏳ Получена ошибка 429 (лимит API). Пауза {delay:.1f} секунд... (попытка {attempt + 1}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                # For other exceptions, don't retry
                raise e
    
    def _create_video_part(self, video_path: str) -> Part:
        """
        Create a Part object from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Part object for Gemini API
        """
        # Get MIME type
        mime_type = mimetypes.guess_type(video_path)[0]
        if not mime_type:
            mime_type = "video/mp4"  # Default to mp4
        
        # Read video data
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
        
        return Part.from_data(video_data, mime_type=mime_type)
    
    def _generate_chunk_prompt(self, chunk_info: Dict) -> str:
        """
        Generate appropriate prompt for video chunk analysis using XML template.
        
        Args:
            chunk_info: Dictionary with chunk information
            
        Returns:
            Prompt string for analysis
        """
        # Format the prompt with chunk information
        base_prompt = self._format_prompt(
            self.chunk_prompt_template,
            chunk_number=chunk_info['index'] + 1,
            total_chunks=chunk_info['total_chunks'],
            start_time_minutes=f"{chunk_info.get('start_time_minutes', 0):.1f}",
            end_time_minutes=f"{chunk_info.get('end_time_minutes', chunk_info['duration']/60):.1f}",
            duration_minutes=f"{chunk_info['duration']/60:.1f}"
        )
        if self.require_json_keyframes:
            # Append shared XML postfix with JSON KeyFrames instructions
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
            Analysis text from Gemini
        """
        print(f"Analyzing chunk {chunk_info['index']+1}/{chunk_info['total_chunks']}: {os.path.basename(video_path)}")
        
        # Create video part
        video_part = self._create_video_part(video_path)
        
        # Generate appropriate prompt
        prompt = self._generate_chunk_prompt(chunk_info)
        
        print(f"Using prompt: {prompt[:100]}...")
        
        # Generate content with retry mechanism
        def _generate_content():
            return self.model.generate_content([video_part, prompt])
        
        response = self._retry_with_backoff(_generate_content)
        return response.text
    
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
            chunk_analyses_text += f"=== ЧАСТЬ {i} ===\n{analysis}\n\n"
        
        # Format the prompt with chunk analyses
        combination_prompt = self._format_prompt(
            self.combine_prompt_template,
            chunk_analyses=chunk_analyses_text
        )
        
        # Generate combined analysis with retry mechanism
        def _generate_combined_content():
            return self.model.generate_content(combination_prompt)
        
        response = self._retry_with_backoff(_generate_combined_content)
        return response.text
    
    def analyze_single_video(self, video_path: str) -> str:
        """
        Analyze a single video file (when no chunking is needed).
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Analysis text from Gemini
        """
        print(f"Analyzing single video: {os.path.basename(video_path)}")
        
        # Create video part
        video_part = self._create_video_part(video_path)
        
        # Use XML prompt template for single video (chunk 1 of 1)
        prompt = self._format_prompt(
            self.chunk_prompt_template,
            chunk_number=1,
            total_chunks=1,
            start_time_minutes="0.0",
            end_time_minutes="0.0",
            duration_minutes="0.0"
        )
        
        # Generate content with retry mechanism
        def _generate_single_content():
            return self.model.generate_content([video_part, prompt])
        
        response = self._retry_with_backoff(_generate_single_content)
        return response.text
