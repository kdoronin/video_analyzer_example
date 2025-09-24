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

# Load environment variables
load_dotenv("config.env")


class GeminiAnalyzer:
    """Handles video analysis using Google Gemini API."""
    
    def __init__(self, project_id: str = None, location: str = None, model_name: str = None):
        """
        Initialize Gemini analyzer.
        
        Args:
            project_id: Google Cloud project ID (defaults to env variable)
            location: Vertex AI location (defaults to env variable)
            model_name: Gemini model name (defaults to env variable)
        """
        # Use environment variables as defaults
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.location = location or os.getenv("VERTEX_AI_LOCATION", "global")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
        
        if not self.project_id:
            raise ValueError("Google Cloud Project ID must be provided via GOOGLE_CLOUD_PROJECT_ID env variable or parameter")
        
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)
    
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
        Generate appropriate prompt for video chunk analysis.
        
        Args:
            chunk_info: Dictionary with chunk information
            
        Returns:
            Prompt string for analysis
        """
        if chunk_info['is_first'] and chunk_info['total_chunks'] == 1:
            # Single chunk (entire video)
            return "Детально опиши, что происходит на видео. ОБЯЗАТЕЛЬНО отвечай на русском языке."
        elif chunk_info['is_first']:
            # First chunk of multiple
            return (f"Это первая часть видео (из {chunk_info['total_chunks']} частей). "
                   f"Длительность этой части: {chunk_info['duration']/60:.1f} минут. "
                   f"Детально опиши, что происходит в этой части видео. ОБЯЗАТЕЛЬНО отвечай на русском языке.")
        elif chunk_info['is_last']:
            # Last chunk
            return (f"Это заключительная {chunk_info['index']+1}-я часть видео "
                   f"(из {chunk_info['total_chunks']} частей). "
                   f"Временные рамки: {chunk_info['start_time_minutes']:.1f}-{chunk_info['end_time_minutes']:.1f} минут. "
                   f"Длительность этой части: {chunk_info['duration']/60:.1f} минут. "
                   f"Детально опиши, что происходит в этой заключительной части видео. ОБЯЗАТЕЛЬНО отвечай на русском языке.")
        else:
            # Middle chunk
            return (f"Это {chunk_info['index']+1}-я часть видео "
                   f"(из {chunk_info['total_chunks']} частей). "
                   f"Временные рамки: {chunk_info['start_time_minutes']:.1f}-{chunk_info['end_time_minutes']:.1f} минут. "
                   f"Длительность этой части: {chunk_info['duration']/60:.1f} минут. "
                   f"Детально опиши, что происходит в этой части видео. ОБЯЗАТЕЛЬНО отвечай на русском языке.")
    
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
        
        # Prepare the prompt for combination
        combined_text = "АНАЛИЗЫ ЧАСТЕЙ ВИДЕО:\n\n"
        
        for i, analysis in enumerate(analyses, 1):
            combined_text += f"=== ЧАСТЬ {i} ===\n{analysis}\n\n"
        
        combination_prompt = (
            "Выше представлены отдельные анализы частей одного видео. "
            "Создай единый связный и подробный анализ всего видео, объединив информацию из всех частей. "
            "Структурируй описание логично, избегай повторений, но сохрани все важные детали. "
            "Результат должен читаться как анализ цельного видео, а не как набор отдельных частей. "
            "ОБЯЗАТЕЛЬНО отвечай на русском языке."
        )
        
        # Generate combined analysis with retry mechanism
        def _generate_combined_content():
            return self.model.generate_content([combined_text, combination_prompt])
        
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
        
        # Use simple prompt for single video
        prompt = "Детально опиши, что происходит на видео. ОБЯЗАТЕЛЬНО отвечай на русском языке."
        
        # Generate content with retry mechanism
        def _generate_single_content():
            return self.model.generate_content([video_part, prompt])
        
        response = self._retry_with_backoff(_generate_single_content)
        return response.text
