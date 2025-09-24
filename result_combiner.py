# Result combination utilities for video analysis

import os
from typing import List, Tuple
from file_utils import save_analysis_to_file


class ResultCombiner:
    """Handles combining and saving video analysis results."""
    
    def __init__(self):
        """Initialize result combiner."""
        pass
    
    def save_chunk_analysis(self, analysis_text: str, chunk_path: str, chunk_info: dict, temp_dir: str) -> str:
        """
        Save analysis for a single chunk to temporary file.
        
        Args:
            analysis_text: Analysis text from Gemini
            chunk_path: Path to the video chunk
            chunk_info: Dictionary with chunk information
            temp_dir: Temporary directory path
            
        Returns:
            Path to the saved analysis file
        """
        # Generate analysis filename
        chunk_filename = os.path.basename(chunk_path)
        analysis_filename = os.path.splitext(chunk_filename)[0] + "_analysis.txt"
        analysis_path = os.path.join(temp_dir, analysis_filename)
        
        # Add metadata to analysis
        metadata = (
            f"Chunk {chunk_info['index']+1} of {chunk_info['total_chunks']}\n"
            f"Time range: {chunk_info['start_time_minutes']:.1f}-{chunk_info['end_time_minutes']:.1f} minutes\n"
            f"Duration: {chunk_info['duration']/60:.1f} minutes\n"
            f"Source file: {chunk_filename}\n"
            f"{'='*50}\n\n"
        )
        
        full_content = metadata + analysis_text
        
        # Save to file
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(full_content)
        
        print(f"Saved chunk analysis to: {analysis_filename}")
        return analysis_path
    
    def load_chunk_analyses(self, chunk_analysis_paths: List[str]) -> List[str]:
        """
        Load all chunk analyses from files.
        
        Args:
            chunk_analysis_paths: List of paths to chunk analysis files
            
        Returns:
            List of analysis texts
        """
        analyses = []
        
        for analysis_path in chunk_analysis_paths:
            if os.path.exists(analysis_path):
                with open(analysis_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    analyses.append(content)
                print(f"Loaded analysis from: {os.path.basename(analysis_path)}")
            else:
                print(f"Warning: Analysis file not found: {analysis_path}")
        
        return analyses
    
    def save_final_analysis(self, final_analysis: str, original_video_path: str) -> str:
        """
        Save the final combined analysis to output file.
        
        Args:
            final_analysis: Combined analysis text
            original_video_path: Path to the original video file
            
        Returns:
            Path to the saved final analysis file
        """
        # Generate output filename
        output_filename = os.path.splitext(original_video_path)[0] + ".txt"
        
        # Save the analysis
        save_analysis_to_file(final_analysis, output_filename, original_video_path)
        
        return output_filename
    
    def generate_summary_report(self, chunk_analysis_paths: List[str], final_analysis_path: str, 
                              original_video_path: str, processing_time: float) -> str:
        """
        Generate a summary report of the analysis process.
        
        Args:
            chunk_analysis_paths: List of paths to chunk analysis files
            final_analysis_path: Path to the final analysis file
            original_video_path: Path to the original video file
            processing_time: Total processing time in seconds
            
        Returns:
            Summary report text
        """
        total_chunks = len(chunk_analysis_paths)
        
        report = f"""
АНАЛИЗ ВИДЕО ЗАВЕРШЕН
{'='*50}

Исходный файл: {original_video_path}
Количество частей: {total_chunks}
Время обработки: {processing_time:.1f} секунд ({processing_time/60:.1f} минут)

ПРОМЕЖУТОЧНЫЕ ФАЙЛЫ:
"""
        
        for i, path in enumerate(chunk_analysis_paths, 1):
            filename = os.path.basename(path)
            size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
            report += f"  {i}. {filename} ({size_kb:.1f} KB)\n"
        
        report += f"""
ИТОГОВЫЙ ФАЙЛ:
  {os.path.basename(final_analysis_path)}
"""
        
        if os.path.exists(final_analysis_path):
            final_size_kb = os.path.getsize(final_analysis_path) / 1024
            report += f"  Размер: {final_size_kb:.1f} KB\n"
        
        return report
