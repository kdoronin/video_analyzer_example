# File utilities for video analysis project

import os
import shutil
from typing import List, Tuple


def ensure_directory_exists(directory_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def get_temp_directory() -> str:
    """
    Get the temporary directory path for video chunks.
    
    Returns:
        Path to the temporary directory
    """
    temp_dir = "temporary"
    ensure_directory_exists(temp_dir)
    return temp_dir


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Remove all files from temporary directory.
    
    Args:
        temp_dir: Path to the temporary directory to clean up
    """
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def get_video_chunks_info(temp_dir: str) -> List[Tuple[str, str]]:
    """
    Get information about video chunks in temporary directory.
    
    Args:
        temp_dir: Path to the temporary directory
        
    Returns:
        List of tuples (chunk_filename, analysis_filename)
    """
    chunks_info = []
    if os.path.exists(temp_dir):
        video_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp4')]
        video_files.sort()  # Sort to ensure proper order
        
        for video_file in video_files:
            chunk_path = os.path.join(temp_dir, video_file)
            analysis_filename = os.path.splitext(video_file)[0] + "_analysis.txt"
            analysis_path = os.path.join(temp_dir, analysis_filename)
            chunks_info.append((chunk_path, analysis_path))
    
    return chunks_info


def generate_output_filename(video_path: str) -> str:
    """
    Generate output filename based on input video path.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Output filename with .txt extension
    """
    return os.path.splitext(video_path)[0] + ".txt"


def save_analysis_to_file(content: str, output_path: str, video_path: str) -> None:
    """
    Save analysis content to file with metadata.
    
    Args:
        content: Analysis content to save
        output_path: Path to save the analysis
        video_path: Original video file path for metadata
    """
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"Video analysis for: {video_path}\n")
        output_file.write(f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("=" * 60 + "\n\n")
        output_file.write(content)
