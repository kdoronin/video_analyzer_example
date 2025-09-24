# Video processing utilities for splitting videos into chunks using ffmpeg

import os
import subprocess
import json
from typing import List
from file_utils import get_temp_directory, cleanup_temp_directory


class VideoProcessor:
    """Handles video processing operations including splitting into chunks using ffmpeg."""
    
    def __init__(self, chunk_duration_minutes: int = 10):
        """
        Initialize video processor.
        
        Args:
            chunk_duration_minutes: Duration of each chunk in minutes
        """
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.temp_dir = get_temp_directory()
    
    def get_video_duration(self, video_path: str) -> float:
        """
        Get video duration in seconds using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
            
        except (subprocess.CalledProcessError, KeyError, ValueError) as e:
            print(f"Error getting video duration: {e}")
            raise
    
    def split_video(self, video_path: str, cleanup_existing: bool = True) -> List[str]:
        """
        Split video into chunks using ffmpeg if it's longer than chunk duration.
        
        Args:
            video_path: Path to the input video file
            cleanup_existing: Whether to clean up existing chunks before processing
            
        Returns:
            List of paths to video chunks (or original video if no splitting needed)
        """
        if cleanup_existing:
            cleanup_temp_directory(self.temp_dir)
        
        duration = self.get_video_duration(video_path)
        print(f"Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # If video is shorter than chunk duration, no need to split
        if duration <= self.chunk_duration_seconds:
            print("Video is shorter than chunk duration. No splitting needed.")
            return [video_path]
        
        # Calculate number of chunks needed
        num_chunks = int(duration // self.chunk_duration_seconds) + (1 if duration % self.chunk_duration_seconds > 0 else 0)
        print(f"Splitting video into {num_chunks} chunks of {self.chunk_duration_seconds/60} minutes each")
        
        chunk_paths = []
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        
        for i in range(num_chunks):
            start_time = i * self.chunk_duration_seconds
            chunk_filename = f"{base_filename}_chunk_{i+1:03d}.mp4"
            chunk_path = os.path.join(self.temp_dir, chunk_filename)
            
            print(f"Creating chunk {i+1}/{num_chunks}: starting at {start_time:.2f}s")
            
            # Use ffmpeg to extract the chunk
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_time),  # Start time
                '-t', str(self.chunk_duration_seconds),  # Duration
                '-c', 'copy',  # Copy streams without re-encoding for speed
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output file if exists
                chunk_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                chunk_paths.append(chunk_path)
                
            except subprocess.CalledProcessError as e:
                print(f"Error creating chunk {i+1}: {e}")
                continue
        
        print(f"Video split into {len(chunk_paths)} chunks successfully")
        return chunk_paths
    
    def get_chunk_info(self, chunk_path: str, chunk_index: int, total_chunks: int) -> dict:
        """
        Get information about a video chunk.
        
        Args:
            chunk_path: Path to the chunk file
            chunk_index: Index of the chunk (0-based)
            total_chunks: Total number of chunks
            
        Returns:
            Dictionary with chunk information
        """
        try:
            duration = self.get_video_duration(chunk_path)
        except:
            # Fallback to estimated duration
            duration = self.chunk_duration_seconds
        
        return {
            'path': chunk_path,
            'index': chunk_index,
            'total_chunks': total_chunks,
            'duration': duration,
            'start_time_minutes': chunk_index * (self.chunk_duration_seconds / 60),
            'end_time_minutes': (chunk_index * (self.chunk_duration_seconds / 60)) + (duration / 60),
            'is_first': chunk_index == 0,
            'is_last': chunk_index == total_chunks - 1
        }
