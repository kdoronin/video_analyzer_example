# Result combination utilities for video analysis

import os
import re
import json
import subprocess
from typing import List, Tuple, Optional, Dict, Any
from file_utils import save_analysis_to_file, ensure_directory_exists


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
    
    @staticmethod
    def extract_key_frames_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract a JSON object with key "key_frames" from model output text.
        Returns parsed dict or None if not found/parsable.
        """
        # 1) Look for fenced code blocks ```json ... ``` or ``` ... ```
        fence_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]
        for pat in fence_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict) and "key_frames" in data:
                        return data
                except Exception:
                    pass
        # 2) Fallback: find the first JSON-looking object containing "key_frames"
        json_like = re.findall(r"\{[\s\S]*?\}", text)
        for block in json_like:
            if "key_frames" in block:
                try:
                    data = json.loads(block)
                    if isinstance(data, dict) and "key_frames" in data:
                        return data
                except Exception:
                    continue
        return None

    @staticmethod
    def _sanitize_timecode_for_ffmpeg(timecode: str) -> str:
        """Take timecode possibly with ranges and return a single start time suitable for ffmpeg -ss.
        Accepts formats like 'hh:mm:ss', 'mm:ss', 'hh:mm:ss-hh:mm:ss'.
        """
        if not timecode:
            return "00:00:00"
        # Split on common range separators
        for sep in ["-", "–", "—", " to ", "→", ">>", "→ "]:
            if sep in timecode:
                timecode = timecode.split(sep)[0].strip()
                break
        return timecode.strip()

    @staticmethod
    def timecode_to_seconds(timecode: str) -> int:
        """Convert 'hh:mm:ss' or 'mm:ss' into total seconds. Handles ranges by taking start."""
        t = ResultCombiner._sanitize_timecode_for_ffmpeg(str(timecode))
        parts = t.split(":")
        try:
            if len(parts) == 3:
                h, m, s = [int(float(p)) for p in parts]
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m, s = [int(float(p)) for p in parts]
                return m * 60 + s
            # Fallback if single number
            return int(float(parts[0]))
        except Exception:
            return 0

    @staticmethod
    def seconds_to_timecode(seconds: int) -> str:
        """Convert seconds into zero-padded 'hh:mm:ss'."""
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _export_key_frame_image(self, video_path: str, timecode: str, out_path: str) -> bool:
        """Export a single frame from video at timecode to out_path using ffmpeg."""
        # Ensure output directory exists
        ensure_directory_exists(os.path.dirname(out_path))
        # ffmpeg -ss TIME -i INPUT -frames:v 1 -q:v 2 OUTPUT
        cmd = [
            "ffmpeg", "-y",
            "-ss", timecode,
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            print(f"Warning: failed to export key frame at {timecode}: {e}")
            return False

    def save_final_analysis(self, final_analysis: str, original_video_path: str, require_json_keyframes: bool = False, key_frames_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Save the final combined analysis to output file.
        
        Args:
            final_analysis: Combined analysis text
            original_video_path: Path to the original video file
            require_json_keyframes: If True, try to extract key frames JSON and export images
            
        Returns:
            Path to the saved final analysis file (.md)
        """
        # Prepare output directories and filenames
        video_dir = os.path.dirname(original_video_path)
        base_name = os.path.splitext(os.path.basename(original_video_path))[0]
        out_dir = os.path.join(video_dir, base_name)
        ensure_directory_exists(out_dir)
        
        final_output_path = os.path.join(out_dir, f"{base_name}.md")
        # Save analysis content as markdown (header + content)
        save_analysis_to_file(final_analysis, final_output_path, original_video_path)

        if require_json_keyframes:
            if key_frames_data is None:
                key_frames_data = ResultCombiner.extract_key_frames_json(final_analysis)
            if key_frames_data and isinstance(key_frames_data.get("key_frames"), list):
                # Save JSON file
                kf_dir = os.path.join(out_dir, "key_frames")
                ensure_directory_exists(kf_dir)
                kf_json_path = os.path.join(kf_dir, "key_frames.json")
                with open(kf_json_path, "w", encoding="utf-8") as jf:
                    json.dump(key_frames_data, jf, ensure_ascii=False, indent=2)
                print(f"Saved key frames JSON to: {kf_json_path}")

                # Export images for each frame
                for idx, frame in enumerate(key_frames_data["key_frames"], start=1):
                    tc = self._sanitize_timecode_for_ffmpeg(str(frame.get("timecode", "00:00:00")))
                    safe_tc = tc.replace(":", "-")
                    img_name = f"{idx:03d}_{safe_tc}.jpg"
                    img_path = os.path.join(kf_dir, img_name)
                    self._export_key_frame_image(original_video_path, tc, img_path)
            else:
                print("No parsable key_frames JSON found in the analysis output.")

        return final_output_path
    
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
