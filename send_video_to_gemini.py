# Advanced video analyzer with automatic chunking and analysis
# Splits long videos into chunks, analyzes each part separately, then combines results

import os
import time
import glob
from dotenv import load_dotenv
from video_processor import VideoProcessor
from gemini_analyzer import GeminiAnalyzer
from result_combiner import ResultCombiner
from file_utils import get_temp_directory, ensure_directory_exists

# Load environment variables
load_dotenv("config.env")


def find_video_files(video_directory: str) -> list[str]:
    """
    Find all video files in the specified directory.
    
    Args:
        video_directory: Path to the directory containing video files
        
    Returns:
        List of video file paths
    """
    # Supported video extensions
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    
    video_files = []
    for extension in video_extensions:
        pattern = os.path.join(video_directory, extension)
        video_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase extensions
        pattern_upper = os.path.join(video_directory, extension.upper())
        video_files.extend(glob.glob(pattern_upper, recursive=False))
    
    # Sort files for consistent processing order
    video_files.sort()
    return video_files


def process_single_video(video_path: str, chunk_duration_minutes: int) -> bool:
    """
    Process a single video file.
    
    Args:
        video_path: Path to the video file
        chunk_duration_minutes: Duration for video chunks
        
    Returns:
        True if processing was successful, False otherwise
    """
    video_name = os.path.basename(video_path)
    print(f"\n{'='*80}")
    print(f"🎬 Starting analysis for: {video_name}")
    print(f"📁 Temporary files will be stored in: {get_temp_directory()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Initialize components
        video_processor = VideoProcessor(chunk_duration_minutes)
        analyzer = GeminiAnalyzer()
        combiner = ResultCombiner()
        
        # Step 1: Split video into chunks if necessary
        print("🔪 Step 1: Processing video chunks...")
        chunk_paths = video_processor.split_video(video_path)
        
        # Step 2: Analyze each chunk
        print(f"\n🤖 Step 2: Analyzing {len(chunk_paths)} chunk(s) with Gemini...")
        chunk_analyses = []
        chunk_analysis_paths = []
        
        for i, chunk_path in enumerate(chunk_paths):
            # Get chunk information
            chunk_info = video_processor.get_chunk_info(chunk_path, i, len(chunk_paths))
            
            # Analyze the chunk
            analysis_text = analyzer.analyze_video_chunk(chunk_path, chunk_info)
            chunk_analyses.append(analysis_text)
            
            # Save chunk analysis to temporary file
            temp_dir = get_temp_directory()
            analysis_path = combiner.save_chunk_analysis(
                analysis_text, chunk_path, chunk_info, temp_dir
            )
            chunk_analysis_paths.append(analysis_path)
            
            print(f"✅ Completed analysis of chunk {i+1}/{len(chunk_paths)}")
        
        # Step 3: Combine analyses or use single analysis
        print(f"\n🔗 Step 3: Creating final analysis...")
        
        if len(chunk_analyses) > 1:
            # Multiple chunks - combine them
            final_analysis = analyzer.combine_analyses(chunk_analyses, video_path)
        else:
            # Single chunk - use as is
            final_analysis = chunk_analyses[0]
        
        # Step 4: Save final result
        print(f"\n💾 Step 4: Saving final results...")
        final_output_path = combiner.save_final_analysis(final_analysis, video_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate and display summary
        summary = combiner.generate_summary_report(
            chunk_analysis_paths, final_output_path, video_path, processing_time
        )
        
        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY for {video_name}!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        print(f"📄 Analysis saved to: {final_output_path}")
        
        # Show preview
        print(f"\n📖 Preview of analysis:")
        print("-" * 50)
        preview_length = 300
        if len(final_analysis) > preview_length:
            print(final_analysis[:preview_length] + "...")
        else:
            print(final_analysis)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {video_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all videos in the input directory."""
    
    # Configuration from environment variables
    video_directory = os.getenv("VIDEO_INPUT_DIRECTORY", "video")
    chunk_duration_minutes = int(os.getenv("CHUNK_DURATION_MINUTES", "10"))
    
    # Ensure video directory exists
    if not os.path.exists(video_directory):
        print(f"📁 Creating video directory: {video_directory}")
        ensure_directory_exists(video_directory)
        print(f"✨ Directory created! Please place your video files in '{video_directory}/' and run the script again.")
        return
    
    # Find all video files in the directory
    video_files = find_video_files(video_directory)
    
    if not video_files:
        print(f"📂 No video files found in '{video_directory}/' directory!")
        print(f"Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V")
        print(f"Please add video files to the '{video_directory}/' directory and try again.")
        return
    
    print(f"🎬 Found {len(video_files)} video file(s) to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(video_file)}")
    
    # Process each video file
    successful_count = 0
    failed_count = 0
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n🔄 Processing video {i}/{len(video_files)}")
        
        success = process_single_video(video_path, chunk_duration_minutes)
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    # Final summary
    total_processing_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"🏁 BATCH PROCESSING COMPLETED!")
    print(f"{'='*80}")
    print(f"📊 Summary:")
    print(f"   • Total videos: {len(video_files)}")
    print(f"   • Successfully processed: {successful_count}")
    print(f"   • Failed: {failed_count}")
    print(f"   • Total processing time: {total_processing_time:.1f} seconds ({total_processing_time/60:.1f} minutes)")
    
    if failed_count > 0:
        print(f"\n⚠️  Some videos failed to process. Check the error messages above.")
    else:
        print(f"\n🎉 All videos processed successfully!")


if __name__ == "__main__":
    main() 