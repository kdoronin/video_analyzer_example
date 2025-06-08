# This script sends a video file to Gemini 2.5 Pro Preview 06-05 via Google Vertex AI API
# and asks the model to describe in detail what is happening in the video.

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import mimetypes
import os

# Initialize Vertex AI with your project and global location
vertexai.init(project="PROJECT_NAME", location="global")

# Load the Gemini 2.5 Pro model
model = GenerativeModel("gemini-2.5-pro-preview-06-05")

# Path to your video file
video_path = "FILE_PATH/FILE_NAME.mp4"  # Video file in the project root

# Check if file exists
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    exit(1)

# Generate output filename with .txt extension
output_filename = os.path.splitext(video_path)[0] + ".txt"

# Get MIME type
mime_type = mimetypes.guess_type(video_path)[0]
if not mime_type:
    mime_type = "video/mp4"  # Default to mp4

print(f"Processing video file: {video_path}")
print(f"MIME type: {mime_type}")
print(f"Output will be saved to: {output_filename}")

try:
    # Create a Part object from local file
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
    
    video_part = Part.from_data(video_data, mime_type=mime_type)
    
    # Send the video to the model with a prompt in Russian
    prompt = "детально опиши, что происходит на видео"
    
    print("Sending video to Gemini 2.5 Pro for analysis...")
    
    # Generate content
    response = model.generate_content([video_part, prompt])
    
    # Save the response to file
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(f"Video analysis for: {video_path}\n")
        output_file.write(f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("=" * 60 + "\n\n")
        output_file.write(response.text)
    
    # Print confirmation
    print(f"✅ Analysis completed and saved to: {output_filename}")
    print("\nPreview of the analysis:")
    print("-" * 50)
    print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
    
except Exception as e:
    print(f"Error: {e}") 