from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
from google.cloud import aiplatform
from google.oauth2 import service_account
import os
import tempfile
import subprocess
import math
import time
from typing import List, Tuple
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
from datetime import datetime
from app.config import get_settings

load_dotenv()

# Rate limiting configuration
RATE_LIMIT_DELAY = 1.0  # seconds between requests
EMBEDDING_RATE_LIMIT = 30  # requests per minute for text-embedding-gecko
last_request_time = 0

class RateLimitError(Exception):
    """Custom exception for rate limiting"""
    pass

def rate_limit():
    """Simple rate limiting function"""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - time_since_last)
    last_request_time = time.time()

def init_vertex_ai(project_id: str, location: str):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        vertexai.init(project=project_id, location=location, credentials=credentials)
    except Exception as e:
        raise Exception(f"Failed to initialize Vertex AI: {str(e)}")

def split_video(video_bytes: bytes, chunk_duration: int = 600) -> List[Tuple[bytes, int, int]]:
    """
    Split video into chunks of specified duration (in seconds)
    Returns list of tuples: (chunk_bytes, start_time, end_time)
    """
    chunks = []
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
        temp_input.write(video_bytes)
        temp_input_path = temp_input.name
    
    try:
        # Get video duration using ffprobe
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            temp_input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        
        num_chunks = math.ceil(duration / chunk_duration)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
                output_path = temp_output.name
            
            # Extract video segment using ffmpeg
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', temp_input_path,
                '-t', str(end_time - start_time),
                '-c', 'copy',
                '-y',
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read the chunk bytes
            with open(output_path, 'rb') as f:
                chunk_bytes = f.read()
            
            chunks.append((chunk_bytes, start_time, end_time))
            
            # Clean up
            os.unlink(output_path)
            
    finally:
        # Clean up
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
    
    return chunks

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(ResourceExhausted),
    reraise=True
)
def analyze_video_chunk(chunk: Tuple[bytes, int, int]) -> str:
    """Analyze a single video chunk with enhanced prompting for surgical procedures"""
    chunk_bytes, start_time, end_time = chunk
    
    SYSTEM_INSTRUCTIONS = f"""
    You are analyzing a surgical/medical procedure video segment from {format_timestamp(start_time)} to {format_timestamp(end_time)}.
    
    **CRITICAL: ONLY describe what you can DIRECTLY SEE in the video frames.**
    
    Format your response as:
    
    🕒 {format_timestamp(start_time)} – {format_timestamp(end_time)}
    Process: [Brief name of the surgical step/procedure observed]
    Explanation: [Detailed description of exactly what is happening - include specific actions, instruments used, anatomical structures visible, and any techniques demonstrated]
    
    **STRICT RULES - NO EXCEPTIONS:**
    1. ONLY describe what is DIRECTLY VISIBLE in the video
    2. DO NOT make assumptions, inferences, or educated guesses
    3. DO NOT use general medical knowledge to fill in gaps
    4. DO NOT hallucinate or illustrate content that is not visible
    5. If the view is unclear, state "[View unclear/obstructed]"
    6. If you cannot see what is happening, say "Not clearly visible"
    7. Be consistent - the same video segment should always return the same analysis
    8. Use proper medical terminology ONLY for what you can see
    9. Do not assume the purpose or goal of the procedure
    10. Do not add information based on what you think should happen
    
    **Focus ONLY on:**
    - Surgical instruments being used (if visible)
    - Anatomical structures visible
    - Specific surgical techniques (if clearly visible)
    - Step-by-step actions (only what you can see)
    
    **If you cannot see clearly, say so rather than guessing.**
    """
    
    try:
        rate_limit()  # Apply rate limiting
        model = GenerativeModel('gemini-2.5-flash-preview-05-20')
        video_part = Part.from_data(data=chunk_bytes, mime_type='video/mp4')
        prompt_part = Part.from_text(SYSTEM_INSTRUCTIONS)
        
        response = model.generate_content(
            [video_part, prompt_part],
            generation_config={"temperature": 0.2}  # Lower temperature for consistency
        )
        
        # Add a small delay between chunks to avoid rate limiting
        time.sleep(1.0)
        
        return response.text
    except ResourceExhausted as e:
        print(f"Rate limit exceeded for chunk {format_timestamp(start_time)}-{format_timestamp(end_time)}, retrying...")
        # Add exponential backoff
        time.sleep(5)
        raise
    except Exception as e:
        print(f"Error analyzing video chunk {format_timestamp(start_time)}-{format_timestamp(end_time)}: {e}")
        return f"""🕒 {format_timestamp(start_time)} – {format_timestamp(end_time)}
Process: Analysis Error
Explanation: [Error analyzing this segment: {str(e)}]"""

def generate_summary(successful_analyses: List[str]) -> str:
    """Generate a structured summary from the analysis results"""
    
    summary_prompt = """
    Based on the following timestamped surgical procedure analysis, create a concise summary that:
    
    1. Identifies the type of surgical procedure
    2. Lists the main procedural phases in chronological order
    3. Notes key anatomical structures involved
    4. Highlights any educational aspects
    
    Format as:
    ✅ Summary
    [Brief description of the complete surgical procedure with main phases]
    
    Here are the detailed observations:
    """ + "\n\n".join(successful_analyses)
    
    try:
        rate_limit()
        model = GenerativeModel('gemini-2.5-flash-preview-05-20')
        summary_response = model.generate_content(
            summary_prompt,
            generation_config={"temperature": 0.3}
        )
        return summary_response.text
    except Exception as e:
        return f"✅ Summary\nUnable to generate summary due to error: {str(e)}"

def analyze_video(video_bytes: bytes) -> str:
    """
    Analyze a surgical video by splitting it into chunks and providing structured analysis
    matching the expected format with timestamps and procedural breakdown.
    
    Args:
        video_bytes: The video file content as bytes
        
    Returns:
        str: Structured analysis of the video with timestamps and procedural breakdown
    """
    settings = get_settings()
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    
    try:
        # Initialize Vertex AI with error handling
        init_vertex_ai(project_id, location)
        
        # Split video into 10-minute chunks (adjust as needed)
        print("🔄 Splitting video into chunks...")
        chunks = split_video(video_bytes, chunk_duration=600)
        print(f"✅ Split video into {len(chunks)} chunks for processing")
        
        # Process each chunk sequentially
        results = []
        successful_results = []
        
        for i, chunk in enumerate(chunks):
            chunk_bytes, start_time, end_time = chunk
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    print(f"🔍 Processing chunk {i+1}/{len(chunks)} ({format_timestamp(start_time)}-{format_timestamp(end_time)})...")
                    result = analyze_video_chunk(chunk)
                    
                    if result and len(result.strip()) > 20:  # Basic validation
                        results.append(result)
                        successful_results.append(result)
                        success = True
                        print(f"✅ Successfully processed chunk {i+1}")
                        
                        # Add delay between successful chunks to avoid rate limiting
                        if i < len(chunks) - 1:
                            time.sleep(2.0)
                    else:
                        raise ValueError("Empty or invalid response from API")
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        error_segment = f"""🕒 {format_timestamp(start_time)} – {format_timestamp(end_time)}
Process: Processing Failed
Explanation: [Unable to analyze this segment after {max_retries} attempts: {str(e)}]"""
                        results.append(error_segment)
                        print(f"❌ Failed to process chunk {i+1} after {max_retries} attempts")
                        time.sleep(5)
                    else:
                        wait_time = 5 * retry_count
                        print(f"🔄 Retry {retry_count}/{max_retries} for chunk {i+1} after error: {str(e)}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
        
        # Generate structured output
        if successful_results:
            print("📊 Generating summary of analysis...")
            summary = generate_summary(successful_results)
            
            # Combine all results with clear section headers
            final_analysis = "## Video Analysis Results\n\n"
            final_analysis += "### Detailed Timestamped Analysis\n\n"
            final_analysis += "\n\n".join(results)
            
            if summary:
                final_analysis += "\n\n### Summary of Findings\n\n"
                final_analysis += summary
            
            print("✅ Video analysis completed successfully")
            return final_analysis
        else:
            error_msg = """❌ Analysis Failed
Unable to analyze any video segments. This could be due to:
- API quota limitations
- Video format incompatibility
- Network connectivity issues
- Video content not suitable for analysis

Please check your setup and try again with a different video."""
            print(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"""❌ Video Analysis Error
Error in video analysis pipeline: {str(e)}

Technical details:
- Error type: {type(e).__name__}
- Timestamp: {datetime.now().isoformat()}
- Project: {project_id}
- Location: {location}

Please check:
1. Your API quota and billing status
2. Video file format and size (max 10GB)
3. Internet connectivity
4. Google Cloud project configuration

If the problem persists, contact support with the error details above."""
        print(f"❌ Error in video analysis pipeline: {e}")
        return error_msg