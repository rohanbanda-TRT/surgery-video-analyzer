from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import time
import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from app.agent import analyze_surgury_analysis, comparison_surgery
from app.config import get_settings
from app.vertex_ai_client import analyze_video as vertex_analyze_video

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")

app = FastAPI(title="Surgery Video Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "status": "online",
        "service": "Surgery Video Analysis API",
        "version": "1.0.0"
    }

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """
    Upload a surgical video, analyze it using Vertex AI, and format the result with the agent.
    Returns both the raw analysis and the agent-formatted output.
    """
    try:
        # Validate video file
        if not video.content_type.startswith('video/') and not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.error(f"File rejected: {video.filename} (type: {video.content_type})")
            raise HTTPException(status_code=400, detail="File must be a video")

        # Read the video file
        video_content = await video.read()
        if len(video_content) == 0:
            logger.error("Uploaded video file is empty.")
            raise HTTPException(status_code=400, detail="Video file is empty")

        logger.info(f"Processing video: {video.filename}, size: {len(video_content)} bytes")
        session_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()

        # Step 1: Raw analysis with Vertex AI
        logger.info("Starting raw video analysis with Vertex AI pipeline...")
        raw_analysis = vertex_analyze_video(video_content)
        
        # Format the analysis with the agent and store in database
        agent_input = {
            "messages": [HumanMessage(content=f"Format the following surgical video analysis into a clear, professional medical report and store it in the database:\n\n{raw_analysis}\n\nVideo ID: {video.filename}")]
        }
        agent_result = analyze_surgury_analysis.invoke(agent_input)
        formatted_output = agent_result["output"] if isinstance(agent_result, dict) and "output" in agent_result else str(agent_result)
        
        processing_time = time.time() - start_time

        logger.info(f"Video analysis and agent formatting completed in {processing_time:.2f} seconds")

        return {
            "video_id": video.filename,
            "raw_analysis": raw_analysis,
            "agent_output": formatted_output,
            "processing_time_seconds": processing_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    

@app.post("/compare-video")
async def compare_video(video: UploadFile = File(...)):
    """
    Upload a surgical video, analyze it using Vertex AI, and format the result with the agent.
    Returns both the raw analysis and the agent-formatted output.
    """
    try:
        # Validate video file
        if not video.content_type.startswith('video/') and not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.error(f"File rejected: {video.filename} (type: {video.content_type})")
            raise HTTPException(status_code=400, detail="File must be a video")

        # Read the video file
        video_content = await video.read()
        if len(video_content) == 0:
            logger.error("Uploaded video file is empty.")
            raise HTTPException(status_code=400, detail="Video file is empty")

        logger.info(f"Processing video: {video.filename}, size: {len(video_content)} bytes")
        session_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()

        # Step 1: Raw analysis with Vertex AI
        logger.info("Starting raw video analysis with Vertex AI pipeline...")
        raw_analysis = vertex_analyze_video(video_content)
        
        # Provide the analysis to the agent for comparison
        agent_input = {
            "messages": [HumanMessage(content=f"Here is the new surgical analysis. Please process it according to your instructions:\n\n{raw_analysis}\n\nVideo ID: {video.filename}")]
        }
        agent_result = comparison_surgery.invoke(agent_input)
        formatted_output = agent_result["output"] if isinstance(agent_result, dict) and "output" in agent_result else str(agent_result)
        
        processing_time = time.time() - start_time

        logger.info(f"Video analysis and agent formatting completed in {processing_time:.2f} seconds")

        return {
            "video_id": video.filename,
            "raw_analysis": raw_analysis,
            "agent_output": formatted_output,
            "processing_time_seconds": processing_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting Surgery Video Analysis API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
