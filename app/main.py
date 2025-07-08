from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
import logging
import time
import os
import cv2
import json
import asyncio
import queue
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from app.agent import analyze_surgury_analysis, comparison_surgery
from app.db_functions import get_master_surgeries_with_steps_db
from app.config import get_settings
from app.vertex_ai_client import analyze_video as vertex_analyze_video
from app.realtime_video_analyzer import RealtimeVideoAnalyzer

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

# Mount static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the HTML frontend for real-time video analysis"""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        return HTMLResponse(content="<html><body><h1>Error: Frontend file not found</h1></body></html>", status_code=404)

@app.get("/api/status")
async def api_status():
    """API status endpoint to check if the API is running."""
    return {
        "status": "online",
        "service": "Surgery Video Analysis API",
        "version": "1.0.0"
    }

@app.get("/master-surgeries/types")
def get_surgery_types():
    """Get all unique surgery types from master surgeries"""
    try:
        # Get all master surgeries with steps
        master_surgeries = get_master_surgeries_with_steps_db()
        
        # Extract unique surgery types
        surgery_types = []
        for surgery in master_surgeries:
            surgery_type = surgery.get("surgery_type", "")
            if surgery_type and surgery_type not in surgery_types:
                surgery_types.append(surgery_type)
        
        return {"surgery_types": surgery_types}
    except Exception as e:
        logger.error(f"Error getting surgery types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting surgery types: {str(e)}")

@app.get("/master-surgeries/steps")
def get_surgery_steps(surgery_type: str):
    """Get procedure steps for a specific surgery type"""
    try:
        # Get all master surgeries with steps
        master_surgeries = get_master_surgeries_with_steps_db()
        
        # Find the surgery with the matching type
        for surgery in master_surgeries:
            if surgery.get("surgery_type", "") == surgery_type:
                return {
                    "id": surgery.get("id", ""),
                    "surgery_type": surgery_type,
                    "procedure_steps": surgery.get("procedure_steps", []),
                    "summary": surgery.get("summary", "")
                }
        
        # If no matching surgery found
        raise HTTPException(status_code=404, detail=f"Surgery type '{surgery_type}' not found")
    except Exception as e:
        logger.error(f"Error getting surgery steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting surgery steps: {str(e)}")

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


# Initialize the real-time video analyzer
realtime_analyzer = RealtimeVideoAnalyzer()

@app.get("/realtime-video/start")
async def start_realtime_analysis(camera_index: int = 0):
    """
    Start real-time video analysis from webcam.
    Returns a status message indicating if the capture was successfully started.
    """
    try:
        # Stop any existing capture
        if realtime_analyzer.is_running:
            realtime_analyzer.stop_capture()
        
        # Start new capture
        success = realtime_analyzer.start_capture(camera_index)
        if success:
            return {"status": "success", "message": f"Started real-time video analysis on camera {camera_index}"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to start video capture on camera {camera_index}")
    except Exception as e:
        logger.error(f"Error starting real-time analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting real-time analysis: {str(e)}")

@app.get("/realtime-video/stop")
async def stop_realtime_analysis():
    """
    Stop real-time video analysis.
    Returns a status message indicating the capture was stopped.
    """
    try:
        realtime_analyzer.stop_capture()
        return {"status": "success", "message": "Stopped real-time video analysis"}
    except Exception as e:
        logger.error(f"Error stopping real-time analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error stopping real-time analysis: {str(e)}")

@app.get("/realtime-video/stream")
async def video_feed():
    """
    Stream the real-time video with analysis overlay.
    Returns a streaming response with the video feed.
    """
    if not realtime_analyzer.is_running:
        raise HTTPException(status_code=400, detail="Real-time video analysis is not running. Start it first.")
    
    async def generate_frames():
        while realtime_analyzer.is_running:
            frame = realtime_analyzer.get_current_frame_with_analysis()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
                
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                await asyncio.sleep(0.1)
                continue
                
            # Yield the frame in multipart/x-mixed-replace format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            await asyncio.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/realtime-video/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time analysis updates.
    Sends analysis results from the queue to connected clients.
    """
    await websocket.accept()
    
    try:
        while True:
            if realtime_analyzer.is_running:
                # Check if there are new analysis results in the queue
                try:
                    # Non-blocking check for new results
                    if not realtime_analyzer.result_queue.empty():
                        result = realtime_analyzer.result_queue.get_nowait()
                        
                        # Prepare the response data
                        response_data = {
                            "timestamp": result['timestamp'],
                            "analysis": result['analysis'],
                            "procedure_step": result.get('procedure_step', '')
                        }
                        
                        # Add comparison result if available
                        if 'comparison_result' in result:
                            response_data["comparison_result"] = result['comparison_result']
                        
                        # Send the result to the WebSocket client
                        await websocket.send_text(json.dumps(response_data))
                        
                        # Mark as done
                        realtime_analyzer.result_queue.task_done()
                except queue.Empty:
                    # No new results, continue
                    pass
            
            # Small wait to prevent CPU overuse
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting Surgery Video Analysis API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
