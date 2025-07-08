import cv2
import time
import base64
import threading
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
from google.oauth2 import service_account
import queue
import asyncio
# Removed settings import - using environment variables directly

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("realtime_video_analyzer")

# Rate limiting configuration
RATE_LIMIT_DELAY = 1.0  # seconds between requests
last_request_time = 0

# Using environment variables directly

class RealtimeVideoAnalyzer:
    """
    Class for real-time video analysis using Google Generative AI with queue-based pipeline
    """
    def __init__(self):
        self.model = None
        self.video_capture = None
        self.is_running = False
        self.capture_thread = None
        self.analysis_thread = None
        self.current_frame = None
        self.current_analysis = "Initializing analysis..."
        self.frame_interval = 1.0  # Capture one frame per second for analysis
        self.last_capture_time = 0
        
        # Create queues for the pipeline
        self.frame_queue = queue.Queue(maxsize=10)  # Queue for frames to be analyzed
        self.result_queue = queue.Queue()  # Queue for analysis results
        
        self.init_vertex_ai()

    def init_vertex_ai(self):
        """Initialize Vertex AI with credentials"""
        try:
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
                
            credentials = service_account.Credentials.from_service_account_file(
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            )
            vertexai.init(project=project_id, location=location, credentials=credentials)
            self.model = GenerativeModel('gemini-2.5-flash-preview-05-20')
            logger.info("✅ Vertex AI initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI: {str(e)}")
            raise Exception(f"Failed to initialize Vertex AI: {str(e)}")

    def start_capture(self, camera_index=0):
        """Start video capture from webcam with separate threads for capture and analysis"""
        try:
            self.video_capture = cv2.VideoCapture(camera_index)
            if not self.video_capture.isOpened():
                raise Exception(f"Could not open video capture device {camera_index}")
            
            # Clear any existing items in the queues
            while not self.frame_queue.empty():
                self.frame_queue.get()
            while not self.result_queue.empty():
                self.result_queue.get()
            
            self.is_running = True
            
            # Start capture thread - continuously captures frames without delay
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            # Start analysis thread - processes frames from the queue
            self.analysis_thread = threading.Thread(target=self.analyze_frames_from_queue)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
            logger.info(f"✅ Started video capture pipeline from camera {camera_index}")
            return True
        except Exception as e:
            logger.error(f"❌ Error starting video capture: {str(e)}")
            return False

    def stop_capture(self):
        """Stop video capture and analysis threads"""
        self.is_running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=1.0)
        
        # Release video capture
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        logger.info("✅ Stopped video capture pipeline")
        
    def capture_frames(self):
        """Continuously capture frames in a separate thread without delay"""
        while self.is_running and self.video_capture:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Store the latest frame for display regardless of analysis
                self.current_frame = frame
                
                # Check if it's time to capture a frame for analysis (once per second)
                current_time = time.time()
                if current_time - self.last_capture_time >= self.frame_interval:
                    # Try to add to queue without blocking
                    try:
                        if not self.frame_queue.full():
                            self.frame_queue.put_nowait(frame.copy())
                            self.last_capture_time = current_time
                            logger.debug(f"Frame queued for analysis at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                    except queue.Full:
                        # Queue is full, skip this frame for analysis
                        pass
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in capture thread: {str(e)}")
                time.sleep(0.5)

    def rate_limit(self):
        """Simple rate limiting function"""
        global last_request_time
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last)
        last_request_time = time.time()

    def analyze_frame(self, frame):
        """Analyze a single frame with Gemini AI"""
        try:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            # System instructions for surgical video analysis
            system_instructions = """
            You are analyzing a surgical/medical procedure in real-time.
            
            **CRITICAL: ONLY describe what you can DIRECTLY SEE in the video frame.**
            
            Format your response as:
            
            Process: [Brief name of the surgical step/procedure observed]
            Explanation: [Detailed description of exactly what is happening - include specific actions, instruments used, anatomical structures visible, and any techniques demonstrated]
            
            **STRICT RULES - NO EXCEPTIONS:**
            1. ONLY describe what is DIRECTLY VISIBLE in the frame
            2. DO NOT make assumptions, inferences, or educated guesses
            3. DO NOT use general medical knowledge to fill in gaps
            4. DO NOT hallucinate or illustrate content that is not visible
            5. If the view is unclear, state "[View unclear/obstructed]"
            6. If you cannot see what is happening, say "Not clearly visible"
            7. Use proper medical terminology ONLY for what you can see
            8. Keep your response concise and focused
            
            **Focus ONLY on:**
            - Surgical instruments being used (if visible)
            - Anatomical structures visible
            - Specific surgical techniques (if clearly visible)
            - Step-by-step actions (only what you can see)
            
            **If you cannot see clearly, say so rather than guessing.**
            """
            
            self.rate_limit()  # Apply rate limiting
            
            # Create parts for the model
            video_part = Part.from_data(data=image_bytes, mime_type='image/jpeg')
            prompt_part = Part.from_text(system_instructions)
            
            # Generate content
            response = self.model.generate_content(
                [video_part, prompt_part],
                generation_config={"temperature": 0.2}  # Lower temperature for consistency
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing frame: {str(e)}")
            return f"Error analyzing frame: {str(e)}"

    def analyze_frames_from_queue(self):
        """Process frames from the queue in a separate thread"""
        while self.is_running:
            try:
                # Get a frame from the queue with a timeout
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    # No frames in queue, continue waiting
                    continue
                
                # Process the frame
                analysis = self.analyze_frame(frame)
                self.current_analysis = analysis
                
                # Put the result in the result queue for potential WebSocket broadcasting
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                self.result_queue.put({
                    'timestamp': timestamp,
                    'analysis': analysis
                })
                
                logger.info(f"Frame analyzed at {timestamp}")
                
                # Mark the task as done
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in analysis thread: {str(e)}")
                time.sleep(0.5)

    def get_current_frame_with_analysis(self):
        """Get the current frame with analysis overlay"""
        if self.current_frame is None:
            return None
        
        frame = self.current_frame.copy()
        
        # Add analysis text to the frame
        if self.current_analysis:
            # Convert OpenCV BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
            
            # Add semi-transparent background for text
            h, w = frame.shape[:2]
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.rectangle([(0, h-200), (w, h)], fill=(0, 0, 0, 160))
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
            
            # Wrap text to fit the frame width
            text = self.current_analysis
            lines = []
            current_line = ""
            max_width = w - 20
            
            for word in text.split():
                test_line = current_line + word + " "
                text_width = draw.textlength(test_line, font=font)
                
                if text_width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word + " "
            
            if current_line:
                lines.append(current_line)
            
            # Draw text lines
            y_position = h - 180
            for line in lines[:8]:  # Limit to 8 lines
                draw.text((10, y_position), line, font=font, fill=(255, 255, 255))
                y_position += 20
            
            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return frame
        
    def get_latest_analysis_result(self):
        """Get the latest analysis result without removing it from the queue"""
        if not self.result_queue.empty():
            # Peek at the latest result without removing it
            result = self.result_queue.queue[-1] if self.result_queue.qsize() > 0 else None
            return result
        return None

    def get_current_analysis(self):
        """Get the current analysis text"""
        return self.current_analysis
