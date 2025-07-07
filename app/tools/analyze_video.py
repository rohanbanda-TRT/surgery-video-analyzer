from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from app.vertex_ai_client import analyze_video as real_analyze_video
@tool
def analyze_video_tool(video_bytes: bytes) -> str:
    """Analyze a surgical video using Vertex AI. Args: video_bytes (bytes): The video file content."""
    return real_analyze_video(video_bytes)

class AddToMasterInput(BaseModel):
    """Input for adding analysis to master surgeries."""
    surgery_type: str = Field(
        description="Type of surgery identified in the video"
    )
    procedure_steps: List[str] = Field(
        description="List of timestamped procedure steps"
    )
    summary: str = Field(
        description="Concise summary of the procedure"
    )

@tool("add_to_master_surgeries", args_schema=AddToMasterInput)
def add_to_master_surgeries(
    surgery_type: str,
    procedure_steps: List[str],
    summary: str
) -> str:
    """
    Add the analysis to the master surgeries collection.
    
    Args:
        surgery_type: Type of surgery identified in the video
        procedure_steps: List of timestamped procedure steps
        summary: Concise summary of the procedure
        
    Returns:
        Confirmation message
    """
    try:
        from app.mongodb_client import mongodb_client
        
        # Add to master surgeries
        master_id = mongodb_client.add_to_master_surgeries(
            surgery_type=surgery_type,
            procedure_steps=procedure_steps,
            summary=summary
        )
        
        return f"Added to master surgeries with ID: {master_id}"
    except Exception as e:
        error_message = f"Error adding to master surgeries: {str(e)}"
        print(f"❌ {error_message}")
        return error_message
