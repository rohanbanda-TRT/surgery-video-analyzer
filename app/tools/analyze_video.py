from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from app.vertex_ai_client import analyze_video as real_analyze_video
from app.vertex_ai_client import analyze_video
@tool
def analyze_video_tool(video_bytes: bytes) -> str:
    """Analyze a surgical video using Vertex AI. Args: video_bytes (bytes): The video file content."""
    return real_analyze_video(video_bytes)

class AnalyzeVideoInput(BaseModel):
    """Input for the analyze_video tool."""
    video_bytes: bytes = Field(
        description="Binary content of the video file to analyze"
    )

class AnalyzeSurgeryVideo(BaseTool):
    """Tool for analyzing surgery videos using Google Vertex AI."""
    
    name: str = "analyze_surgery_video"
    description: str = (
        "Use this tool to analyze surgical videos and extract detailed information about "
        "the procedure, instruments used, and surgical steps. The tool will return a "
        "structured analysis with timestamps and descriptions."
    )
    args_schema: type[BaseModel] = AnalyzeVideoInput
    
    def _run(
        self,
        video_bytes: bytes,
        **kwargs
    ) -> str:
        """
        Run the video analysis using Google Vertex AI
        
        Args:
            video_bytes: Binary content of the video file
            
        Returns:
            Detailed analysis text from the Vertex AI model
        """
        try:
            # Call the analyze_video function from vertex_ai_client
            analysis_result = analyze_video(video_bytes)
            return analysis_result
            
        except Exception as e:
            error_message = f"Error analyzing video: {str(e)}"
            print(f"❌ {error_message}")
            return error_message
    
    async def _arun(self, video_bytes: bytes) -> str:
        """Async version is not implemented."""
        raise NotImplementedError("Async version of this tool is not implemented.")


class StoreAnalysisInput(BaseModel):
    """Input for storing analysis results in the database."""
    surgery_type: str = Field(
        description="Type of surgery identified in the video"
    )
    procedure_steps: List[str] = Field(
        description="List of timestamped procedure steps"
    )
    description: str = Field(
        description="Detailed description of the procedure"
    )
    summary: str = Field(
        description="Concise summary of the procedure"
    )
    video_id: str = Field(
        description="Name/ID of the video file"
    )

@tool("store_surgery_analysis", args_schema=StoreAnalysisInput)
def store_surgery_analysis(
    surgery_type: str,
    procedure_steps: List[str],
    description: str,
    summary: str,
    video_id: str
) -> str:
    """
    Store the surgery video analysis in the database.
    
    Args:
        surgery_type: Type of surgery identified in the video
        procedure_steps: List of timestamped procedure steps
        description: Detailed description of the procedure
        summary: Concise summary of the procedure
        video_id: Name/ID of the video file
        
    Returns:
        Confirmation message with the ID of the stored analysis
    """
    try:
        from app.mongodb_client import mongodb_client
        
        # Prepare the analysis data
        analysis_data = {
            "video_id": video_id,
            "surgery_type": surgery_type,
            "procedure_steps": procedure_steps,
            "description": description,
            "summary": summary
        }
        
        # Store in the database
        analysis_id = mongodb_client.store_analysis(analysis_data)
        
        return f"Analysis stored successfully with ID: {analysis_id}"
    except Exception as e:
        error_message = f"Error storing analysis: {str(e)}"
        print(f"❌ {error_message}")
        return error_message


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
