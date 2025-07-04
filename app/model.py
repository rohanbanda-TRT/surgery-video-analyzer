from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SurgeryVideoAnalysisResult(BaseModel):
    """Schema for surgery video analysis results."""
    surgery_type: str = Field(description="Type of surgery identified in the video")
    procedure_steps: List[str] = Field(description="List of timestamped procedure steps")
    description: str = Field(description="Detailed description of the procedure")
    summary: str = Field(description="Concise summary of the procedure")

class VideoChunkAnalysis(BaseModel):
    """Schema for individual video chunk analysis."""
    chunk_index: int = Field(description="Index of the video chunk")
    start_time: str = Field(description="Start time of the chunk in format HH:MM:SS")
    end_time: str = Field(description="End time of the chunk in format HH:MM:SS")
    analysis: str = Field(description="Raw analysis text for this chunk")
    
class CombinedVideoAnalysis(BaseModel):
    """Schema for combined analysis from multiple video chunks."""
    video_id: str = Field(description="Name/ID of the video file")
    total_chunks: int = Field(description="Total number of chunks analyzed")
    chunk_analyses: List[VideoChunkAnalysis] = Field(description="List of individual chunk analyses")
    combined_analysis: str = Field(description="Combined analysis text from all chunks")
    
class SurgeryAnalysisState(BaseModel):
    """Schema for the state of the surgery analysis agent."""
    video_id: str = Field(description="Name/ID of the video file")
    video_bytes: Optional[bytes] = Field(None, description="Binary content of the video file")
    video_chunks: Optional[List[bytes]] = Field(None, description="List of video chunks")
    current_chunk_index: int = Field(0, description="Index of the current chunk being processed")
    chunk_analyses: List[VideoChunkAnalysis] = Field(default_factory=list, description="List of analyses for each chunk")
    combined_analysis: Optional[str] = Field(None, description="Combined analysis from all chunks")
    structured_result: Optional[SurgeryVideoAnalysisResult] = Field(None, description="Structured analysis result")
    workflow_stage: str = Field("initialized", description="Current stage in the workflow")
    error: Optional[str] = Field(None, description="Error message if any")
