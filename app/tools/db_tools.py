from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from app.db_functions import store_analysis_in_db, add_to_master_surgeries_db, get_master_surgeries_db, get_master_surgeries_with_steps_db

class StoreAnalysisInput(BaseModel):
    """Input for storing analysis results in the database."""
    video_id: str = Field(
        description="Name/ID of the video file"
    )
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

@tool("store_analysis", args_schema=StoreAnalysisInput)
def store_analysis_tool(
    video_id: str,
    surgery_type: str,
    procedure_steps: List[str],
    description: str,
    summary: str
) -> str:
    """
    Store the surgery video analysis in the database.
    
    Args:
        video_id: Name/ID of the video file
        surgery_type: Type of surgery identified in the video
        procedure_steps: List of timestamped procedure steps
        description: Detailed description of the procedure
        summary: Concise summary of the procedure
        
    Returns:
        Confirmation message with the ID of the stored analysis
    """
    analysis_id = store_analysis_in_db(
        video_id=video_id,
        surgery_type=surgery_type,
        procedure_steps=procedure_steps,
        description=description,
        summary=summary
    )
    
    return f"Analysis stored successfully with ID: {analysis_id}"

class AddToMasterInput(BaseModel):
    """Input for adding to master surgeries collection."""
    surgery_type: str = Field(
        description="Type of surgery identified in the video"
    )
    procedure_steps: List[str] = Field(
        description="List of timestamped procedure steps"
    )
    summary: str = Field(
        description="Concise summary of the procedure"
    )
    master_id: Optional[str] = Field(
        default=None,
        description="Optional ID of an existing master surgery to update directly"
    )

@tool("get_master_surgeries")
def get_master_surgeries_tool() -> str:
    """
    Get all surgery details from the master surgeries collection.
    Use this tool before adding to master surgeries to check if similar surgeries already exist.
    This tool returns ALL surgeries in the master collection for comparison.
        
    Returns:
        JSON string with all surgery details from master surgeries collection
    """
    surgeries = get_master_surgeries_db()
    
    if not surgeries:
        return "No matching surgeries found in the master collection."
    
    # Format the result for better readability
    result = {
        "total_surgeries": len(surgeries),
        "surgeries": surgeries
    }
    
    import json
    return json.dumps(result, indent=2)

@tool("add_to_master_surgeries", args_schema=AddToMasterInput)
def add_to_master_surgeries_tool(
    surgery_type: str,
    procedure_steps: List[str],
    summary: str,
    master_id: Optional[str] = None
) -> str:
    """
    Add the analysis to the master surgeries collection.
    Before using this tool, use get_master_surgeries to check if similar surgeries exist.
    
    Args:
        surgery_type: Type of surgery identified in the video
        procedure_steps: List of timestamped procedure steps
        summary: Concise summary of the procedure
        master_id: Optional ID of an existing master surgery to update directly
        
    Returns:
        Confirmation message with the ID of the master surgery entry
    """
    result_id = add_to_master_surgeries_db(
        surgery_type=surgery_type,
        procedure_steps=procedure_steps,
        summary=summary,
        master_id=master_id
    )
    
    return f"Added to master surgeries with ID: {result_id}"




@tool("get_master_surgeries_with_steps")
def get_master_surgeries_with_steps_tool() -> str:
    """
    Get all surgery details from the master surgeries collection.
    Use this tool before adding to master surgeries to check if similar surgeries already exist.
    This tool returns ALL surgeries in the master collection for comparison.
        
    Returns:
        JSON string with all surgery details from master surgeries collection
    """
    surgeries = get_master_surgeries_with_steps_db()
    
    if not surgeries:
        return "No matching surgeries found in the master collection."
    
    # Format the result for better readability
    result = {
        "total_surgeries": len(surgeries),
        "surgeries": surgeries
    }
    
    import json
    return json.dumps(result, indent=2)
