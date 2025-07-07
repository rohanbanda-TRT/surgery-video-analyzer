from typing import Dict, List, Any, Optional
from datetime import datetime
from app.mongodb_client import mongodb_client
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("db_functions")

def store_analysis_in_db(
    video_id: str,
    surgery_type: str,
    procedure_steps: List[str],
    description: str,
    summary: str
) -> str:
    """
    Store the video analysis in the MongoDB analysis collection
    
    Args:
        video_id: Name/ID of the video file
        surgery_type: Type of surgery identified
        procedure_steps: List of timestamped procedure steps
        description: Detailed description of the procedure
        summary: Concise summary of the procedure
        
    Returns:
        ID of the stored analysis document
    """
    analysis_data = {
        "video_id": video_id,
        "surgery_type": surgery_type,
        "procedure_steps": procedure_steps,
        "description": description,
        "summary": summary
    }
    
    try:
        analysis_id = mongodb_client.store_analysis(analysis_data)
        logger.info(f"Analysis stored with ID: {analysis_id}")
        return str(analysis_id)
    except Exception as e:
        logger.error(f"Error storing analysis in MongoDB: {str(e)}")
        return f"Error storing analysis: {str(e)}"

def get_master_surgeries_db() -> List[Dict[str, Any]]:
    """
    Get all surgery details from the master surgeries collection
    
    Returns:
        List of all surgery details from master surgeries collection
    """
    try:
        # Always get all surgeries from the master collection
        surgeries = mongodb_client.master_collection.find()
        
        # Convert to list and format the results
        result = []
        for surgery in surgeries:
            # Extract original_surgery_types if available
            # original_types = []
            # if "original_surgery_types" in surgery and surgery["original_surgery_types"]:
            #     for orig in surgery["original_surgery_types"]:
            #         original_types.append({
            #             "name": orig.get("name", ""),
            #             "summary": orig.get("summary", "")
            #         })
            
            result.append({
                "id": str(surgery["_id"]),
                "surgery_type": surgery.get("surgery_type", ""),
                "summary": surgery.get("summary", ""),
                # "original_surgery_types": original_types
            })
        
        logger.info(f"Retrieved {len(result)} surgeries from master collection")
        return result
    except Exception as e:
        logger.error(f"Error retrieving master surgeries: {str(e)}")
        return []

def add_to_master_surgeries_db(
    surgery_type: str,
    procedure_steps: List[str],
    summary: str,
    master_id: Optional[str] = None
) -> str:
    """
    Add the analysis to the master surgeries collection
    
    Args:
        surgery_type: Type of surgery identified
        procedure_steps: List of timestamped procedure steps
        summary: Concise summary of the procedure
        master_id: Optional ID of an existing master surgery to update
        
    Returns:
        ID of the master surgery document
    """
    try:
        master_id = mongodb_client.add_to_master_surgeries(
            surgery_type=surgery_type,
            procedure_steps=procedure_steps,
            summary=summary,
            master_id=master_id
        )
        logger.info(f"Added to master surgeries with ID: {master_id}")
        return str(master_id)
    except Exception as e:
        logger.error(f"Error adding to master surgeries: {str(e)}")
        return f"Error adding to master surgeries: {str(e)}"





def get_master_surgeries_with_steps_db() -> List[Dict[str, Any]]:
    """
    Get all surgery details from the master surgeries collection
    
    Returns:
        List of all surgery details from master surgeries collection
    """
    try:
        # Always get all surgeries from the master collection
        surgeries = mongodb_client.master_collection.find()
        
        # Convert to list and format the results
        result = []
        for surgery in surgeries:
            # Get all procedure steps - if it's a nested array, flatten it
            procedure_steps = surgery.get("procedure_steps", [])
            
            result.append({
                "id": str(surgery["_id"]),
                "surgery_type": surgery.get("surgery_type", ""),
                "procedure_steps": procedure_steps,
                "summary": surgery.get("summary", "")
            })
        
        logger.info(f"Retrieved {len(result)} surgeries with procedure steps and summary from master collection")
        return result
    except Exception as e:
        logger.error(f"Error retrieving master surgeries: {str(e)}")
        return []
