from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from typing import Dict, List, Any, Optional
from app.config import get_settings
import re

settings = get_settings()

class MongoDBClient:
    """MongoDB client for surgery video analysis data storage and retrieval"""
    
    def __init__(self):
        """Initialize MongoDB connection with proper server API"""
        self.client = MongoClient(settings.MONGODB_URI, server_api=ServerApi('1'))
        self.db = self.client[settings.MONGODB_DB_NAME]
        self.analysis_collection = self.db['analysis']
        self.master_collection = self.db['master_surgeries']
        
        # Verify connection
        try:
            self.client.admin.command('ping')
            print("✅ MongoDB connection successful")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
    
    def store_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """
        Store surgery video analysis in the analysis collection
        
        Args:
            analysis_data: Dictionary containing analysis results
                - video_id: Name/ID of the video file
                - surgery_type: Type of surgery identified
                - procedure_steps: List of timestamped procedure steps
                - description: Detailed description of the procedure
                - summary: Concise summary of the procedure
                
        Returns:
            ID of the inserted document
        """
        # Add timestamps
        now = datetime.now().isoformat()
        analysis_data["created_at"] = now
        
        # Insert into analysis collection
        result = self.analysis_collection.insert_one(analysis_data)
        return result.inserted_id
    
    def add_to_master_surgeries(self, 
                               surgery_type: str, 
                               procedure_steps: List[str], 
                               summary: str,
                               master_id: Optional[str] = None) -> str:
        """
        Add or update master surgery information.

        If a match is found, it updates the top-level summary with the new synthesized
        summary from the agent, appends procedure steps, and adds the original analysis
        summary to a historical list.
        
        Args:
            surgery_type: Type of surgery
            procedure_steps: List of procedure steps
            summary: The new or synthesized summary of the procedure
            master_id: Optional ID of an existing master surgery to update
            
        Returns:
            ID of the inserted/updated document
        """
        # If master_id is provided, try to find that specific document
        if master_id:
            try:
                from bson.objectid import ObjectId
                existing = self.master_collection.find_one({"_id": ObjectId(master_id)})
                if not existing:
                    # If ID not found, fall back to surgery_type search
                    existing = self.master_collection.find_one({"surgery_type": surgery_type})
            except Exception:
                # If ID is invalid, fall back to surgery_type search
                existing = self.master_collection.find_one({"surgery_type": surgery_type})
        else:
            # Check if this surgery type already exists in master collection
            existing = self.master_collection.find_one({"surgery_type": surgery_type})
        
        if existing:
            # Update existing master surgery
            existing_steps = existing.get("procedure_steps", [])
            all_steps = existing_steps + procedure_steps
            
            # original_types = existing.get("original_surgery_types", [])
            # # Create a record of the original analysis that led to this update
            # original_types.append({
            #     "name": surgery_type,
            #     "summary": summary # This is the summary from the new analysis
            # })
            
            # Update document, replacing the top-level summary with the new one
            self.master_collection.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "summary": summary,  # Overwrite the top-level summary
                        "procedure_steps": all_steps,
                        # "original_surgery_types": original_types,
                        "last_updated": datetime.now().isoformat()
                    }
                }
            )
            
            return str(existing["_id"])
        else:
            # Create new master surgery entry
            now = datetime.now().isoformat()
            master_data = {
                "surgery_type": surgery_type,
                "summary": summary,  # Add top-level summary on creation
                "procedure_steps": procedure_steps,
                # "original_surgery_types": [
                #     {
                #         "name": surgery_type,
                #         "summary": summary
                #     }
                # ],
                "created_at": now,
                "last_updated": now
            }
            
            result = self.master_collection.insert_one(master_data)
            return str(result.inserted_id)
    
    def get_master_surgery_data(self, surgery_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve master surgery data for a given surgery type
        
        Args:
            surgery_type: Type of surgery to find
            
        Returns:
            Master surgery document or None if not found
        """
        return self.master_collection.find_one({"surgery_type": surgery_type})

# Singleton instance
mongodb_client = MongoDBClient()
