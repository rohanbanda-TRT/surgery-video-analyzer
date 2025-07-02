import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""
    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "surgery_video_analysis")
    
    # Google Cloud settings
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    
    # Video analysis settings
    MAX_CHUNK_DURATION_MINUTES: int = 10  # Maximum duration of video chunks in minutes
    
    # Additional settings that might be in environment variables
    langsmith_tracing: str = None
    langsmith_endpoint: str = None
    langsmith_api_key: str = None
    langsmith_project: str = None
    google_application_credentials: str = None
    mongodb_password: str = None
    
    class Config:
        env_file = ".env"
        extra = "ignore"  

@lru_cache()
def get_settings():
    """Cached settings to avoid reloading"""
    return Settings()
