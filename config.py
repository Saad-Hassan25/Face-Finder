"""
Configuration settings for Face Finder application
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    api_title: str = "Face Finder API"
    api_description: str = "Face detection, embedding, and search using SCRFD, LVFace, and Qdrant"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Face Detection Settings (SCRFD)
    scrfd_model_path: str = Field(
        default="models/scrfd.onnx",
        description="Path to SCRFD ONNX model"
    )
    detection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Face detection confidence threshold"
    )
    
    # Face Embedding Settings (LVFace)
    lvface_model_path: str = Field(
        default="models/lvface.onnx",
        description="Path to LVFace ONNX model"
    )
    use_gpu: bool = Field(
        default=True,
        description="Use GPU for inference"
    )
    embedding_dim: int = Field(
        default=512,
        description="Face embedding dimension"
    )
    
    # Qdrant Settings
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host"
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port"
    )
    qdrant_collection_name: str = Field(
        default="face_embeddings",
        description="Qdrant collection name for face embeddings"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (optional)"
    )
    
    # Search Settings
    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for face matching"
    )
    max_search_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of search results to return"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()
