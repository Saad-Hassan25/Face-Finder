"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class BoundingBox(BaseModel):
    """Face bounding box coordinates"""
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")


class Keypoint(BaseModel):
    """Facial keypoint coordinates"""
    x: float
    y: float


class FaceKeypoints(BaseModel):
    """Five facial keypoints"""
    left_eye: Keypoint
    right_eye: Keypoint
    nose: Keypoint
    left_mouth: Keypoint
    right_mouth: Keypoint


class DetectedFace(BaseModel):
    """Detected face information"""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    keypoints: Optional[FaceKeypoints] = None
    face_index: int = Field(..., description="Index of face in image")


class FaceDetectionResponse(BaseModel):
    """Response for face detection endpoint"""
    success: bool
    message: str
    faces: List[DetectedFace] = []
    total_faces: int = 0
    processing_time_ms: float = 0


class FaceEmbeddingResponse(BaseModel):
    """Response for face embedding endpoint"""
    success: bool
    message: str
    embeddings: List[List[float]] = []
    total_embeddings: int = 0
    processing_time_ms: float = 0


class FaceRegistration(BaseModel):
    """Face registration request"""
    person_id: str = Field(..., min_length=1, max_length=100, description="Unique person identifier")
    person_name: Optional[str] = Field(None, max_length=200, description="Person's name")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class FaceRegistrationResponse(BaseModel):
    """Response for face registration endpoint"""
    success: bool
    message: str
    person_id: str
    face_id: Optional[str] = None
    faces_registered: int = 0


class SearchMatch(BaseModel):
    """A matching face from search"""
    person_id: str
    person_name: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    face_id: str
    metadata: Dict[str, Any] = {}


class FaceSearchResponse(BaseModel):
    """Response for face search endpoint"""
    success: bool
    message: str
    query_faces: int = 0
    matches: List[List[SearchMatch]] = []  # List of matches per query face
    processing_time_ms: float = 0


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ComponentHealth(BaseModel):
    """Health status of a component"""
    status: HealthStatus
    message: str = ""
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Response for health check endpoint"""
    status: HealthStatus
    timestamp: datetime
    components: Dict[str, ComponentHealth]


class CollectionStats(BaseModel):
    """Qdrant collection statistics"""
    total_faces: int
    vectors_count: int
    indexed_vectors_count: int


class StatsResponse(BaseModel):
    """Response for stats endpoint"""
    success: bool
    collection_stats: Optional[CollectionStats] = None
    message: str = ""


class DeleteResponse(BaseModel):
    """Response for delete operations"""
    success: bool
    message: str
    deleted_count: int = 0


# ================== Image Gallery Models ==================

class ImageIndexRequest(BaseModel):
    """Request for indexing an image in the gallery"""
    image_id: Optional[str] = Field(None, description="Custom image ID (auto-generated if not provided)")
    image_name: Optional[str] = Field(None, description="Original filename or display name")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata (album, date, location, etc.)")


class IndexedFace(BaseModel):
    """Information about an indexed face"""
    face_id: str
    image_id: str
    image_name: Optional[str] = None
    face_index: int = Field(..., description="Index of face within the image")
    bbox: BoundingBox


class ImageIndexResponse(BaseModel):
    """Response for image indexing endpoint"""
    success: bool
    message: str
    image_id: str
    faces_indexed: int = 0
    indexed_faces: List[IndexedFace] = []
    processing_time_ms: float = 0


class BulkIndexResponse(BaseModel):
    """Response for bulk image indexing"""
    success: bool
    message: str
    total_images: int = 0
    total_faces_indexed: int = 0
    images_processed: List[Dict[str, Any]] = []
    failed_images: List[Dict[str, Any]] = []
    processing_time_ms: float = 0


class ImageMatch(BaseModel):
    """A matching image from person search"""
    image_id: str
    image_name: Optional[str] = None
    file_path: Optional[str] = Field(None, description="Absolute file path for local images")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Highest similarity score for this image")
    face_id: str
    face_bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = {}


class FindPersonResponse(BaseModel):
    """Response for find person in images endpoint"""
    success: bool
    message: str
    query_face_count: int = 0
    total_images_found: int = 0
    images: List[ImageMatch] = []
    processing_time_ms: float = 0


class GalleryStatsResponse(BaseModel):
    """Response for gallery statistics"""
    success: bool
    total_images: int = 0
    total_faces: int = 0
    unique_images: int = 0
    message: str = ""


class FolderIndexRequest(BaseModel):
    """Request for indexing a folder of images"""
    folder_path: str = Field(..., description="Absolute path to the folder containing images")
    recursive: bool = Field(False, description="Whether to search subdirectories")
    extensions: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp", "bmp"],
        description="Image file extensions to include"
    )


class FolderIndexResponse(BaseModel):
    """Response for folder indexing"""
    success: bool
    message: str
    folder_path: str
    total_images: int = 0
    total_faces_indexed: int = 0
    images_processed: List[Dict[str, Any]] = []
    failed_images: List[Dict[str, Any]] = []
    processing_time_ms: float = 0


# ================== Saved Gallery Models ==================

class SavedGalleryInfo(BaseModel):
    """Information about a saved gallery"""
    name: str = Field(..., description="Name of the saved gallery")
    created_at: str = Field(..., description="When the gallery was saved")
    total_faces: int = Field(0, description="Number of face embeddings")
    unique_images: int = Field(0, description="Number of unique images")


class SaveGalleryRequest(BaseModel):
    """Request to save current gallery"""
    name: str = Field(..., min_length=1, max_length=100, description="Name for the saved gallery")


class SaveGalleryResponse(BaseModel):
    """Response for save gallery operation"""
    success: bool
    message: str
    name: str = ""
    faces_saved: int = 0
    unique_images: int = 0


class LoadGalleryResponse(BaseModel):
    """Response for load gallery operation"""
    success: bool
    message: str
    name: str = ""
    faces_loaded: int = 0
    previous_gallery_cleared: bool = False


class ListSavedGalleriesResponse(BaseModel):
    """Response for listing saved galleries"""
    success: bool
    message: str
    galleries: List[SavedGalleryInfo] = []
