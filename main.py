"""
FastAPI application for Face Finder
"""

import io
import os
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Generator
import logging

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from config import settings
from models import (
    FaceDetectionResponse,
    FaceEmbeddingResponse,
    FaceRegistrationResponse,
    FaceSearchResponse,
    HealthResponse,
    HealthStatus,
    ComponentHealth,
    StatsResponse,
    DeleteResponse,
    FaceRegistration,
    DetectedFace,
    SearchMatch,
    ImageIndexResponse,
    BulkIndexResponse,
    FindPersonResponse,
    IndexedFace,
    BoundingBox,
    GalleryStatsResponse,
    FolderIndexRequest,
    FolderIndexResponse,
    SaveGalleryRequest,
    SaveGalleryResponse,
    LoadGalleryResponse,
    ListSavedGalleriesResponse,
    SavedGalleryInfo
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded service instances
_face_detector = None
_face_embedder = None
_qdrant_service = None


def get_detector():
    """Get face detector instance (lazy initialization)."""
    global _face_detector
    if _face_detector is None:
        from services.face_detection import FaceDetectionService
        _face_detector = FaceDetectionService()
    return _face_detector


def get_embedder():
    """Get face embedder instance (lazy initialization)."""
    global _face_embedder
    if _face_embedder is None:
        from services.face_embedding import FaceEmbeddingService
        _face_embedder = FaceEmbeddingService()
    return _face_embedder


def get_qdrant():
    """Get Qdrant service instance (lazy initialization)."""
    global _qdrant_service
    if _qdrant_service is None:
        from services.qdrant_service import QdrantService
        _qdrant_service = QdrantService()
    return _qdrant_service


async def read_image_file(file: UploadFile) -> np.ndarray:
    """Read uploaded file and convert to numpy array."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def read_image_from_path(file_path: str) -> np.ndarray:
    """Read image from file path and convert to numpy array."""
    image = Image.open(file_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def get_image_files(folder_path: str, extensions: List[str], recursive: bool = False) -> List[Path]:
    """Get all image files from a folder."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")
    
    image_files = []
    pattern_func = folder.rglob if recursive else folder.glob
    
    for ext in extensions:
        image_files.extend(pattern_func(f"*.{ext}"))
        image_files.extend(pattern_func(f"*.{ext.upper()}"))
    
    return sorted(set(image_files))


# ================== Health & Status Endpoints ==================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check health status of all components.
    """
    components = {}
    overall_status = HealthStatus.HEALTHY
    
    # Check Face Detector
    try:
        detector = get_detector()
        start = time.time()
        is_healthy = detector.is_available()
        latency = (time.time() - start) * 1000
        components["face_detector"] = ComponentHealth(
            status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            message="SCRFD model loaded" if is_healthy else "SCRFD model unavailable",
            latency_ms=latency
        )
        if not is_healthy:
            overall_status = HealthStatus.DEGRADED
    except Exception as e:
        components["face_detector"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
        overall_status = HealthStatus.DEGRADED
    
    # Check Face Embedder
    try:
        embedder = get_embedder()
        start = time.time()
        is_healthy = embedder.is_available()
        latency = (time.time() - start) * 1000
        components["face_embedder"] = ComponentHealth(
            status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            message="LVFace model loaded" if is_healthy else "LVFace model unavailable",
            latency_ms=latency
        )
        if not is_healthy:
            overall_status = HealthStatus.DEGRADED
    except Exception as e:
        components["face_embedder"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
        overall_status = HealthStatus.DEGRADED
    
    # Check Qdrant
    try:
        qdrant = get_qdrant()
        is_healthy, latency = qdrant.is_available()
        components["qdrant"] = ComponentHealth(
            status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            message="Qdrant connected" if is_healthy else "Qdrant unavailable",
            latency_ms=latency
        )
        if not is_healthy:
            overall_status = HealthStatus.DEGRADED
    except Exception as e:
        components["qdrant"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
        overall_status = HealthStatus.DEGRADED
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        components=components
    )


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """
    Get database statistics.
    """
    try:
        qdrant = get_qdrant()
        stats = qdrant.get_collection_stats()
        return StatsResponse(
            success=True,
            collection_stats=stats
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return StatsResponse(
            success=False,
            message=str(e)
        )


# ================== Face Detection Endpoints ==================

@app.post("/detect", response_model=FaceDetectionResponse, tags=["Detection"])
async def detect_faces(
    file: UploadFile = File(..., description="Image file to detect faces in"),
    return_keypoints: bool = Query(True, description="Include facial keypoints in response")
):
    """
    Detect faces in an uploaded image.
    
    Returns bounding boxes, confidence scores, and optionally facial keypoints for each detected face.
    """
    start_time = time.time()
    
    try:
        # Read image
        image = await read_image_file(file)
        
        # Get detector and detect faces
        detector = get_detector()
        faces = detector.detect_faces(image, return_keypoints=return_keypoints)
        
        processing_time = (time.time() - start_time) * 1000
        
        return FaceDetectionResponse(
            success=True,
            message=f"Detected {len(faces)} face(s)",
            faces=faces,
            total_faces=len(faces),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return FaceDetectionResponse(
            success=False,
            message=str(e),
            processing_time_ms=(time.time() - start_time) * 1000
        )


# ================== Face Embedding Endpoints ==================

@app.post("/embed", response_model=FaceEmbeddingResponse, tags=["Embedding"])
async def extract_embeddings(
    file: UploadFile = File(..., description="Image file to extract face embeddings from"),
    use_flip_augmentation: bool = Query(False, description="Use horizontal flip augmentation for better accuracy")
):
    """
    Extract face embeddings from an uploaded image.
    
    Automatically detects faces, aligns them, and extracts embeddings using LVFace model.
    """
    start_time = time.time()
    
    try:
        # Read image
        image = await read_image_file(file)
        
        # Detect and align faces
        detector = get_detector()
        faces = detector.detect_faces(image, return_keypoints=True)
        
        if not faces:
            return FaceEmbeddingResponse(
                success=True,
                message="No faces detected in image",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get aligned face images
        aligned_faces = detector.get_aligned_faces(image, faces)
        
        # Extract embeddings
        embedder = get_embedder()
        
        embeddings = []
        if use_flip_augmentation:
            for face_img in aligned_faces:
                emb = embedder.get_embedding_with_flip(face_img)
                embeddings.append(emb.tolist())
        else:
            emb_list = embedder.get_embeddings_batch(aligned_faces)
            embeddings = [emb.tolist() for emb in emb_list]
        
        processing_time = (time.time() - start_time) * 1000
        
        return FaceEmbeddingResponse(
            success=True,
            message=f"Extracted embeddings for {len(embeddings)} face(s)",
            embeddings=embeddings,
            total_embeddings=len(embeddings),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Face embedding extraction failed: {e}")
        return FaceEmbeddingResponse(
            success=False,
            message=str(e),
            processing_time_ms=(time.time() - start_time) * 1000
        )


# ================== Face Registration Endpoints ==================

@app.post("/register", response_model=FaceRegistrationResponse, tags=["Registration"])
async def register_face(
    file: UploadFile = File(..., description="Image file containing the face to register"),
    person_id: str = Form(..., description="Unique identifier for the person"),
    person_name: Optional[str] = Form(None, description="Name of the person")
):
    """
    Register a face in the database.
    
    Detects faces in the image, extracts embeddings, and stores them in Qdrant.
    Multiple faces in the same image will all be associated with the same person_id.
    """
    try:
        # Read image
        image = await read_image_file(file)
        
        # Detect and align faces
        detector = get_detector()
        faces = detector.detect_faces(image, return_keypoints=True)
        
        if not faces:
            return FaceRegistrationResponse(
                success=False,
                message="No faces detected in image",
                person_id=person_id
            )
        
        # Get aligned face images
        aligned_faces = detector.get_aligned_faces(image, faces)
        
        # Extract embeddings with flip augmentation for better quality
        embedder = get_embedder()
        embeddings = []
        for face_img in aligned_faces:
            emb = embedder.get_embedding_with_flip(face_img)
            embeddings.append(emb)
        
        # Store in Qdrant
        qdrant = get_qdrant()
        face_ids = qdrant.add_faces_batch(
            embeddings=embeddings,
            person_ids=[person_id] * len(embeddings),
            person_names=[person_name] * len(embeddings) if person_name else None
        )
        
        return FaceRegistrationResponse(
            success=True,
            message=f"Registered {len(face_ids)} face(s) for person",
            person_id=person_id,
            face_id=face_ids[0] if face_ids else None,
            faces_registered=len(face_ids)
        )
        
    except Exception as e:
        logger.error(f"Face registration failed: {e}")
        return FaceRegistrationResponse(
            success=False,
            message=str(e),
            person_id=person_id
        )


# ================== Face Search Endpoints ==================

@app.post("/search", response_model=FaceSearchResponse, tags=["Search"])
async def search_faces(
    file: UploadFile = File(..., description="Image file to search for matching faces"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results per face"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """
    Search for matching faces in the database.
    
    Detects faces in the query image, extracts embeddings, and searches for similar faces in Qdrant.
    """
    start_time = time.time()
    
    try:
        # Read image
        image = await read_image_file(file)
        
        # Detect and align faces
        detector = get_detector()
        faces = detector.detect_faces(image, return_keypoints=True)
        
        if not faces:
            return FaceSearchResponse(
                success=True,
                message="No faces detected in query image",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get aligned face images
        aligned_faces = detector.get_aligned_faces(image, faces)
        
        # Extract embeddings
        embedder = get_embedder()
        embeddings = []
        for face_img in aligned_faces:
            emb = embedder.get_embedding_with_flip(face_img)
            embeddings.append(emb)
        
        # Search in Qdrant
        qdrant = get_qdrant()
        all_matches = qdrant.search_faces_batch(
            query_embeddings=embeddings,
            limit=limit,
            score_threshold=threshold
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return FaceSearchResponse(
            success=True,
            message=f"Searched {len(faces)} face(s)",
            query_faces=len(faces),
            matches=all_matches,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Face search failed: {e}")
        return FaceSearchResponse(
            success=False,
            message=str(e),
            processing_time_ms=(time.time() - start_time) * 1000
        )


@app.post("/verify", tags=["Verification"])
async def verify_faces(
    file1: UploadFile = File(..., description="First image file"),
    file2: UploadFile = File(..., description="Second image file"),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="Similarity threshold for verification")
):
    """
    Verify if two images contain the same person.
    
    Compares the first detected face from each image and returns similarity score.
    """
    start_time = time.time()
    
    try:
        # Read images
        image1 = await read_image_file(file1)
        image2 = await read_image_file(file2)
        
        # Detect faces
        detector = get_detector()
        faces1 = detector.detect_faces(image1, return_keypoints=True)
        faces2 = detector.detect_faces(image2, return_keypoints=True)
        
        if not faces1:
            return JSONResponse(content={
                "success": False,
                "message": "No face detected in first image",
                "processing_time_ms": (time.time() - start_time) * 1000
            })
        
        if not faces2:
            return JSONResponse(content={
                "success": False,
                "message": "No face detected in second image",
                "processing_time_ms": (time.time() - start_time) * 1000
            })
        
        # Get aligned faces (use first face from each image)
        aligned1 = detector.get_aligned_faces(image1, [faces1[0]])
        aligned2 = detector.get_aligned_faces(image2, [faces2[0]])
        
        # Extract embeddings
        embedder = get_embedder()
        emb1 = embedder.get_embedding_with_flip(aligned1[0])
        emb2 = embedder.get_embedding_with_flip(aligned2[0])
        
        # Calculate similarity
        similarity = embedder.calculate_similarity(emb1, emb2)
        is_same_person = similarity >= threshold
        
        processing_time = (time.time() - start_time) * 1000
        
        return JSONResponse(content={
            "success": True,
            "is_same_person": is_same_person,
            "similarity": float(similarity),
            "threshold": threshold,
            "faces_in_image1": len(faces1),
            "faces_in_image2": len(faces2),
            "processing_time_ms": processing_time
        })
        
    except Exception as e:
        logger.error(f"Face verification failed: {e}")
        return JSONResponse(content={
            "success": False,
            "message": str(e),
            "processing_time_ms": (time.time() - start_time) * 1000
        })


# ================== Image Gallery Endpoints ==================

@app.post("/gallery/index", response_model=ImageIndexResponse, tags=["Gallery"])
async def index_image(
    file: UploadFile = File(..., description="Image file to index in the gallery"),
    image_id: Optional[str] = Form(None, description="Custom image ID (auto-generated if not provided)"),
    image_name: Optional[str] = Form(None, description="Display name for the image")
):
    """
    Index a single image in the gallery.
    
    Detects all faces in the image and stores their embeddings for later searching.
    Use this to build a photo gallery that can later be searched for specific people.
    """
    start_time = time.time()
    
    try:
        # Generate image_id if not provided
        if not image_id:
            image_id = str(uuid.uuid4())
        
        # Use original filename if image_name not provided
        if not image_name and file.filename:
            image_name = file.filename
        
        # Read image
        image = await read_image_file(file)
        
        # Detect and align faces
        detector = get_detector()
        faces = detector.detect_faces(image, return_keypoints=True)
        
        if not faces:
            return ImageIndexResponse(
                success=True,
                message="No faces detected in image",
                image_id=image_id,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get aligned face images
        aligned_faces = detector.get_aligned_faces(image, faces)
        
        # Extract embeddings with flip augmentation
        embedder = get_embedder()
        embeddings = []
        bboxes = []
        for i, (face_img, face) in enumerate(zip(aligned_faces, faces)):
            emb = embedder.get_embedding_with_flip(face_img)
            embeddings.append(emb)
            bboxes.append({
                "x": face.bbox.x,
                "y": face.bbox.y,
                "width": face.bbox.width,
                "height": face.bbox.height
            })
        
        # Store in Qdrant
        qdrant = get_qdrant()
        face_ids = qdrant.index_faces_from_image_batch(
            embeddings=embeddings,
            image_id=image_id,
            image_name=image_name,
            bboxes=bboxes
        )
        
        # Build response with indexed face details
        indexed_faces = [
            IndexedFace(
                face_id=face_ids[i],
                image_id=image_id,
                image_name=image_name,
                face_index=i,
                bbox=BoundingBox(**bboxes[i])
            )
            for i in range(len(face_ids))
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageIndexResponse(
            success=True,
            message=f"Indexed {len(face_ids)} face(s) from image",
            image_id=image_id,
            faces_indexed=len(face_ids),
            indexed_faces=indexed_faces,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Image indexing failed: {e}")
        return ImageIndexResponse(
            success=False,
            message=str(e),
            image_id=image_id or "",
            processing_time_ms=(time.time() - start_time) * 1000
        )


@app.post("/gallery/index-bulk", response_model=BulkIndexResponse, tags=["Gallery"])
async def index_images_bulk(
    files: List[UploadFile] = File(..., description="Multiple image files to index")
):
    """
    Index multiple images in the gallery at once.
    
    Processes all uploaded images and indexes all detected faces.
    This is more efficient than calling /gallery/index multiple times.
    """
    start_time = time.time()
    
    images_processed = []
    failed_images = []
    total_faces = 0
    
    detector = get_detector()
    embedder = get_embedder()
    qdrant = get_qdrant()
    
    for file in files:
        try:
            image_id = str(uuid.uuid4())
            image_name = file.filename or image_id
            
            # Read image
            image = await read_image_file(file)
            
            # Detect and align faces
            faces = detector.detect_faces(image, return_keypoints=True)
            
            if not faces:
                images_processed.append({
                    "image_id": image_id,
                    "image_name": image_name,
                    "faces_indexed": 0,
                    "status": "no_faces"
                })
                continue
            
            # Get aligned face images
            aligned_faces = detector.get_aligned_faces(image, faces)
            
            # Extract embeddings
            embeddings = []
            bboxes = []
            for i, (face_img, face) in enumerate(zip(aligned_faces, faces)):
                emb = embedder.get_embedding_with_flip(face_img)
                embeddings.append(emb)
                bboxes.append({
                    "x": face.bbox.x,
                    "y": face.bbox.y,
                    "width": face.bbox.width,
                    "height": face.bbox.height
                })
            
            # Store in Qdrant
            face_ids = qdrant.index_faces_from_image_batch(
                embeddings=embeddings,
                image_id=image_id,
                image_name=image_name,
                bboxes=bboxes
            )
            
            total_faces += len(face_ids)
            images_processed.append({
                "image_id": image_id,
                "image_name": image_name,
                "faces_indexed": len(face_ids),
                "status": "success"
            })
            
        except Exception as e:
            failed_images.append({
                "image_name": file.filename or "unknown",
                "error": str(e)
            })
            logger.error(f"Failed to index image {file.filename}: {e}")
    
    processing_time = (time.time() - start_time) * 1000
    
    return BulkIndexResponse(
        success=len(failed_images) == 0,
        message=f"Indexed {total_faces} faces from {len(images_processed)} images",
        total_images=len(files),
        total_faces_indexed=total_faces,
        images_processed=images_processed,
        failed_images=failed_images,
        processing_time_ms=processing_time
    )


@app.post("/gallery/find-person", response_model=FindPersonResponse, tags=["Gallery"])
async def find_person_in_gallery(
    file: UploadFile = File(..., description="Reference image of the person to find"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of images to return"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """
    Find all images in the gallery containing a specific person.
    
    Upload a reference photo of a person, and this endpoint will return all
    images from the indexed gallery where that person appears.
    
    **Use Case**: 
    - Upload event/party photos to build a gallery
    - Then search with someone's photo to find all images they appear in
    """
    start_time = time.time()
    
    try:
        # Read reference image
        image = await read_image_file(file)
        
        # Detect faces in reference image
        detector = get_detector()
        faces = detector.detect_faces(image, return_keypoints=True)
        
        if not faces:
            return FindPersonResponse(
                success=False,
                message="No face detected in reference image",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get aligned faces and extract embeddings
        aligned_faces = detector.get_aligned_faces(image, faces)
        
        embedder = get_embedder()
        embeddings = []
        for face_img in aligned_faces:
            emb = embedder.get_embedding_with_flip(face_img)
            embeddings.append(emb)
        
        # Search in gallery using all detected faces (in case multiple angles)
        qdrant = get_qdrant()
        
        if len(embeddings) == 1:
            # Single face query
            matches = qdrant.find_person_in_images(
                query_embedding=embeddings[0],
                limit=limit,
                score_threshold=threshold
            )
        else:
            # Multiple faces - combine results
            matches = qdrant.find_person_in_images_multi(
                query_embeddings=embeddings,
                limit=limit,
                score_threshold=threshold
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return FindPersonResponse(
            success=True,
            message=f"Found {len(matches)} images containing the person",
            query_face_count=len(faces),
            total_images_found=len(matches),
            images=matches,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Find person failed: {e}")
        return FindPersonResponse(
            success=False,
            message=str(e),
            processing_time_ms=(time.time() - start_time) * 1000
        )


def generate_folder_index_progress(
    folder_path: str,
    extensions: List[str],
    recursive: bool
) -> Generator[str, None, None]:
    """Generator that yields SSE events for folder indexing progress."""
    start_time = time.time()
    
    try:
        # Validate folder path
        path = Path(folder_path)
        if not path.exists():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Folder does not exist: {folder_path}'})}\n\n"
            return
        
        if not path.is_dir():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Path is not a directory: {folder_path}'})}\n\n"
            return
        
        # Get image files
        image_files = get_image_files(folder_path, extensions, recursive)
        total_images = len(image_files)
        
        if not image_files:
            yield f"data: {json.dumps({'type': 'complete', 'total_images': 0, 'total_faces': 0, 'message': 'No image files found'})}\n\n"
            return
        
        # Send initial count
        yield f"data: {json.dumps({'type': 'start', 'total_images': total_images, 'message': f'Found {total_images} images'})}\n\n"
        
        detector = get_detector()
        embedder = get_embedder()
        qdrant = get_qdrant()
        
        images_processed = []
        failed_images = []
        total_faces = 0
        
        for idx, image_path in enumerate(image_files):
            try:
                # Read image from disk
                image = read_image_from_path(str(image_path))
                
                # Detect faces
                faces = detector.detect_faces(image, return_keypoints=True)
                
                if not faces:
                    images_processed.append({
                        "image_name": image_path.name,
                        "file_path": str(image_path),
                        "faces_indexed": 0,
                        "status": "no_faces"
                    })
                else:
                    # Get aligned faces
                    aligned_faces = detector.get_aligned_faces(image, faces)
                    
                    # Extract embeddings
                    embeddings = []
                    bboxes = []
                    for i, (face_img, face_info) in enumerate(zip(aligned_faces, faces)):
                        emb = embedder.get_embedding_with_flip(face_img)
                        embeddings.append(emb)
                        bboxes.append({
                            "x": face_info.bbox.x,
                            "y": face_info.bbox.y,
                            "width": face_info.bbox.width,
                            "height": face_info.bbox.height
                        })
                    
                    # Generate image ID
                    image_id = str(uuid.uuid4())
                    
                    # Index faces with file_path
                    qdrant.index_faces_from_image_batch(
                        embeddings=embeddings,
                        image_id=image_id,
                        image_name=image_path.name,
                        file_path=str(image_path),
                        bboxes=bboxes
                    )
                    
                    total_faces += len(faces)
                    images_processed.append({
                        "image_id": image_id,
                        "image_name": image_path.name,
                        "file_path": str(image_path),
                        "faces_indexed": len(faces),
                        "status": "success"
                    })
                
                # Send progress update
                progress_data = {
                    'type': 'progress',
                    'current': idx + 1,
                    'total': total_images,
                    'percent': round(((idx + 1) / total_images) * 100, 1),
                    'current_image': image_path.name,
                    'faces_found': len(faces) if faces else 0,
                    'total_faces_so_far': total_faces
                }
                yield f"data: {json.dumps(progress_data)}\n\n"
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                failed_images.append({
                    "image_name": image_path.name,
                    "file_path": str(image_path),
                    "error": str(e)
                })
                # Still send progress even on failure
                progress_data = {
                    'type': 'progress',
                    'current': idx + 1,
                    'total': total_images,
                    'percent': round(((idx + 1) / total_images) * 100, 1),
                    'current_image': image_path.name,
                    'error': str(e)
                }
                yield f"data: {json.dumps(progress_data)}\n\n"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Send completion event
        complete_data = {
            'type': 'complete',
            'success': len(failed_images) == 0,
            'message': f"Indexed {total_faces} faces from {len(images_processed)} images",
            'folder_path': folder_path,
            'total_images': total_images,
            'total_faces_indexed': total_faces,
            'images_processed': images_processed,
            'failed_images': failed_images,
            'processing_time_ms': processing_time
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        
    except Exception as e:
        logger.error(f"Folder indexing failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/gallery/index-folder-stream", tags=["Gallery"])
async def index_folder_stream(request: FolderIndexRequest):
    """
    Index all images in a local folder with real-time progress updates via SSE.
    
    Returns a stream of Server-Sent Events with progress updates.
    """
    return StreamingResponse(
        generate_folder_index_progress(
            request.folder_path,
            request.extensions,
            request.recursive
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/gallery/index-folder", response_model=FolderIndexResponse, tags=["Gallery"])
async def index_folder(request: FolderIndexRequest):
    """
    Index all images in a local folder.
    
    Scans the folder for image files, detects faces, and indexes them in the gallery.
    Supports jpg, jpeg, png, webp, bmp by default.
    
    **Use Case**: 
    - Point to a folder containing event photos
    - All faces will be indexed for later searching
    """
    start_time = time.time()
    
    try:
        # Validate folder path
        folder_path = Path(request.folder_path)
        if not folder_path.exists():
            return FolderIndexResponse(
                success=False,
                message=f"Folder does not exist: {request.folder_path}",
                folder_path=request.folder_path,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        if not folder_path.is_dir():
            return FolderIndexResponse(
                success=False,
                message=f"Path is not a directory: {request.folder_path}",
                folder_path=request.folder_path,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get image files
        image_files = get_image_files(
            request.folder_path,
            request.extensions,
            request.recursive
        )
        
        if not image_files:
            return FolderIndexResponse(
                success=True,
                message="No image files found in folder",
                folder_path=request.folder_path,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        logger.info(f"Found {len(image_files)} images in folder: {request.folder_path}")
        
        detector = get_detector()
        embedder = get_embedder()
        qdrant = get_qdrant()
        
        images_processed = []
        failed_images = []
        total_faces = 0
        
        for image_path in image_files:
            try:
                # Read image from disk
                image = read_image_from_path(str(image_path))
                
                # Detect faces
                faces = detector.detect_faces(image, return_keypoints=True)
                
                if not faces:
                    images_processed.append({
                        "image_name": image_path.name,
                        "file_path": str(image_path),
                        "faces_indexed": 0,
                        "status": "no_faces"
                    })
                    continue
                
                # Get aligned faces
                aligned_faces = detector.get_aligned_faces(image, faces)
                
                # Extract embeddings
                embeddings = []
                bboxes = []
                for i, (face_img, face_info) in enumerate(zip(aligned_faces, faces)):
                    emb = embedder.get_embedding_with_flip(face_img)
                    embeddings.append(emb)
                    bboxes.append({
                        "x": face_info.bbox.x,
                        "y": face_info.bbox.y,
                        "width": face_info.bbox.width,
                        "height": face_info.bbox.height
                    })
                
                # Generate image ID
                image_id = str(uuid.uuid4())
                
                # Index faces with file_path
                qdrant.index_faces_from_image_batch(
                    embeddings=embeddings,
                    image_id=image_id,
                    image_name=image_path.name,
                    file_path=str(image_path),
                    bboxes=bboxes
                )
                
                total_faces += len(faces)
                images_processed.append({
                    "image_id": image_id,
                    "image_name": image_path.name,
                    "file_path": str(image_path),
                    "faces_indexed": len(faces),
                    "status": "success"
                })
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                failed_images.append({
                    "image_name": image_path.name,
                    "file_path": str(image_path),
                    "error": str(e)
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        return FolderIndexResponse(
            success=len(failed_images) == 0,
            message=f"Indexed {total_faces} faces from {len(images_processed)} images",
            folder_path=request.folder_path,
            total_images=len(image_files),
            total_faces_indexed=total_faces,
            images_processed=images_processed,
            failed_images=failed_images,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Folder indexing failed: {e}")
        return FolderIndexResponse(
            success=False,
            message=str(e),
            folder_path=request.folder_path,
            processing_time_ms=(time.time() - start_time) * 1000
        )


@app.get("/images", tags=["Gallery"])
async def serve_image(path: str = Query(..., description="Absolute file path to the image")):
    """
    Serve an image file from the local filesystem.
    
    This endpoint is used to display images in search results.
    Only allows serving image files for security.
    """
    try:
        # Security: Validate file path
        file_path = Path(path)
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Validate extension for security
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        if file_path.suffix.lower() not in valid_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Determine media type
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif'
        }
        media_type = media_types.get(file_path.suffix.lower(), 'image/jpeg')
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gallery/stats", response_model=GalleryStatsResponse, tags=["Gallery"])
async def get_gallery_stats():
    """
    Get statistics about the indexed image gallery.
    """
    try:
        qdrant = get_qdrant()
        stats = qdrant.get_gallery_stats()
        collection_stats = qdrant.get_collection_stats()
        
        return GalleryStatsResponse(
            success=True,
            total_images=stats.get("unique_images", 0),
            total_faces=stats.get("total_faces", 0),
            unique_images=stats.get("unique_images", 0),
            message=f"Gallery contains {stats.get('unique_images', 0)} images with {stats.get('total_faces', 0)} indexed faces"
        )
    except Exception as e:
        logger.error(f"Failed to get gallery stats: {e}")
        return GalleryStatsResponse(
            success=False,
            message=str(e)
        )


@app.delete("/gallery/image/{image_id}", response_model=DeleteResponse, tags=["Gallery"])
async def delete_image_from_gallery(image_id: str):
    """
    Delete an image and all its face embeddings from the gallery.
    """
    try:
        qdrant = get_qdrant()
        deleted_count = qdrant.delete_by_image_id(image_id)
        
        return DeleteResponse(
            success=True,
            message=f"Deleted image and {deleted_count} face(s)",
            deleted_count=deleted_count
        )
    except Exception as e:
        logger.error(f"Failed to delete image: {e}")
        return DeleteResponse(
            success=False,
            message=str(e)
        )


# ================== Saved Gallery Endpoints ==================

@app.post("/gallery/save", response_model=SaveGalleryResponse, tags=["Saved Galleries"])
async def save_gallery(request: SaveGalleryRequest):
    """
    Save the current gallery with a name.
    This creates a snapshot of all indexed faces that can be loaded later.
    """
    try:
        qdrant = get_qdrant()
        result = qdrant.save_gallery(request.name)
        
        return SaveGalleryResponse(
            success=True,
            message=f"Gallery saved as '{request.name}'",
            name=request.name,
            faces_saved=result["faces_saved"],
            unique_images=result["unique_images"]
        )
    except Exception as e:
        logger.error(f"Failed to save gallery: {e}")
        return SaveGalleryResponse(
            success=False,
            message=str(e)
        )


@app.post("/gallery/load/{name}", response_model=LoadGalleryResponse, tags=["Saved Galleries"])
async def load_gallery(name: str, clear_current: bool = True):
    """
    Load a previously saved gallery.
    By default, this clears the current gallery and replaces it with the saved one.
    """
    try:
        qdrant = get_qdrant()
        result = qdrant.load_gallery(name, clear_current=clear_current)
        
        return LoadGalleryResponse(
            success=True,
            message=f"Gallery '{name}' loaded successfully",
            name=name,
            faces_loaded=result["faces_loaded"],
            previous_gallery_cleared=result["previous_gallery_cleared"]
        )
    except ValueError as e:
        return LoadGalleryResponse(
            success=False,
            message=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to load gallery: {e}")
        return LoadGalleryResponse(
            success=False,
            message=str(e)
        )


def generate_save_gallery_events(name: str):
    """Generate SSE events for save gallery progress."""
    try:
        qdrant = get_qdrant()
        for update in qdrant.save_gallery_streaming(name):
            yield f"data: {json.dumps(update)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'message': str(e), 'progress': 0})}\n\n"


def generate_load_gallery_events(name: str, clear_current: bool):
    """Generate SSE events for load gallery progress."""
    try:
        qdrant = get_qdrant()
        for update in qdrant.load_gallery_streaming(name, clear_current):
            yield f"data: {json.dumps(update)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'message': str(e), 'progress': 0})}\n\n"


@app.post("/gallery/save/stream", tags=["Saved Galleries"])
async def save_gallery_stream(request: SaveGalleryRequest):
    """
    Save the current gallery with progress updates via Server-Sent Events.
    Returns a stream of progress events.
    """
    return StreamingResponse(
        generate_save_gallery_events(request.name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/gallery/load/{name}/stream", tags=["Saved Galleries"])
async def load_gallery_stream(name: str, clear_current: bool = True):
    """
    Load a saved gallery with progress updates via Server-Sent Events.
    Returns a stream of progress events.
    """
    return StreamingResponse(
        generate_load_gallery_events(name, clear_current),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/gallery/saved", response_model=ListSavedGalleriesResponse, tags=["Saved Galleries"])
async def list_saved_galleries():
    """
    List all saved galleries.
    """
    try:
        qdrant = get_qdrant()
        galleries = qdrant.list_saved_galleries()
        
        return ListSavedGalleriesResponse(
            success=True,
            message=f"Found {len(galleries)} saved galleries",
            galleries=[SavedGalleryInfo(**g) for g in galleries]
        )
    except Exception as e:
        logger.error(f"Failed to list saved galleries: {e}")
        return ListSavedGalleriesResponse(
            success=False,
            message=str(e)
        )


@app.delete("/gallery/saved/{name}", response_model=DeleteResponse, tags=["Saved Galleries"])
async def delete_saved_gallery(name: str):
    """
    Delete a saved gallery.
    """
    try:
        qdrant = get_qdrant()
        success = qdrant.delete_saved_gallery(name)
        
        if success:
            return DeleteResponse(
                success=True,
                message=f"Deleted saved gallery: {name}",
                deleted_count=1
            )
        else:
            return DeleteResponse(
                success=False,
                message=f"Failed to delete saved gallery: {name}"
            )
    except Exception as e:
        logger.error(f"Failed to delete saved gallery: {e}")
        return DeleteResponse(
            success=False,
            message=str(e)
        )


@app.delete("/gallery/clear", response_model=DeleteResponse, tags=["Gallery"])
async def clear_gallery():
    """
    Clear the current gallery (remove all indexed images).
    This does not affect saved galleries.
    """
    try:
        qdrant = get_qdrant()
        deleted_count = qdrant.clear_gallery()
        
        return DeleteResponse(
            success=True,
            message=f"Gallery cleared: {deleted_count} face(s) removed",
            deleted_count=deleted_count
        )
    except Exception as e:
        logger.error(f"Failed to clear gallery: {e}")
        return DeleteResponse(
            success=False,
            message=str(e)
        )


# ================== Delete Endpoints ==================

@app.delete("/person/{person_id}", response_model=DeleteResponse, tags=["Management"])
async def delete_person(person_id: str):
    """
    Delete all face embeddings for a person.
    """
    try:
        qdrant = get_qdrant()
        deleted_count = qdrant.delete_by_person_id(person_id)
        
        return DeleteResponse(
            success=True,
            message=f"Deleted all faces for person: {person_id}",
            deleted_count=deleted_count
        )
        
    except Exception as e:
        logger.error(f"Delete person failed: {e}")
        return DeleteResponse(
            success=False,
            message=str(e)
        )


@app.delete("/face/{face_id}", response_model=DeleteResponse, tags=["Management"])
async def delete_face(face_id: str):
    """
    Delete a specific face embedding.
    """
    try:
        qdrant = get_qdrant()
        success = qdrant.delete_by_face_id(face_id)
        
        return DeleteResponse(
            success=success,
            message="Face deleted" if success else "Failed to delete face",
            deleted_count=1 if success else 0
        )
        
    except Exception as e:
        logger.error(f"Delete face failed: {e}")
        return DeleteResponse(
            success=False,
            message=str(e)
        )


@app.delete("/collection", response_model=DeleteResponse, tags=["Management"])
async def clear_collection():
    """
    Clear all face embeddings from the database.
    
    WARNING: This will delete ALL stored face data!
    """
    try:
        qdrant = get_qdrant()
        stats = qdrant.get_collection_stats()
        count = stats.total_faces
        
        success = qdrant.clear_collection()
        
        return DeleteResponse(
            success=success,
            message="Collection cleared" if success else "Failed to clear collection",
            deleted_count=count if success else 0
        )
        
    except Exception as e:
        logger.error(f"Clear collection failed: {e}")
        return DeleteResponse(
            success=False,
            message=str(e)
        )


# ================== Google Drive Endpoints ==================

from services.google_drive import get_drive_service

@app.get("/drive/status", tags=["Google Drive"])
async def drive_status():
    """Check if user is connected to Google Drive."""
    drive = get_drive_service()
    if drive.is_authenticated:
        user_info = drive.get_user_info()
        return {
            "connected": True,
            "user": user_info
        }
    return {"connected": False, "user": None}


@app.get("/drive/auth/url", tags=["Google Drive"])
async def get_drive_auth_url(redirect_uri: str = "http://localhost:5173/drive/callback"):
    """Get the Google OAuth authorization URL."""
    drive = get_drive_service()
    auth_url = drive.get_auth_url(redirect_uri)
    if auth_url:
        return {"auth_url": auth_url}
    raise HTTPException(status_code=500, detail="credentials.json not found. Please set up Google OAuth.")


@app.get("/drive/auth/callback", tags=["Google Drive"])
async def drive_auth_callback(code: str, redirect_uri: str = "http://localhost:5173/drive/callback"):
    """Handle the OAuth callback from Google."""
    drive = get_drive_service()
    success = drive.handle_callback(code, redirect_uri)
    if success:
        return {"success": True, "message": "Successfully connected to Google Drive"}
    raise HTTPException(status_code=400, detail="Failed to authenticate with Google Drive")


@app.post("/drive/logout", tags=["Google Drive"])
async def drive_logout():
    """Disconnect from Google Drive."""
    drive = get_drive_service()
    success = drive.logout()
    return {"success": success}


@app.get("/drive/folders", tags=["Google Drive"])
async def list_drive_folders(parent_id: str = "root"):
    """List folders in Google Drive."""
    drive = get_drive_service()
    if not drive.is_authenticated:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive")
    
    folders = drive.list_folders(parent_id)
    return {"folders": folders, "parent_id": parent_id}


@app.get("/drive/contents/{folder_id}", tags=["Google Drive"])
async def list_drive_contents(folder_id: str = "root"):
    """List all contents (files and folders) in a Google Drive folder."""
    drive = get_drive_service()
    if not drive.is_authenticated:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive")
    
    contents = drive.list_all_contents(folder_id)
    
    # Separate folders and files
    folders = [c for c in contents if c['mimeType'] == 'application/vnd.google-apps.folder']
    files = [c for c in contents if c['mimeType'] != 'application/vnd.google-apps.folder']
    
    return {
        "folder_id": folder_id,
        "folders": folders,
        "files": files,
        "total_items": len(contents)
    }


@app.get("/drive/images/{folder_id}", tags=["Google Drive"])
async def list_drive_images(folder_id: str, recursive: bool = False):
    """List only image files in a Google Drive folder."""
    drive = get_drive_service()
    if not drive.is_authenticated:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive")
    
    images = drive.list_files(folder_id, images_only=True)
    total_count = len(images)
    
    if recursive:
        total_count = drive.get_folder_image_count(folder_id, recursive=True)
    
    return {
        "folder_id": folder_id,
        "images": images,
        "count": len(images),
        "total_recursive": total_count if recursive else None
    }


def generate_drive_index_events(folder_id: str, recursive: bool):
    """Generate SSE events for indexing a Google Drive folder."""
    import time
    start_time = time.time()
    
    drive = get_drive_service()
    detector = get_detector()
    embedder = get_embedder()
    qdrant = get_qdrant()
    
    if not drive.is_authenticated:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Not connected to Google Drive'})}\n\n"
        return
    
    try:
        # Count total images first
        yield f"data: {json.dumps({'type': 'start', 'message': 'Counting images...'})}\n\n"
        
        total_images = drive.get_folder_image_count(folder_id, recursive=recursive)
        
        if total_images == 0:
            yield f"data: {json.dumps({'type': 'complete', 'message': 'No images found in folder', 'total_images': 0, 'total_faces': 0, 'processing_time_ms': 0})}\n\n"
            return
        
        yield f"data: {json.dumps({'type': 'progress', 'message': f'Found {total_images} images', 'current': 0, 'total': total_images})}\n\n"
        
        processed = 0
        total_faces = 0
        failed = 0
        
        # Process each image
        for image_data in drive.iterate_images(folder_id, recursive=recursive):
            try:
                # Load image from bytes
                image = Image.open(io.BytesIO(image_data['content']))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_np = np.array(image)
                
                # Detect faces
                faces = detector.detect_faces(image_np, return_keypoints=True)
                
                if faces:
                    # Get aligned faces
                    aligned_faces = detector.get_aligned_faces(image_np, faces)
                    
                    # Extract embeddings and bboxes
                    embeddings = []
                    bboxes = []
                    for i, (face_img, face_info) in enumerate(zip(aligned_faces, faces)):
                        emb = embedder.get_embedding_with_flip(face_img)
                        embeddings.append(emb)
                        bboxes.append({
                            "x": face_info.bbox.x,
                            "y": face_info.bbox.y,
                            "width": face_info.bbox.width,
                            "height": face_info.bbox.height
                        })
                    
                    # Generate image ID and store in Qdrant
                    image_id = str(uuid.uuid4())
                    qdrant.index_faces_from_image_batch(
                        embeddings=embeddings,
                        image_id=image_id,
                        image_name=image_data['name'],
                        file_path=f"gdrive://{image_data['id']}",  # Special prefix for Drive files
                        bboxes=bboxes
                    )
                    total_faces += len(faces)
                
                processed += 1
                
                # Send progress update
                progress_data = {
                    'type': 'progress',
                    'current': processed,
                    'total': total_images,
                    'percent': int((processed / total_images) * 100),
                    'current_image': image_data['name'],
                    'faces_found': len(faces) if faces else 0,
                    'total_faces_so_far': total_faces
                }
                yield f"data: {json.dumps(progress_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Failed to process {image_data['name']}: {e}")
                failed += 1
                processed += 1
            
            # Clear memory
            del image_data['content']
        
        # Complete
        processing_time_ms = int((time.time() - start_time) * 1000)
        complete_data = {
            'type': 'complete',
            'message': f'Successfully indexed {processed} images with {total_faces} faces',
            'total_images': processed,
            'total_faces': total_faces,
            'failed': failed,
            'processing_time_ms': processing_time_ms
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        
    except Exception as e:
        logger.error(f"Drive indexing failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/drive/index/{folder_id}", tags=["Google Drive"])
async def index_drive_folder(folder_id: str, recursive: bool = True):
    """
    Index all images from a Google Drive folder.
    Returns Server-Sent Events for progress tracking.
    """
    return StreamingResponse(
        generate_drive_index_events(folder_id, recursive),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/drive/thumbnail/{file_id}", tags=["Google Drive"])
async def get_drive_thumbnail(file_id: str):
    """Get a thumbnail for a Google Drive image."""
    drive = get_drive_service()
    if not drive.is_authenticated:
        raise HTTPException(status_code=401, detail="Not connected to Google Drive")
    
    content = drive.download_file(file_id)
    if content:
        return StreamingResponse(
            io.BytesIO(content),
            media_type="image/jpeg"
        )
    raise HTTPException(status_code=404, detail="Image not found")


# ================== Application Entry Point ==================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
