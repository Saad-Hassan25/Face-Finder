"""
Vector Database Service using Qdrant
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Generator
from datetime import datetime
import uuid
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import settings
from models import SearchMatch, CollectionStats, ImageMatch, BoundingBox

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Vector database service using Qdrant for storing and searching face embeddings.
    Qdrant provides efficient similarity search with filtering capabilities.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize Qdrant service.
        
        Args:
            host: Qdrant server host (for local)
            port: Qdrant server port (for local)
            url: Qdrant Cloud URL (takes precedence over host/port)
            collection_name: Name of the collection for face embeddings
            api_key: Optional API key for Qdrant Cloud
            embedding_dim: Dimension of face embeddings
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.api_key = api_key or settings.qdrant_api_key
        self.embedding_dim = embedding_dim or settings.embedding_dim
        
        # Initialize Qdrant client
        try:
            if self.url:
                # Qdrant Cloud (URL-based connection)
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key
                )
                logger.info(f"Connected to Qdrant Cloud at {self.url}")
            elif self.api_key:
                # Qdrant Cloud (host-based with API key)
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    https=True
                )
                logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            else:
                # Local Qdrant
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port
                )
                logger.info(f"Connected to local Qdrant at {self.host}:{self.port}")
            
            # Ensure collection exists
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Failed to initialize Qdrant service: {e}")
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist and ensure payload indexes."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
            
            # Ensure payload indexes exist (required for Qdrant Cloud filtering)
            self._ensure_payload_indexes()
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering (required for Qdrant Cloud)."""
        try:
            # Get existing indexes
            collection_info = self.client.get_collection(self.collection_name)
            existing_indexes = set()
            if collection_info.payload_schema:
                existing_indexes = set(collection_info.payload_schema.keys())
            
            # Define required indexes
            required_indexes = {
                "type": qdrant_models.PayloadSchemaType.KEYWORD,
                "person_id": qdrant_models.PayloadSchemaType.KEYWORD,
                "image_id": qdrant_models.PayloadSchemaType.KEYWORD,
            }
            
            # Create missing indexes
            for field_name, field_type in required_indexes.items():
                if field_name not in existing_indexes:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                        wait=True
                    )
                    logger.info(f"Created payload index for field: {field_name}")
                    
        except Exception as e:
            logger.warning(f"Failed to create payload indexes: {e}")
    
    def add_face(
        self,
        embedding: np.ndarray,
        person_id: str,
        person_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a face embedding to the database.
        
        Args:
            embedding: Face embedding vector
            person_id: Unique identifier for the person
            person_name: Optional name of the person
            metadata: Optional additional metadata
            
        Returns:
            Generated face_id for the stored embedding
        """
        face_id = str(uuid.uuid4())
        
        payload = {
            "person_id": person_id,
            "person_name": person_name or "",
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        point = qdrant_models.PointStruct(
            id=face_id,
            vector=embedding.tolist(),
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True
        )
        
        logger.info(f"Added face embedding for person_id: {person_id}, face_id: {face_id}")
        return face_id
    
    def add_faces_batch(
        self,
        embeddings: List[np.ndarray],
        person_ids: List[str],
        person_names: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple face embeddings to the database.
        
        Args:
            embeddings: List of face embedding vectors
            person_ids: List of person identifiers
            person_names: Optional list of person names
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of generated face_ids
        """
        if not embeddings:
            return []
        
        face_ids = []
        points = []
        
        for i, embedding in enumerate(embeddings):
            face_id = str(uuid.uuid4())
            face_ids.append(face_id)
            
            payload = {
                "person_id": person_ids[i],
                "person_name": person_names[i] if person_names else "",
                "created_at": datetime.utcnow().isoformat()
            }
            
            if metadata_list and i < len(metadata_list):
                payload.update(metadata_list[i])
            
            point = qdrant_models.PointStruct(
                id=face_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        logger.info(f"Added {len(face_ids)} face embeddings in batch")
        return face_ids
    
    def search_faces(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_person_id: Optional[str] = None
    ) -> List[SearchMatch]:
        """
        Search for similar faces in the database.
        
        Args:
            query_embedding: Query face embedding
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            filter_person_id: Optional filter by person_id
            
        Returns:
            List of SearchMatch objects
        """
        threshold = score_threshold or settings.similarity_threshold
        
        # Build filter if needed
        query_filter = None
        if filter_person_id:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="person_id",
                        match=qdrant_models.MatchValue(value=filter_person_id)
                    )
                ]
            )
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=limit,
            score_threshold=threshold
        ).points
        
        matches = []
        for result in results:
            payload = result.payload or {}
            match = SearchMatch(
                person_id=payload.get("person_id", "unknown"),
                person_name=payload.get("person_name"),
                similarity=result.score,
                face_id=str(result.id),
                metadata={k: v for k, v in payload.items() 
                         if k not in ["person_id", "person_name"]}
            )
            matches.append(match)
        
        return matches
    
    def search_faces_batch(
        self,
        query_embeddings: List[np.ndarray],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[List[SearchMatch]]:
        """
        Search for similar faces for multiple query embeddings.
        
        Args:
            query_embeddings: List of query face embeddings
            limit: Maximum number of results per query
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of SearchMatch lists (one per query)
        """
        results = []
        for embedding in query_embeddings:
            matches = self.search_faces(
                embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            results.append(matches)
        return results
    
    def delete_by_person_id(self, person_id: str) -> int:
        """
        Delete all face embeddings for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Number of deleted entries
        """
        # First, count entries to delete
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="person_id",
                        match=qdrant_models.MatchValue(value=person_id)
                    )
                ]
            )
        )
        
        # Delete entries
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="person_id",
                            match=qdrant_models.MatchValue(value=person_id)
                        )
                    ]
                )
            ),
            wait=True
        )
        
        deleted_count = count_result.count
        logger.info(f"Deleted {deleted_count} embeddings for person_id: {person_id}")
        return deleted_count
    
    def delete_by_face_id(self, face_id: str) -> bool:
        """
        Delete a specific face embedding.
        
        Args:
            face_id: Face identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[face_id]
                ),
                wait=True
            )
            logger.info(f"Deleted face_id: {face_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete face_id {face_id}: {e}")
            return False
    
    def get_collection_stats(self) -> CollectionStats:
        """
        Get collection statistics.
        
        Returns:
            CollectionStats object
        """
        try:
            info = self.client.get_collection(self.collection_name)
            points_count = info.points_count or 0
            
            # Handle different API versions for vectors_count
            vectors_count = 0
            indexed_vectors_count = 0
            
            if hasattr(info, 'vectors_count') and info.vectors_count is not None:
                vectors_count = info.vectors_count
            else:
                vectors_count = points_count
                
            if hasattr(info, 'indexed_vectors_count') and info.indexed_vectors_count is not None:
                indexed_vectors_count = info.indexed_vectors_count
            else:
                indexed_vectors_count = points_count
            
            return CollectionStats(
                total_faces=points_count,
                vectors_count=vectors_count,
                indexed_vectors_count=indexed_vectors_count
            )
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return CollectionStats(
                total_faces=0,
                vectors_count=0,
                indexed_vectors_count=0
            )
    
    def get_person_faces(self, person_id: str) -> List[str]:
        """
        Get all face IDs for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            List of face IDs
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="person_id",
                        match=qdrant_models.MatchValue(value=person_id)
                    )
                ]
            ),
            limit=1000,  # Adjust as needed
            with_payload=False,
            with_vectors=False
        )
        
        return [str(point.id) for point in results[0]]
    
    def clear_collection(self) -> bool:
        """
        Delete and recreate the collection (clear all data).
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def is_available(self) -> Tuple[bool, float]:
        """
        Check if Qdrant is available and measure latency.
        
        Returns:
            Tuple of (is_available, latency_ms)
        """
        import time
        start = time.time()
        try:
            self.client.get_collections()
            latency = (time.time() - start) * 1000
            return True, latency
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False, 0

    # ================== Image Gallery Methods ==================
    
    def index_face_from_image(
        self,
        embedding: np.ndarray,
        image_id: str,
        image_name: Optional[str] = None,
        file_path: Optional[str] = None,
        face_index: int = 0,
        bbox: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Index a face from an image in the gallery.
        
        Args:
            embedding: Face embedding vector
            image_id: Unique identifier for the source image
            image_name: Original filename or display name
            file_path: Absolute path to the image file (for local images)
            face_index: Index of this face within the image
            bbox: Bounding box of the face
            metadata: Optional additional metadata
            
        Returns:
            Generated face_id for the stored embedding
        """
        face_id = str(uuid.uuid4())
        
        payload = {
            "image_id": image_id,
            "image_name": image_name or "",
            "file_path": file_path or "",
            "face_index": face_index,
            "bbox": bbox,
            "indexed_at": datetime.utcnow().isoformat(),
            "type": "gallery",  # Mark as gallery image vs registered person
            **(metadata or {})
        }
        
        point = qdrant_models.PointStruct(
            id=face_id,
            vector=embedding.tolist(),
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True
        )
        
        logger.info(f"Indexed face from image_id: {image_id}, face_id: {face_id}")
        return face_id
    
    def index_faces_from_image_batch(
        self,
        embeddings: List[np.ndarray],
        image_id: str,
        image_name: Optional[str] = None,
        file_path: Optional[str] = None,
        bboxes: Optional[List[Dict[str, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Index multiple faces from a single image.
        
        Args:
            embeddings: List of face embedding vectors
            image_id: Unique identifier for the source image
            image_name: Original filename or display name
            file_path: Absolute path to the image file (for local images)
            bboxes: List of bounding boxes for each face
            metadata: Optional metadata to apply to all faces
            
        Returns:
            List of generated face_ids
        """
        if not embeddings:
            return []
        
        face_ids = []
        points = []
        
        for i, embedding in enumerate(embeddings):
            face_id = str(uuid.uuid4())
            face_ids.append(face_id)
            
            payload = {
                "image_id": image_id,
                "image_name": image_name or "",
                "file_path": file_path or "",
                "face_index": i,
                "bbox": bboxes[i] if bboxes and i < len(bboxes) else None,
                "indexed_at": datetime.utcnow().isoformat(),
                "type": "gallery",
                **(metadata or {})
            }
            
            point = qdrant_models.PointStruct(
                id=face_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        logger.info(f"Indexed {len(face_ids)} faces from image_id: {image_id}")
        return face_ids
    
    def find_person_in_images(
        self,
        query_embedding: np.ndarray,
        limit: int = 100,
        score_threshold: Optional[float] = None,
        deduplicate_images: bool = True
    ) -> List[ImageMatch]:
        """
        Find all images containing a person matching the query embedding.
        
        Args:
            query_embedding: Query face embedding
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            deduplicate_images: If True, return only the best match per image
            
        Returns:
            List of ImageMatch objects
        """
        threshold = score_threshold or settings.similarity_threshold
        
        # Search for gallery images only
        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="gallery")
                )
            ]
        )
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=limit * 3 if deduplicate_images else limit,  # Get more for dedup
            score_threshold=threshold
        ).points
        
        matches = []
        seen_images = set()
        
        for result in results:
            payload = result.payload or {}
            image_id = payload.get("image_id", "")
            
            # Deduplicate by image_id (keep highest scoring match per image)
            if deduplicate_images and image_id in seen_images:
                continue
            seen_images.add(image_id)
            
            # Parse bbox if present
            bbox_data = payload.get("bbox")
            face_bbox = None
            if bbox_data and isinstance(bbox_data, dict):
                face_bbox = BoundingBox(
                    x=bbox_data.get("x", 0),
                    y=bbox_data.get("y", 0),
                    width=bbox_data.get("width", 0),
                    height=bbox_data.get("height", 0)
                )
            
            match = ImageMatch(
                image_id=image_id,
                image_name=payload.get("image_name"),
                file_path=payload.get("file_path"),
                similarity=result.score,
                face_id=str(result.id),
                face_bbox=face_bbox,
                metadata={k: v for k, v in payload.items() 
                         if k not in ["image_id", "image_name", "file_path", "bbox", "type", "face_index", "indexed_at"]}
            )
            matches.append(match)
            
            if len(matches) >= limit:
                break
        
        return matches
    
    def find_person_in_images_multi(
        self,
        query_embeddings: List[np.ndarray],
        limit: int = 100,
        score_threshold: Optional[float] = None
    ) -> List[ImageMatch]:
        """
        Find all images containing a person using multiple query embeddings.
        Combines results from all query embeddings and deduplicates by image.
        
        Args:
            query_embeddings: List of query face embeddings (e.g., from different photos of same person)
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of ImageMatch objects (deduplicated by image_id)
        """
        all_matches = {}  # image_id -> best ImageMatch
        
        for embedding in query_embeddings:
            matches = self.find_person_in_images(
                embedding,
                limit=limit,
                score_threshold=score_threshold,
                deduplicate_images=True
            )
            
            for match in matches:
                existing = all_matches.get(match.image_id)
                if existing is None or match.similarity > existing.similarity:
                    all_matches[match.image_id] = match
        
        # Sort by similarity (highest first) and limit
        sorted_matches = sorted(all_matches.values(), key=lambda m: m.similarity, reverse=True)
        return sorted_matches[:limit]
    
    def delete_by_image_id(self, image_id: str) -> int:
        """
        Delete all face embeddings for an image.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Number of deleted entries
        """
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="image_id",
                        match=qdrant_models.MatchValue(value=image_id)
                    )
                ]
            )
        )
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="image_id",
                            match=qdrant_models.MatchValue(value=image_id)
                        )
                    ]
                )
            ),
            wait=True
        )
        
        deleted_count = count_result.count
        logger.info(f"Deleted {deleted_count} embeddings for image_id: {image_id}")
        return deleted_count
    
    def get_gallery_stats(self) -> Dict[str, int]:
        """
        Get gallery-specific statistics.
        
        Returns:
            Dictionary with total_faces, unique_images counts
        """
        try:
            # Count gallery entries
            gallery_count = self.client.count(
                collection_name=self.collection_name,
                count_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="type",
                            match=qdrant_models.MatchValue(value="gallery")
                        )
                    ]
                )
            )
            
            # Get unique image IDs by scrolling (limited sample for performance)
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="type",
                            match=qdrant_models.MatchValue(value="gallery")
                        )
                    ]
                ),
                limit=10000,
                with_payload=["image_id"],
                with_vectors=False
            )
            
            unique_images = len(set(
                p.payload.get("image_id") for p in results[0] if p.payload
            ))
            
            return {
                "total_faces": gallery_count.count,
                "unique_images": unique_images
            }
        except Exception as e:
            logger.error(f"Failed to get gallery stats: {e}")
            return {"total_faces": 0, "unique_images": 0}
    
    # ================== Saved Gallery Methods ==================
    
    def _get_saved_collection_name(self, gallery_name: str) -> str:
        """Generate collection name for a saved gallery."""
        # Sanitize name: replace spaces and special chars
        safe_name = "".join(c if c.isalnum() else "_" for c in gallery_name.lower())
        return f"saved_gallery_{safe_name}"
    
    def save_gallery(self, name: str) -> Dict[str, Any]:
        """
        Save the current gallery to a new collection.
        
        Args:
            name: Name for the saved gallery
            
        Returns:
            Dictionary with save results
        """
        saved_collection = self._get_saved_collection_name(name)
        
        try:
            # Check if saved collection already exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if saved_collection in collection_names:
                # Delete existing saved collection
                self.client.delete_collection(saved_collection)
                logger.info(f"Deleted existing saved collection: {saved_collection}")
            
            # Create new collection for saved gallery
            self.client.create_collection(
                collection_name=saved_collection,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedding_dim,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            
            # Copy all gallery entries to saved collection
            offset = None
            total_copied = 0
            unique_images = set()
            batch_size = 100
            
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="type",
                                match=qdrant_models.MatchValue(value="gallery")
                            )
                        ]
                    ),
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not results:
                    break
                
                # Prepare points for upsert
                points = []
                for point in results:
                    # Add metadata about save
                    payload = dict(point.payload) if point.payload else {}
                    payload["saved_at"] = datetime.utcnow().isoformat()
                    payload["original_gallery"] = name
                    
                    if payload.get("image_id"):
                        unique_images.add(payload["image_id"])
                    
                    points.append(qdrant_models.PointStruct(
                        id=str(point.id),
                        vector=point.vector if isinstance(point.vector, list) else list(point.vector),
                        payload=payload
                    ))
                
                if points:
                    self.client.upsert(
                        collection_name=saved_collection,
                        points=points,
                        wait=True
                    )
                    total_copied += len(points)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"Saved gallery '{name}': {total_copied} faces, {len(unique_images)} images")
            
            return {
                "success": True,
                "name": name,
                "faces_saved": total_copied,
                "unique_images": len(unique_images)
            }
            
        except Exception as e:
            logger.error(f"Failed to save gallery: {e}")
            # Cleanup on failure
            try:
                self.client.delete_collection(saved_collection)
            except:
                pass
            raise
    
    def load_gallery(self, name: str, clear_current: bool = True) -> Dict[str, Any]:
        """
        Load a saved gallery into the main collection.
        
        Args:
            name: Name of the saved gallery to load
            clear_current: Whether to clear the current gallery first
            
        Returns:
            Dictionary with load results
        """
        saved_collection = self._get_saved_collection_name(name)
        
        try:
            # Check if saved collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if saved_collection not in collection_names:
                raise ValueError(f"Saved gallery '{name}' not found")
            
            # Clear current gallery if requested
            if clear_current:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="type",
                                    match=qdrant_models.MatchValue(value="gallery")
                                )
                            ]
                        )
                    ),
                    wait=True
                )
                logger.info("Cleared current gallery")
            
            # Copy from saved collection to main collection
            offset = None
            total_loaded = 0
            batch_size = 100
            
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=saved_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not results:
                    break
                
                # Prepare points for upsert
                points = []
                for point in results:
                    payload = dict(point.payload) if point.payload else {}
                    # Remove save metadata
                    payload.pop("saved_at", None)
                    payload.pop("original_gallery", None)
                    
                    points.append(qdrant_models.PointStruct(
                        id=str(uuid.uuid4()),  # Generate new IDs
                        vector=point.vector if isinstance(point.vector, list) else list(point.vector),
                        payload=payload
                    ))
                
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    total_loaded += len(points)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"Loaded gallery '{name}': {total_loaded} faces")
            
            return {
                "success": True,
                "name": name,
                "faces_loaded": total_loaded,
                "previous_gallery_cleared": clear_current
            }
            
        except Exception as e:
            logger.error(f"Failed to load gallery: {e}")
            raise
    
    def save_gallery_streaming(self, name: str) -> Generator[Dict[str, Any], None, None]:
        """
        Save the current gallery to a new collection with progress updates.
        
        Yields:
            Progress updates as dictionaries
        """
        saved_collection = self._get_saved_collection_name(name)
        
        try:
            yield {"status": "starting", "message": "Preparing to save...", "progress": 0}
            
            # Get total count first
            gallery_stats = self.get_gallery_stats()
            total_faces = gallery_stats["total_faces"]
            
            if total_faces == 0:
                yield {"status": "error", "message": "No faces in current gallery to save", "progress": 0}
                return
            
            yield {"status": "progress", "message": f"Found {total_faces} faces to save", "progress": 5, "total": total_faces}
            
            # Check if saved collection already exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if saved_collection in collection_names:
                yield {"status": "progress", "message": "Removing existing album...", "progress": 8}
                self.client.delete_collection(saved_collection)
            
            # Create new collection for saved gallery
            yield {"status": "progress", "message": "Creating album...", "progress": 10}
            self.client.create_collection(
                collection_name=saved_collection,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedding_dim,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            
            # Copy all gallery entries to saved collection
            offset = None
            total_copied = 0
            unique_images = set()
            batch_size = 100
            
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="type",
                                match=qdrant_models.MatchValue(value="gallery")
                            )
                        ]
                    ),
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not results:
                    break
                
                # Prepare points for upsert
                points = []
                for point in results:
                    payload = dict(point.payload) if point.payload else {}
                    payload["saved_at"] = datetime.utcnow().isoformat()
                    payload["original_gallery"] = name
                    
                    if payload.get("image_id"):
                        unique_images.add(payload["image_id"])
                    
                    points.append(qdrant_models.PointStruct(
                        id=str(point.id),
                        vector=point.vector if isinstance(point.vector, list) else list(point.vector),
                        payload=payload
                    ))
                
                if points:
                    self.client.upsert(
                        collection_name=saved_collection,
                        points=points,
                        wait=True
                    )
                    total_copied += len(points)
                    
                    # Calculate progress (10-95% range for copying)
                    progress = 10 + int((total_copied / total_faces) * 85)
                    yield {
                        "status": "progress",
                        "message": f"Saving faces... ({total_copied}/{total_faces})",
                        "progress": min(progress, 95),
                        "copied": total_copied,
                        "total": total_faces
                    }
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"Saved gallery '{name}': {total_copied} faces, {len(unique_images)} images")
            
            yield {
                "status": "complete",
                "message": f"Album saved successfully!",
                "progress": 100,
                "name": name,
                "faces_saved": total_copied,
                "unique_images": len(unique_images)
            }
            
        except Exception as e:
            logger.error(f"Failed to save gallery: {e}")
            # Cleanup on failure
            try:
                self.client.delete_collection(saved_collection)
            except:
                pass
            yield {"status": "error", "message": str(e), "progress": 0}
    
    def load_gallery_streaming(self, name: str, clear_current: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Load a saved gallery with progress updates.
        
        Yields:
            Progress updates as dictionaries
        """
        saved_collection = self._get_saved_collection_name(name)
        
        try:
            yield {"status": "starting", "message": "Preparing to load...", "progress": 0}
            
            # Check if saved collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if saved_collection not in collection_names:
                yield {"status": "error", "message": f"Album '{name}' not found", "progress": 0}
                return
            
            # Get total count from saved collection
            info = self.client.get_collection(saved_collection)
            total_faces = info.points_count or 0
            
            if total_faces == 0:
                yield {"status": "error", "message": "Saved album is empty", "progress": 0}
                return
            
            yield {"status": "progress", "message": f"Found {total_faces} faces to load", "progress": 5, "total": total_faces}
            
            # Clear current gallery if requested
            if clear_current:
                yield {"status": "progress", "message": "Clearing current gallery...", "progress": 10}
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="type",
                                    match=qdrant_models.MatchValue(value="gallery")
                                )
                            ]
                        )
                    ),
                    wait=True
                )
            
            # Copy from saved collection to main collection
            offset = None
            total_loaded = 0
            batch_size = 100
            
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=saved_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not results:
                    break
                
                # Prepare points for upsert
                points = []
                for point in results:
                    payload = dict(point.payload) if point.payload else {}
                    payload.pop("saved_at", None)
                    payload.pop("original_gallery", None)
                    
                    points.append(qdrant_models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=point.vector if isinstance(point.vector, list) else list(point.vector),
                        payload=payload
                    ))
                
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    total_loaded += len(points)
                    
                    # Calculate progress (15-95% range for loading)
                    progress = 15 + int((total_loaded / total_faces) * 80)
                    yield {
                        "status": "progress",
                        "message": f"Loading faces... ({total_loaded}/{total_faces})",
                        "progress": min(progress, 95),
                        "loaded": total_loaded,
                        "total": total_faces
                    }
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"Loaded gallery '{name}': {total_loaded} faces")
            
            yield {
                "status": "complete",
                "message": f"Album loaded successfully!",
                "progress": 100,
                "name": name,
                "faces_loaded": total_loaded,
                "previous_gallery_cleared": clear_current
            }
            
        except Exception as e:
            logger.error(f"Failed to load gallery: {e}")
            yield {"status": "error", "message": str(e), "progress": 0}
    
    def list_saved_galleries(self) -> List[Dict[str, Any]]:
        """
        List all saved galleries.
        
        Returns:
            List of saved gallery information
        """
        try:
            collections = self.client.get_collections().collections
            saved_galleries = []
            
            for collection in collections:
                if collection.name.startswith("saved_gallery_"):
                    # Extract gallery name
                    gallery_name = collection.name[14:]  # Remove "saved_gallery_" prefix
                    
                    # Get collection stats
                    try:
                        info = self.client.get_collection(collection.name)
                        points_count = info.points_count or 0
                        
                        # Get sample point to get created_at
                        created_at = ""
                        unique_images = 0
                        
                        sample, _ = self.client.scroll(
                            collection_name=collection.name,
                            limit=1000,
                            with_payload=["saved_at", "image_id"],
                            with_vectors=False
                        )
                        
                        if sample:
                            created_at = sample[0].payload.get("saved_at", "") if sample[0].payload else ""
                            unique_images = len(set(
                                p.payload.get("image_id", "") for p in sample if p.payload
                            ))
                        
                        saved_galleries.append({
                            "name": gallery_name,
                            "created_at": created_at,
                            "total_faces": points_count,
                            "unique_images": unique_images
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get stats for {collection.name}: {e}")
            
            return saved_galleries
            
        except Exception as e:
            logger.error(f"Failed to list saved galleries: {e}")
            return []
    
    def delete_saved_gallery(self, name: str) -> bool:
        """
        Delete a saved gallery.
        
        Args:
            name: Name of the saved gallery to delete
            
        Returns:
            True if deleted successfully
        """
        saved_collection = self._get_saved_collection_name(name)
        
        try:
            self.client.delete_collection(saved_collection)
            logger.info(f"Deleted saved gallery: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete saved gallery: {e}")
            return False
    
    def clear_gallery(self) -> int:
        """
        Clear all gallery entries from main collection.
        
        Returns:
            Number of deleted entries
        """
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="type",
                            match=qdrant_models.MatchValue(value="gallery")
                        )
                    ]
                )
            )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="type",
                                match=qdrant_models.MatchValue(value="gallery")
                            )
                        ]
                    )
                ),
                wait=True
            )
            
            deleted_count = count_result.count
            logger.info(f"Cleared {deleted_count} gallery entries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear gallery: {e}")
            raise


# Singleton instance
_qdrant_instance: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """Get or create the singleton Qdrant service instance."""
    global _qdrant_instance
    if _qdrant_instance is None:
        _qdrant_instance = QdrantService()
    return _qdrant_instance
