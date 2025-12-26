"""
Vector Database Service using Qdrant
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
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
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize Qdrant service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection for face embeddings
            api_key: Optional API key for Qdrant Cloud
            embedding_dim: Dimension of face embeddings
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.api_key = api_key or settings.qdrant_api_key
        self.embedding_dim = embedding_dim or settings.embedding_dim
        
        # Initialize Qdrant client
        try:
            if self.api_key:
                # Qdrant Cloud
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    https=True
                )
            else:
                # Local Qdrant
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port
                )
            
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
            # Ensure collection exists
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Failed to initialize Qdrant service: {e}")
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
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
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
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
            return CollectionStats(
                total_faces=info.points_count or 0,
                vectors_count=info.vectors_count or 0,
                indexed_vectors_count=info.indexed_vectors_count or 0
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


# Singleton instance
_qdrant_instance: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """Get or create the singleton Qdrant service instance."""
    global _qdrant_instance
    if _qdrant_instance is None:
        _qdrant_instance = QdrantService()
    return _qdrant_instance
