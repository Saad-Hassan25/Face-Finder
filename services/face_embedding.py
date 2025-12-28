"""
Face Embedding Service using LVFace
"""

import numpy as np
import cv2
import onnxruntime
from typing import List, Optional, Union
from PIL import Image
import io
import logging

from config import settings

logger = logging.getLogger(__name__)


class FaceEmbeddingService:
    """
    Face embedding service using LVFace.
    LVFace (Large Vision Face) provides state-of-the-art face recognition embeddings
    using Vision Transformer architecture.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: Optional[bool] = None
    ):
        """
        Initialize the LVFace embedding service.
        
        Args:
            model_path: Path to LVFace ONNX model file
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path or settings.lvface_model_path
        self.use_gpu = use_gpu if use_gpu is not None else settings.use_gpu
        self.input_size = (112, 112)  # LVFace expects 112x112 input
        self.embedding_dim = settings.embedding_dim
        
        # Initialize ONNX Runtime session
        try:
            available_providers = onnxruntime.get_available_providers()
            
            if self.use_gpu:
                # Try GPU providers in order of preference
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                elif 'DmlExecutionProvider' in available_providers:
                    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                else:
                    logger.warning("No GPU provider available, falling back to CPU")
                    providers = ['CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"LVFace model loaded from {self.model_path}")
            logger.info(f"Using providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load LVFace model: {e}")
            raise RuntimeError(f"Failed to initialize face embedding model: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for LVFace inference.
        
        Args:
            image: Input image as numpy array (RGB format, 112x112)
            
        Returns:
            Preprocessed image tensor ready for inference
        """
        # Ensure correct size
        if image.shape[:2] != self.input_size[::-1]:
            image = cv2.resize(image, self.input_size)
        
        # Convert to float and transpose to (C, H, W)
        img_transposed = np.transpose(image, (2, 0, 1))
        
        # Normalize to [-1, 1] using torch-style normalization
        # (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1
        img_normalized = (img_transposed / 255.0 - 0.5) / 0.5
        
        # Add batch dimension and convert to float32
        img_tensor = img_normalized.astype(np.float32)[np.newaxis, ...]
        
        return img_tensor
    
    def _convert_to_numpy(self, image: Union[np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """
        Convert various image formats to numpy array in RGB format.
        
        Args:
            image: Input image as numpy array, PIL Image, or bytes
            
        Returns:
            Numpy array in RGB format
        """
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # If BGR (from OpenCV), convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's already RGB if it came from PIL
            pass
        
        return image
    
    def get_embedding(self, face_image: Union[np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """
        Extract face embedding from a single aligned face image.
        
        Args:
            face_image: Aligned face image (112x112, RGB format preferred)
            
        Returns:
            Face embedding as 1D numpy array
        """
        # Convert to numpy array
        img_array = self._convert_to_numpy(face_image)
        
        # Preprocess
        img_tensor = self._preprocess_image(img_array)
        
        # Run inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: img_tensor}
        )
        
        embedding = output[0]
        
        # Flatten and normalize
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def get_embeddings_batch(
        self,
        face_images: List[Union[np.ndarray, Image.Image, bytes]]
    ) -> List[np.ndarray]:
        """
        Extract face embeddings from multiple aligned face images.
        
        Args:
            face_images: List of aligned face images
            
        Returns:
            List of face embeddings
        """
        if not face_images:
            return []
        
        # Preprocess all images
        batch_tensors = []
        for img in face_images:
            img_array = self._convert_to_numpy(img)
            img_tensor = self._preprocess_image(img_array)
            batch_tensors.append(img_tensor[0])  # Remove batch dimension
        
        # Stack into batch
        batch = np.stack(batch_tensors, axis=0).astype(np.float32)
        
        # Run batch inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: batch}
        )
        
        embeddings = output[0]
        
        # Normalize each embedding
        result = []
        for i in range(embeddings.shape[0]):
            emb = embeddings[i].flatten()
            emb = emb / np.linalg.norm(emb)
            result.append(emb)
        
        return result
    
    def get_embedding_with_flip(
        self,
        face_image: Union[np.ndarray, Image.Image, bytes]
    ) -> np.ndarray:
        """
        Extract face embedding with horizontal flip augmentation.
        The final embedding is the average of original and flipped embeddings.
        This typically improves recognition accuracy.
        
        Args:
            face_image: Aligned face image
            
        Returns:
            Face embedding as 1D numpy array
        """
        # Convert to numpy array
        img_array = self._convert_to_numpy(face_image)
        
        # Create flipped version
        img_flipped = np.fliplr(img_array)
        
        # Get embeddings for both
        emb_original = self.get_embedding(img_array)
        emb_flipped = self.get_embedding(img_flipped)
        
        # Average and normalize
        combined = emb_original + emb_flipped
        combined = combined / np.linalg.norm(combined)
        
        return combined
    
    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Cosine similarity score (range: [-1, 1], higher is more similar)
        """
        # Flatten embeddings
        emb1 = embedding1.flatten()
        emb2 = embedding2.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 > 0 and norm2 > 0:
            return float(dot_product / (norm1 * norm2))
        return 0.0
    
    @staticmethod
    def is_same_person(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.6
    ) -> bool:
        """
        Determine if two face embeddings belong to the same person.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold for matching
            
        Returns:
            True if embeddings are from the same person
        """
        similarity = FaceEmbeddingService.calculate_similarity(embedding1, embedding2)
        return similarity >= threshold
    
    def is_available(self) -> bool:
        """Check if the embedding service is available and working."""
        try:
            # Create a test image
            test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            embedding = self.get_embedding(test_image)
            return embedding is not None and len(embedding) > 0
        except Exception as e:
            logger.error(f"Face embedding health check failed: {e}")
            return False


# Singleton instance
_embedder_instance: Optional[FaceEmbeddingService] = None


def get_face_embedder() -> FaceEmbeddingService:
    """Get or create the singleton face embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = FaceEmbeddingService()
    return _embedder_instance
