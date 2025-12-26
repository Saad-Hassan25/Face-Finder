"""
Face Detection Service using SCRFD
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
import cv2
import io
import logging

from scrfd import SCRFD, Threshold

from config import settings
from models import DetectedFace, BoundingBox, FaceKeypoints, Keypoint

logger = logging.getLogger(__name__)


class FaceDetectionService:
    """
    Face detection service using SCRFD (Sample and Computation Redistribution for Face Detection).
    SCRFD provides efficient and accurate face detection with keypoint localization.
    """
    
    def __init__(self, model_path: Optional[str] = None, threshold: Optional[float] = None):
        """
        Initialize the face detection service.
        
        Args:
            model_path: Path to SCRFD ONNX model file
            threshold: Detection confidence threshold (0.0 - 1.0)
        """
        self.model_path = model_path or settings.scrfd_model_path
        self.threshold_value = threshold or settings.detection_threshold
        
        # Initialize SCRFD model
        try:
            self.detector = SCRFD.from_path(self.model_path)
            self.threshold = Threshold(probability=self.threshold_value)
            logger.info(f"SCRFD model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load SCRFD model: {e}")
            raise RuntimeError(f"Failed to initialize face detector: {e}")
    
    def _convert_to_pil_image(self, image: Union[np.ndarray, Image.Image, bytes]) -> Image.Image:
        """
        Convert various image formats to PIL Image.
        
        Args:
            image: Input image as numpy array, PIL Image, or bytes
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            # Assume BGR format from OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def detect_faces(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        return_keypoints: bool = True
    ) -> List[DetectedFace]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (numpy array, PIL Image, or bytes)
            return_keypoints: Whether to return facial keypoints
            
        Returns:
            List of DetectedFace objects with bounding boxes and optionally keypoints
        """
        # Convert to PIL Image
        pil_image = self._convert_to_pil_image(image)
        
        # Run detection
        faces = self.detector.detect(pil_image, threshold=self.threshold)
        
        detected_faces = []
        for idx, face in enumerate(faces):
            # Extract bounding box (SCRFD uses upper_left/lower_right Points)
            bbox = BoundingBox(
                x=float(face.bbox.upper_left.x),
                y=float(face.bbox.upper_left.y),
                width=float(face.bbox.lower_right.x - face.bbox.upper_left.x),
                height=float(face.bbox.lower_right.y - face.bbox.upper_left.y)
            )
            
            # Extract keypoints if available and requested
            keypoints = None
            if return_keypoints and face.keypoints is not None:
                kps = face.keypoints
                keypoints = FaceKeypoints(
                    left_eye=Keypoint(x=float(kps.left_eye.x), y=float(kps.left_eye.y)),
                    right_eye=Keypoint(x=float(kps.right_eye.x), y=float(kps.right_eye.y)),
                    nose=Keypoint(x=float(kps.nose.x), y=float(kps.nose.y)),
                    left_mouth=Keypoint(x=float(kps.left_mouth.x), y=float(kps.left_mouth.y)),
                    right_mouth=Keypoint(x=float(kps.right_mouth.x), y=float(kps.right_mouth.y))
                )
            
            detected_face = DetectedFace(
                bbox=bbox,
                confidence=float(face.probability),
                keypoints=keypoints,
                face_index=idx
            )
            detected_faces.append(detected_face)
        
        return detected_faces

    def extract_face_regions(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        faces: Optional[List[DetectedFace]] = None,
        margin: float = 0.1,
        target_size: Tuple[int, int] = (112, 112)
    ) -> List[np.ndarray]:
        """
        Extract and align face regions from image.
        
        Args:
            image: Input image
            faces: List of detected faces (if None, will detect automatically)
            margin: Margin around face as percentage of face size
            target_size: Target size for extracted faces (width, height)
            
        Returns:
            List of extracted face images as numpy arrays (RGB format)
        """
        # Convert to numpy array
        pil_image = self._convert_to_pil_image(image)
        img_array = np.array(pil_image)
        
        # Detect faces if not provided
        if faces is None:
            faces = self.detect_faces(image)
        
        face_images = []
        img_h, img_w = img_array.shape[:2]
        
        for face in faces:
            bbox = face.bbox
            
            # Calculate margin
            margin_w = bbox.width * margin
            margin_h = bbox.height * margin
            
            # Expand bounding box with margin
            x1 = max(0, int(bbox.x - margin_w))
            y1 = max(0, int(bbox.y - margin_h))
            x2 = min(img_w, int(bbox.x + bbox.width + margin_w))
            y2 = min(img_h, int(bbox.y + bbox.height + margin_h))
            
            # Extract face region
            face_region = img_array[y1:y2, x1:x2]
            
            # Resize to target size
            if face_region.size > 0:
                face_resized = cv2.resize(face_region, target_size)
                face_images.append(face_resized)
        
        return face_images
    
    def align_face(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        keypoints: FaceKeypoints,
        target_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        Align face using facial keypoints (similar to ArcFace alignment).
        
        Args:
            image: Input image
            keypoints: Facial keypoints
            target_size: Target size for aligned face
            
        Returns:
            Aligned face image as numpy array
        """
        # Convert to numpy array
        pil_image = self._convert_to_pil_image(image)
        img_array = np.array(pil_image)
        
        # Source keypoints from detection
        src_pts = np.array([
            [keypoints.left_eye.x, keypoints.left_eye.y],
            [keypoints.right_eye.x, keypoints.right_eye.y],
            [keypoints.nose.x, keypoints.nose.y],
            [keypoints.left_mouth.x, keypoints.left_mouth.y],
            [keypoints.right_mouth.x, keypoints.right_mouth.y]
        ], dtype=np.float32)
        
        # Standard reference keypoints for 112x112 aligned face
        # Based on InsightFace/ArcFace alignment
        dst_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Compute similarity transform
        from skimage import transform as trans
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, dst_pts)
        
        # Apply transformation
        aligned = cv2.warpAffine(
            img_array,
            tform.params[0:2, :],
            target_size,
            borderValue=0.0
        )
        
        return aligned
    
    def get_aligned_faces(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        faces: Optional[List[DetectedFace]] = None
    ) -> List[np.ndarray]:
        """
        Get aligned face images using keypoint-based alignment.
        
        Args:
            image: Input image
            faces: List of detected faces (if None, will detect automatically)
            
        Returns:
            List of aligned face images (112x112, RGB format)
        """
        if faces is None:
            faces = self.detect_faces(image, return_keypoints=True)
        
        aligned_faces = []
        for face in faces:
            if face.keypoints is not None:
                aligned = self.align_face(image, face.keypoints)
                aligned_faces.append(aligned)
            else:
                # Fallback to simple crop if no keypoints
                crops = self.extract_face_regions(image, [face])
                if crops:
                    aligned_faces.append(crops[0])
        
        return aligned_faces
    
    def is_available(self) -> bool:
        """Check if the detector is available and working."""
        try:
            # Create a small test image
            test_image = Image.new('RGB', (100, 100), color='white')
            self.detector.detect(test_image, threshold=self.threshold)
            return True
        except Exception as e:
            logger.error(f"Face detector health check failed: {e}")
            return False


# Singleton instance
_detector_instance: Optional[FaceDetectionService] = None


def get_face_detector() -> FaceDetectionService:
    """Get or create the singleton face detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetectionService()
    return _detector_instance
