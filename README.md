# Face-Finder
Face Finder is a face recognition system designed to help users **find themselves (or any person) across a collection of photos**. Think of it like Google Photos or Apple Photos' "Find My Face" feature.

## Table of Contents
1. [Overview](#overview)
2. [Primary Use Case: Find Me In Photos](#primary-use-case-find-me-in-photos)
3. [Functional Requirements](#functional-requirements)
4. [System Components](#system-components)
5. [Application Flow](#application-flow)
6. [Component Details](#component-details)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [API Endpoints Summary](#api-endpoints-summary)

---

## Overview

Face Finder is a face recognition system designed to help users **find themselves (or any person) across a collection of photos**. Think of it like Google Photos or Apple Photos' "Find My Face" feature.

### Primary Features
- **Image Gallery Indexing**: Upload multiple images to build a searchable gallery
- **Person Search**: Find all images where a specific person appears
- **Face Detection**: Locate faces in images
- **Face Verification**: Compare two faces for identity match

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Face Detection | SCRFD | Detect faces and extract keypoints |
| Face Embedding | LVFace | Generate 512-dim face embeddings |
| Vector Database | Qdrant | Store and search face embeddings |
| API Framework | FastAPI | REST API endpoints |
| Runtime | ONNX Runtime | Model inference engine |

---

## Primary Use Case: Find Me In Photos

### The Problem
You have a collection of photos (from an event, party, wedding, etc.) and want to find all photos where a specific person appears.

### The Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: INDEX PHOTOS                              │
│                                                                              │
│   Upload all event photos                                                    │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Photo 1   │     │   Photo 2   │     │   Photo N   │                  │
│   │  3 people   │     │  2 people   │ ... │  5 people   │                  │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│          │                   │                   │                          │
│          ▼                   ▼                   ▼                          │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                    SCRFD: Detect All Faces                          │   │
│   │              LVFace: Extract Face Embeddings                        │   │
│   └───────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                         QDRANT DATABASE                             │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │ Face 1 → embedding + {image_id: "photo1", face_index: 0}    │   │   │
│   │  │ Face 2 → embedding + {image_id: "photo1", face_index: 1}    │   │   │
│   │  │ Face 3 → embedding + {image_id: "photo1", face_index: 2}    │   │   │
│   │  │ Face 4 → embedding + {image_id: "photo2", face_index: 0}    │   │   │
│   │  │ ...                                                          │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

                                    │
                                    │ Later...
                                    ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: FIND PERSON                                  │
│                                                                              │
│   Upload reference photo of person to find                                   │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────────┐                                                       │
│   │  "Find photos   │                                                       │
│   │   of John"      │                                                       │
│   │   ┌─────┐       │                                                       │
│   │   │ 😊  │       │                                                       │
│   │   └─────┘       │                                                       │
│   └────────┬────────┘                                                       │
│            │                                                                 │
│            ▼                                                                 │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │           Extract John's Face Embedding                             │   │
│   └───────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                    QDRANT: Vector Similarity Search                 │   │
│   │                                                                      │   │
│   │     Query: John's embedding                                          │   │
│   │     Find: All similar face embeddings                                │   │
│   │     Return: Unique image_ids where similarity > threshold            │   │
│   └───────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                         RESULTS                                      │   │
│   │                                                                      │   │
│   │   Found John in 15 images:                                           │   │
│   │   - photo1.jpg (similarity: 0.92)                                    │   │
│   │   - photo5.jpg (similarity: 0.88)                                    │   │
│   │   - photo12.jpg (similarity: 0.85)                                   │   │
│   │   - ...                                                              │   │
│   └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example Workflow

```python
# Step 1: Index all event photos
POST /gallery/index-bulk
Files: [event_photo_001.jpg, event_photo_002.jpg, ..., event_photo_100.jpg]

# Response:
{
    "success": true,
    "total_images": 100,
    "total_faces_indexed": 287,  # 287 faces found across 100 images
    "images_processed": [...]
}

# Step 2: Find a specific person
POST /gallery/find-person
File: johns_selfie.jpg  # Reference photo of John

# Response:
{
    "success": true,
    "total_images_found": 15,
    "images": [
        {"image_id": "uuid-1", "image_name": "event_photo_007.jpg", "similarity": 0.92},
        {"image_id": "uuid-2", "image_name": "event_photo_023.jpg", "similarity": 0.88},
        {"image_id": "uuid-3", "image_name": "event_photo_041.jpg", "similarity": 0.85},
        ...
    ]
}
```

---

## Functional Requirements

### FR-1: Image Gallery Indexing (Primary)
- **Input**: One or more image files
- **Output**: Confirmation with image_id and face count
- **Process**:
  - Accept image upload(s)
  - Detect all faces in each image using SCRFD
  - Extract embeddings for each face using LVFace
  - Store embeddings in Qdrant with image metadata
- **Endpoints**: `POST /gallery/index`, `POST /gallery/index-bulk`

### FR-2: Find Person In Images (Primary)
- **Input**: Reference image of the person to find
- **Output**: List of images where the person appears
- **Process**:
  - Detect face(s) in reference image
  - Extract embedding(s)
  - Search Qdrant for similar face vectors
  - Return deduplicated list of matching images
- **Endpoint**: `POST /gallery/find-person`

### FR-3: Face Detection
- **Input**: Image file (JPEG, PNG, etc.)
- **Output**: List of detected faces with bounding boxes and keypoints
- **Process**: 
  - Accept image upload
  - Run SCRFD model to detect faces
  - Return face locations and confidence scores
- **Endpoint**: `POST /detect`

### FR-4: Face Embedding Extraction
- **Input**: Image file containing one or more faces
- **Output**: 512-dimensional embedding vector per face
- **Process**:
  - Detect faces using SCRFD
  - Align faces using 5-point keypoints
  - Extract embeddings using LVFace model
- **Endpoint**: `POST /embed`

### FR-5: Face Verification
- **Input**: Two image files
- **Output**: Boolean match result + similarity score
- **Process**:
  - Extract embeddings from both images
  - Calculate cosine similarity
  - Compare against threshold
- **Endpoint**: `POST /verify`

### FR-6: Face Registration (Person Database)
- **Input**: Image file + Person ID + Optional Name
- **Output**: Confirmation with face ID(s)
- **Process**:
  - Detect and align faces
  - Extract embeddings
  - Store embeddings in Qdrant with person metadata
- **Endpoint**: `POST /register`

### FR-7: Face Search (Person Database)
- **Input**: Query image file
- **Output**: List of matching persons with similarity scores
- **Process**:
  - Detect and align faces in query image
  - Extract embeddings
  - Search Qdrant for similar vectors in registered persons
  - Return ranked matches
- **Endpoint**: `POST /search`

---

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                            │
│                         (main.py)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   /detect    │  │   /embed     │  │   /register          │  │
│  │   /search    │  │   /verify    │  │   /person/{id}       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
├─────────┴─────────────────┴──────────────────────┴──────────────┤
│                      Service Layer                               │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐   │
│  │ FaceDetection  │ │ FaceEmbedding  │ │   QdrantService    │   │
│  │    Service     │ │    Service     │ │                    │   │
│  │   (SCRFD)      │ │   (LVFace)     │ │   (Vector DB)      │   │
│  └────────┬───────┘ └────────┬───────┘ └─────────┬──────────┘   │
│           │                  │                    │              │
├───────────┴──────────────────┴────────────────────┴──────────────┤
│                      Model/Storage Layer                         │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐   │
│  │  scrfd.onnx    │ │  lvface.onnx   │ │   Qdrant Server    │   │
│  │  (Detection)   │ │  (Embedding)   │ │   (Port 6333)      │   │
│  └────────────────┘ └────────────────┘ └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Application Flow

### Flow 1: Gallery Indexing (`/gallery/index-bulk`) - PRIMARY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GALLERY INDEXING FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │ User uploads│
    │ 100 photos  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    FOR EACH IMAGE                                    │
    │                                                                      │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
    │  │   SCRFD     │───▶│   LVFace    │───▶│  Store in Qdrant        │ │
    │  │  Detect all │    │  Extract    │    │                         │ │
    │  │  faces (N)  │    │  N embeds   │    │  For each face:         │ │
    │  └─────────────┘    └─────────────┘    │  - embedding vector     │ │
    │                                         │  - image_id             │ │
    │                                         │  - image_name           │ │
    │                                         │  - face_index           │ │
    │                                         │  - bounding_box         │ │
    │                                         │  - type: "gallery"      │ │
    │                                         └─────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  QDRANT DATABASE STATE AFTER INDEXING                               │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │  287 face vectors stored from 100 images                        │ │
    │  │                                                                  │ │
    │  │  Vector 1: [0.12, -0.45, ...] → {image_id: "img001", face: 0}  │ │
    │  │  Vector 2: [0.08, -0.32, ...] → {image_id: "img001", face: 1}  │ │
    │  │  Vector 3: [-0.15, 0.28, ...] → {image_id: "img001", face: 2}  │ │
    │  │  Vector 4: [0.22, -0.18, ...] → {image_id: "img002", face: 0}  │ │
    │  │  ... (287 total)                                                │ │
    │  └────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
```

### Flow 2: Find Person (`/gallery/find-person`) - PRIMARY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIND PERSON FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ User uploads    │
    │ reference photo │
    │ of "John"       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 1: Extract Query Embedding                                     │
    │                                                                      │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
    │  │   SCRFD     │───▶│   LVFace    │───▶│  Query Embedding        │ │
    │  │  Detect     │    │  Extract    │    │  [0.15, -0.42, ...]     │ │
    │  │  John's face│    │  embedding  │    │  (512 dimensions)       │ │
    │  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 2: Vector Similarity Search                                    │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │  QDRANT: Find all vectors similar to John's embedding          │ │
    │  │                                                                  │ │
    │  │  Query: [0.15, -0.42, ...]                                      │ │
    │  │  Filter: type = "gallery"                                        │ │
    │  │  Threshold: similarity > 0.6                                     │ │
    │  │                                                                  │ │
    │  │  Results:                                                        │ │
    │  │  - Vector 4 (img002, face 0): similarity = 0.92  ✓ John!        │ │
    │  │  - Vector 15 (img007, face 2): similarity = 0.88 ✓ John!        │ │
    │  │  - Vector 23 (img012, face 1): similarity = 0.85 ✓ John!        │ │
    │  │  - Vector 89 (img045, face 0): similarity = 0.71 ✓ John!        │ │
    │  │  ...                                                             │ │
    │  └────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STEP 3: Deduplicate by Image                                        │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │  Multiple faces in same image? Keep only highest similarity.   │ │
    │  │                                                                  │ │
    │  │  Final Results (unique images):                                  │ │
    │  │  1. img002.jpg - similarity: 0.92                               │ │
    │  │  2. img007.jpg - similarity: 0.88                               │ │
    │  │  3. img012.jpg - similarity: 0.85                               │ │
    │  │  4. img045.jpg - similarity: 0.71                               │ │
    │  │  ... (15 total images)                                          │ │
    │  └────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  RESPONSE                                                            │
    │  {                                                                   │
    │    "success": true,                                                  │
    │    "total_images_found": 15,                                         │
    │    "images": [                                                       │
    │      {"image_name": "img002.jpg", "similarity": 0.92, ...},         │
    │      {"image_name": "img007.jpg", "similarity": 0.88, ...},         │
    │      ...                                                             │
    │    ]                                                                 │
    │  }                                                                   │
    └─────────────────────────────────────────────────────────────────────┘
```

### Flow 3: Face Detection (`/detect`)

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Client  │───▶│  FastAPI    │───▶│   SCRFD     │───▶│  Response    │
│  Image   │    │  Endpoint   │    │  Detector   │    │  (faces)     │
└──────────┘    └─────────────┘    └─────────────┘    └──────────────┘

Step-by-step:
1. Client uploads image to /detect endpoint
2. Image converted to PIL format (RGB)
3. SCRFD model processes image:
   - Resizes to 640x640 with padding
   - Runs neural network inference
   - Applies NMS (Non-Maximum Suppression)
4. Returns detected faces with:
   - Bounding box (x, y, width, height)
   - Confidence score (0-1)
   - 5 facial keypoints (eyes, nose, mouth corners)
```

### Flow 2: Face Embedding (`/embed`)

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Client  │───▶│  FastAPI    │───▶│   SCRFD     │───▶│  LVFace     │───▶│  Response    │
│  Image   │    │  Endpoint   │    │  Detection  │    │  Embedding  │    │  (vectors)   │
└──────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘

Step-by-step:
1. Client uploads image to /embed endpoint
2. SCRFD detects faces and extracts keypoints
3. Face alignment using similarity transform:
   - Source: detected 5 keypoints
   - Target: standard ArcFace alignment template
   - Output: 112x112 aligned face image
4. LVFace model extracts embedding:
   - Preprocesses: normalize to [-1, 1]
   - Runs ViT (Vision Transformer) inference
   - Outputs 512-dimensional vector
5. Returns normalized embedding vectors
```

### Flow 3: Face Registration (`/register`)

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Client  │───▶│  FastAPI    │───▶│   SCRFD     │───▶│  LVFace     │───▶│   Qdrant    │───▶│  Response    │
│  Image   │    │  Endpoint   │    │  Detection  │    │  Embedding  │    │   Storage   │    │  (face_id)   │
│  +ID     │    │             │    │  +Alignment │    │             │    │             │    │              │
└──────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘

Step-by-step:
1. Client uploads image + person_id + optional name
2. SCRFD detects and provides keypoints
3. Faces aligned to 112x112 standard format
4. LVFace extracts embeddings (with flip augmentation)
5. Qdrant stores:
   - Vector: 512-dim embedding
   - Payload: {person_id, person_name, created_at}
   - ID: UUID for each face
6. Returns face_id(s) for stored embeddings
```

### Flow 4: Face Search (`/search`)

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Client  │───▶│  FastAPI    │───▶│   SCRFD     │───▶│  LVFace     │───▶│   Qdrant    │───▶│  Response    │
│  Query   │    │  Endpoint   │    │  Detection  │    │  Embedding  │    │   Search    │    │  (matches)   │
│  Image   │    │             │    │  +Alignment │    │             │    │             │    │              │
└──────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘

Step-by-step:
1. Client uploads query image
2. SCRFD detects faces in query
3. Faces aligned and embeddings extracted
4. Qdrant performs similarity search:
   - Distance metric: Cosine similarity
   - Filters: score_threshold, limit
5. Returns ranked matches per query face:
   - person_id, person_name
   - similarity score
   - face_id, metadata
```

### Flow 5: Face Verification (`/verify`)

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Client  │───▶│  FastAPI    │───▶│   SCRFD     │───▶│  LVFace     │───▶│  Cosine     │───▶│  Response    │
│  Image1  │    │  Endpoint   │    │  Detection  │    │  Embedding  │    │  Similarity │    │  (match)     │
│  Image2  │    │             │    │  (both)     │    │  (both)     │    │  Compare    │    │              │
└──────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘

Step-by-step:
1. Client uploads two images
2. SCRFD detects faces in both images
3. First face from each image aligned
4. LVFace extracts embeddings for both
5. Cosine similarity calculated:
   similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
6. Compare against threshold (default: 0.6)
7. Returns: is_same_person, similarity score
```

---

## Component Details

### SCRFD (Face Detection)

**Model**: `scrfd_10g_bnkps.onnx`

**Purpose**: Detect faces and extract 5 facial keypoints for alignment

**How it works**:
```
Input Image (any size)
        │
        ▼
┌───────────────────┐
│  Resize & Pad     │  → 640x640 with aspect ratio preserved
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  SCRFD Network    │  → Multi-scale feature extraction
│  (ONNX Runtime)   │  → Anchor-based detection
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Post-processing  │  → Score filtering (threshold=0.5)
│                   │  → NMS (IoU threshold=0.5)
└─────────┬─────────┘
          │
          ▼
Output: List[Face]
  - bbox: Bbox(upper_left, lower_right)
  - probability: float (0-1)
  - keypoints: FaceKeypoints(left_eye, right_eye, nose, left_mouth, right_mouth)
```

**Key Features**:
- Efficient single-stage detector
- Returns 5-point facial landmarks
- Handles multiple faces per image
- Scale-invariant detection

---

### LVFace (Face Embedding)

**Model**: `lvface.onnx` (LVFace-T, LVFace-S, or LVFace-B)

**Purpose**: Generate discriminative 512-dimensional face embeddings

**How it works**:
```
Aligned Face (112x112 RGB)
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  → Normalize: (pixel/255 - 0.5) / 0.5
│                   │  → Transpose: HWC → CHW
│                   │  → Shape: (1, 3, 112, 112)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Vision           │  → Patch embedding (9x9 patches)
│  Transformer      │  → Self-attention layers
│  (ONNX Runtime)   │  → Feature aggregation
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Post-processing  │  → L2 normalization
└─────────┬─────────┘
          │
          ▼
Output: numpy.ndarray (512,)
  - Unit-normalized embedding vector
  - Comparable via cosine similarity
```

**Face Alignment Process**:
```python
# Standard reference keypoints for 112x112 aligned face
# Based on InsightFace/ArcFace alignment
dst_pts = [
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041]    # right mouth
]

# Similarity transform: detected keypoints → reference keypoints
# Applied via cv2.warpAffine
```

**Flip Augmentation**:
- Original + horizontally flipped image
- Average both embeddings
- Improves recognition accuracy

---

### Qdrant (Vector Database)

**Purpose**: Store, index, and search face embedding vectors

**Configuration**:
```
Collection: face_embeddings
Vector Size: 512
Distance Metric: Cosine
```

**Data Structure**:
```json
{
  "id": "uuid-string",
  "vector": [0.1, -0.2, ..., 0.05],  // 512 dimensions
  "payload": {
    "person_id": "john_doe",
    "person_name": "John Doe",
    "created_at": "2025-12-23T10:00:00Z"
  }
}
```

**Operations**:

| Operation | Method | Description |
|-----------|--------|-------------|
| Insert | `upsert()` | Add new face embedding |
| Search | `query_points()` | Find similar vectors |
| Delete | `delete()` | Remove by ID or filter |
| Count | `count()` | Get collection statistics |

**Search Process**:
```
Query Vector (512-dim)
        │
        ▼
┌───────────────────┐
│  HNSW Index       │  → Approximate Nearest Neighbor
│  (Qdrant)         │  → Cosine similarity scoring
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Filtering        │  → Score threshold (default: 0.6)
│                   │  → Result limit (default: 10)
└─────────┬─────────┘
          │
          ▼
Output: List[ScoredPoint]
  - id: face_id
  - score: similarity (0-1)
  - payload: metadata
```

---

## Data Flow Diagrams

### Complete Registration Flow

```
                                    ┌─────────────────────────────────────┐
                                    │           INPUT                      │
                                    │  - Image file (JPG/PNG)             │
                                    │  - person_id: "john_doe"            │
                                    │  - person_name: "John Doe"          │
                                    └─────────────┬───────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    DETECTION PHASE                                   │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────────────────────┐ │
│  │ Load Image   │───▶│ SCRFD Detection   │───▶│ Output: 2 faces detected         │ │
│  │ Convert RGB  │    │ threshold=0.5     │    │ Face 1: bbox, kps, score=0.87    │ │
│  └──────────────┘    └───────────────────┘    │ Face 2: bbox, kps, score=0.82    │ │
│                                               └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    ALIGNMENT PHASE                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ For each detected face:                                                       │  │
│  │   1. Extract 5 keypoints from detection                                       │  │
│  │   2. Compute similarity transform to reference points                         │  │
│  │   3. Apply warpAffine to get 112x112 aligned face                            │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌─────────────────┐                              ┌─────────────────┐              │
│  │ Aligned Face 1  │                              │ Aligned Face 2  │              │
│  │   112x112 RGB   │                              │   112x112 RGB   │              │
│  └─────────────────┘                              └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   EMBEDDING PHASE                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ For each aligned face:                                                        │  │
│  │   1. Preprocess: normalize to [-1, 1]                                        │  │
│  │   2. Run LVFace model (with flip augmentation)                               │  │
│  │   3. L2 normalize output embedding                                           │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌─────────────────────────┐                  ┌─────────────────────────┐          │
│  │ Embedding 1             │                  │ Embedding 2             │          │
│  │ [0.02, -0.15, ..., 0.08]│                  │ [0.05, -0.12, ..., 0.03]│          │
│  │ (512 dimensions)        │                  │ (512 dimensions)        │          │
│  └─────────────────────────┘                  └─────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    STORAGE PHASE                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ Qdrant upsert:                                                                │  │
│  │   Collection: face_embeddings                                                 │  │
│  │   Points:                                                                     │  │
│  │     - {id: "uuid-1", vector: emb1, payload: {person_id, person_name, ...}}   │  │
│  │     - {id: "uuid-2", vector: emb2, payload: {person_id, person_name, ...}}   │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────────────────┐
                                    │           OUTPUT                     │
                                    │  {                                   │
                                    │    "success": true,                  │
                                    │    "person_id": "john_doe",          │
                                    │    "faces_registered": 2,            │
                                    │    "face_id": "uuid-1"               │
                                    │  }                                   │
                                    └─────────────────────────────────────┘
```

### Complete Search Flow

```
Query Image
     │
     ▼
┌────────────────┐
│ Face Detection │ ──▶ N faces found
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Face Alignment │ ──▶ N aligned faces (112x112)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ LVFace Embed   │ ──▶ N embeddings (512-dim each)
└───────┬────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────────────────────────┐
│ Qdrant Search  │ ───▶│ For each query embedding:           │
│                │     │   - Find top-K similar vectors      │
│                │     │   - Filter by score_threshold       │
│                │     │   - Return matches with metadata    │
└───────┬────────┘     └─────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│ Response:                                                   │
│ {                                                           │
│   "query_faces": 2,                                         │
│   "matches": [                                              │
│     [  // Matches for face 1                                │
│       {"person_id": "john", "similarity": 0.92, ...},       │
│       {"person_id": "jane", "similarity": 0.71, ...}        │
│     ],                                                      │
│     [  // Matches for face 2                                │
│       {"person_id": "bob", "similarity": 0.88, ...}         │
│     ]                                                       │
│   ]                                                         │
│ }                                                           │
└────────────────────────────────────────────────────────────┘
```

---

## API Endpoints Summary

### Gallery Endpoints (Primary Use Case)

| Endpoint | Method | Input | Output | Description |
|----------|--------|-------|--------|-------------|
| `/gallery/index` | POST | Image + optional metadata | image_id, face count | Index a single image |
| `/gallery/index-bulk` | POST | Multiple images | Summary of all indexed | Index many images at once |
| `/gallery/find-person` | POST | Reference image | List of matching images | Find all photos of a person |
| `/gallery/stats` | GET | - | Gallery statistics | Get indexed image/face counts |
| `/gallery/image/{id}` | DELETE | image_id | Deleted count | Remove image from gallery |

### Face Operations

| Endpoint | Method | Input | Output | Components Used |
|----------|--------|-------|--------|-----------------|
| `/detect` | POST | Image | Faces with bboxes & keypoints | SCRFD |
| `/embed` | POST | Image | 512-dim embeddings | SCRFD + LVFace |
| `/verify` | POST | 2 Images | similarity score | SCRFD + LVFace |

### Person Registration (Secondary Use Case)

| Endpoint | Method | Input | Output | Description |
|----------|--------|-------|--------|-------------|
| `/register` | POST | Image + person_id | face_id(s) | Register a known person |
| `/search` | POST | Image | Matched persons | Search registered persons |
| `/person/{id}` | DELETE | person_id | Deleted count | Remove person |
| `/face/{id}` | DELETE | face_id | Success status | Remove specific face |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Component health status |
| `/stats` | GET | Database statistics |
| `/collection` | DELETE | Clear all data |

---

## File Structure

```
Face Finder/
├── main.py                      # FastAPI app, all endpoints
├── config.py                    # Settings from environment
├── models.py                    # Pydantic request/response schemas
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration
├── services/
│   ├── face_detection.py        # SCRFD wrapper
│   ├── face_embedding.py        # LVFace wrapper  
│   └── qdrant_service.py        # Qdrant operations
└── models/
    ├── scrfd.onnx               # Face detection model
    └── lvface.onnx              # Face embedding model
```

---

## Performance Considerations

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Face Detection | 50-100ms | Single image, CPU |
| Face Alignment | 5-10ms | Per face |
| Face Embedding | 30-50ms | Per face, CPU |
| Qdrant Search | 1-5ms | Depends on collection size |
| Index Single Image | 100-300ms | Depends on faces in image |
| Find Person | 50-150ms | 1 query face, CPU |
| Bulk Index (100 images) | 10-30s | Depends on total faces |

**GPU Acceleration**: Set `USE_GPU=true` for 3-5x speedup on embedding extraction.

---

## Similarity Thresholds

| Threshold | Use Case | False Accept Rate |
|-----------|----------|-------------------|
| 0.4 | Loose matching, recall-focused (find more photos, may include wrong matches) | Higher |
| 0.5 | Balanced | Moderate |
| 0.6 | Default, good balance | Low |
| 0.7 | Strict matching (fewer results, higher confidence) | Very Low |
| 0.8+ | High security | Minimal |

Recommended: Start with 0.6 and adjust based on your use case.

---

## Two Modes of Operation

### Mode 1: Photo Gallery Search (Primary)
Use when you have a collection of photos and want to find specific people.

```
/gallery/index-bulk → Index all photos
/gallery/find-person → Find someone in photos
```

**Data stored per face**:
- `type: "gallery"`
- `image_id`: Unique identifier for the source image
- `image_name`: Original filename
- `face_index`: Which face in the image (0, 1, 2, ...)
- `bbox`: Bounding box location
- `embedding`: 512-dim vector

### Mode 2: Person Database (Secondary)
Use when you want to register known individuals and identify them later.

```
/register → Register known people (with person_id)
/search → Identify who is in a photo
```

**Data stored per face**:
- `type: "registered"` (or no type field)
- `person_id`: Identifier for the person
- `person_name`: Person's name
- `embedding`: 512-dim vector

Both modes can coexist in the same database, filtered by the `type` field.
