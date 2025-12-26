# Face Finder

A face detection, recognition, and search API using state-of-the-art models.

## Features

- **Face Detection**: SCRFD (Sample and Computation Redistribution for Face Detection)
- **Face Embeddings**: LVFace (Large Vision Face - ICCV 2025 Highlight)
- **Vector Search**: Qdrant for efficient similarity search
- **REST API**: FastAPI for high-performance API endpoints

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client        │────▶│   FastAPI        │────▶│   SCRFD         │
│   (HTTP)        │     │   Server         │     │   Detection     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               │                         ▼
                               │                 ┌─────────────────┐
                               │                 │   LVFace        │
                               │                 │   Embeddings    │
                               │                 └─────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌──────────────────────────────────────┐
                        │              Qdrant                   │
                        │         Vector Database               │
                        └──────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Download default models (SCRFD + LVFace-T)
python download_models.py

# Or download specific models
python download_models.py --models scrfd lvface-b

# List available models
python download_models.py --list
```

### 3. Start Qdrant

Using Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or install locally - see [Qdrant Installation Guide](https://qdrant.tech/documentation/guides/installation/)

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Start the API

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check for all components |
| `/stats` | GET | Database statistics |

### Face Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | Detect faces in an image |
| `/embed` | POST | Extract face embeddings from an image |
| `/register` | POST | Register a face in the database |
| `/search` | POST | Search for matching faces |
| `/verify` | POST | Verify if two images are the same person |

### Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/person/{person_id}` | DELETE | Delete all faces for a person |
| `/face/{face_id}` | DELETE | Delete a specific face |
| `/collection` | DELETE | Clear all faces (use with caution!) |

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Register a face
with open("person1.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/register",
        files={"file": f},
        data={"person_id": "john_doe", "person_name": "John Doe"}
    )
print(response.json())

# Search for a face
with open("query.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/search",
        files={"file": f},
        params={"limit": 5, "threshold": 0.6}
    )
print(response.json())

# Verify two faces
with open("image1.jpg", "rb") as f1, open("image2.jpg", "rb") as f2:
    response = requests.post(
        f"{BASE_URL}/verify",
        files={"file1": f1, "file2": f2},
        params={"threshold": 0.6}
    )
print(response.json())
```

### cURL Examples

```bash
# Detect faces
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -F "file=@image.jpg"

# Register a face
curl -X POST "http://localhost:8000/register" \
  -H "accept: application/json" \
  -F "file=@person.jpg" \
  -F "person_id=john_doe" \
  -F "person_name=John Doe"

# Search faces
curl -X POST "http://localhost:8000/search" \
  -H "accept: application/json" \
  -F "file=@query.jpg"
```

## Configuration

Environment variables (can be set in `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `SCRFD_MODEL_PATH` | `models/scrfd.onnx` | Path to SCRFD model |
| `DETECTION_THRESHOLD` | `0.5` | Face detection confidence threshold |
| `LVFACE_MODEL_PATH` | `models/lvface.onnx` | Path to LVFace model |
| `USE_GPU` | `true` | Use GPU for inference |
| `EMBEDDING_DIM` | `512` | Face embedding dimension |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | `face_embeddings` | Qdrant collection name |
| `SIMILARITY_THRESHOLD` | `0.6` | Minimum similarity for matching |

## Models

### SCRFD (Face Detection)

SCRFD is an efficient and accurate face detection model. We use the 10G variant with keypoint detection for face alignment.

- [SCRFD GitHub](https://github.com/cospectrum/scrfd)
- Model: `scrfd_10g_bnkps.onnx`

### LVFace (Face Recognition)

LVFace (ICCV 2025 Highlight) provides state-of-the-art face recognition using Vision Transformer architecture.

- [LVFace GitHub](https://github.com/bytedance/LVFace)
- Available variants:
  - **LVFace-T** (Tiny): Fastest, good accuracy
  - **LVFace-S** (Small): Balanced speed/accuracy
  - **LVFace-B** (Base): Best accuracy

### Qdrant (Vector Database)

Qdrant is a high-performance vector similarity search engine optimized for filtering and neural network matching.

- [Qdrant Documentation](https://qdrant.tech/documentation/)

## Project Structure

```
Face Finder/
├── main.py                 # FastAPI application
├── config.py               # Configuration settings
├── models.py               # Pydantic data models
├── requirements.txt        # Python dependencies
├── download_models.py      # Model download script
├── .env.example            # Environment template
├── README.md               # This file
├── models/                 # Model files (created by download_models.py)
│   ├── scrfd.onnx
│   └── lvface.onnx
└── services/
    ├── __init__.py
    ├── face_detection.py   # SCRFD face detection service
    ├── face_embedding.py   # LVFace embedding service
    └── qdrant_service.py   # Qdrant vector database service
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, for faster inference)
- Qdrant server (local or cloud)

## License

- Code: MIT License
- SCRFD Model: [InsightFace License](https://github.com/deepinsight/insightface)
- LVFace Model: MIT License (non-commercial research only)

## Acknowledgments

- [SCRFD](https://github.com/cospectrum/scrfd) - Face detection
- [LVFace](https://github.com/bytedance/LVFace) - Face recognition (ByteDance)
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
