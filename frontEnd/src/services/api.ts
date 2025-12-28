import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for bulk uploads
});

// Types
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedFace {
  bbox: BoundingBox;
  confidence: number;
  face_index: number;
  keypoints?: {
    left_eye: { x: number; y: number };
    right_eye: { x: number; y: number };
    nose: { x: number; y: number };
    left_mouth: { x: number; y: number };
    right_mouth: { x: number; y: number };
  };
}

export interface IndexedFace {
  face_id: string;
  image_id: string;
  image_name?: string;
  face_index: number;
  bbox: BoundingBox;
}

export interface ImageMatch {
  image_id: string;
  image_name?: string;
  file_path?: string;
  similarity: number;
  face_id: string;
  face_bbox?: BoundingBox;
  metadata: Record<string, unknown>;
}

export interface ImageIndexResponse {
  success: boolean;
  message: string;
  image_id: string;
  faces_indexed: number;
  indexed_faces: IndexedFace[];
  processing_time_ms: number;
}

export interface BulkIndexResponse {
  success: boolean;
  message: string;
  total_images: number;
  total_faces_indexed: number;
  images_processed: Array<{
    image_id: string;
    image_name: string;
    faces_indexed: number;
    status: string;
  }>;
  failed_images: Array<{
    image_name: string;
    error: string;
  }>;
  processing_time_ms: number;
}

export interface FindPersonResponse {
  success: boolean;
  message: string;
  query_face_count: number;
  total_images_found: number;
  images: ImageMatch[];
  processing_time_ms: number;
}

export interface GalleryStatsResponse {
  success: boolean;
  total_images: number;
  total_faces: number;
  unique_images: number;
  message: string;
}

export interface FolderIndexRequest {
  folder_path: string;
  recursive?: boolean;
  extensions?: string[];
}

export interface FolderIndexResponse {
  success: boolean;
  message: string;
  folder_path: string;
  total_images: number;
  total_faces_indexed: number;
  images_processed: Array<{
    image_id?: string;
    image_name: string;
    file_path: string;
    faces_indexed: number;
    status: string;
  }>;
  failed_images: Array<{
    image_name: string;
    file_path: string;
    error: string;
  }>;
  processing_time_ms: number;
}

export interface FolderIndexProgress {
  type: 'start' | 'progress' | 'complete' | 'error';
  current?: number;
  total?: number;
  percent?: number;
  current_image?: string;
  faces_found?: number;
  total_faces_so_far?: number;
  total_images?: number;
  message?: string;
  error?: string;
}

export interface VerifyResponse {
  success: boolean;
  is_same_person?: boolean;
  similarity?: number;
  threshold?: number;
  faces_in_image1?: number;
  faces_in_image2?: number;
  processing_time_ms: number;
  message?: string;
}

export interface DetectionResponse {
  success: boolean;
  message: string;
  faces: DetectedFace[];
  total_faces: number;
  processing_time_ms: number;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  components: Record<string, {
    status: 'healthy' | 'unhealthy' | 'degraded';
    message: string;
    latency_ms?: number;
  }>;
}

// API Functions

export const galleryApi = {
  // Index a single image
  indexImage: async (file: File, imageName?: string): Promise<ImageIndexResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (imageName) {
      formData.append('image_name', imageName);
    }
    const response = await api.post<ImageIndexResponse>('/gallery/index', formData);
    return response.data;
  },

  // Index multiple images (extended timeout for large batches)
  indexBulk: async (
    files: File[],
    onProgress?: (progress: number) => void
  ): Promise<BulkIndexResponse> => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    const response = await api.post<BulkIndexResponse>('/gallery/index-bulk', formData, {
      timeout: 600000, // 10 minutes for bulk uploads
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
    return response.data;
  },

  // Index a folder of images with real-time progress via SSE
  indexFolder: async (
    request: FolderIndexRequest,
    onProgress?: (data: FolderIndexProgress) => void
  ): Promise<FolderIndexResponse> => {
    return new Promise((resolve, reject) => {
      fetch(`${API_BASE_URL}/gallery/index-folder-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('No response body');
          }
          
          const decoder = new TextDecoder();
          let buffer = '';
          
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  
                  if (data.type === 'progress' && onProgress) {
                    onProgress(data);
                  } else if (data.type === 'start' && onProgress) {
                    onProgress({ ...data, current: 0, percent: 0 });
                  } else if (data.type === 'complete') {
                    resolve(data as FolderIndexResponse);
                    return;
                  } else if (data.type === 'error') {
                    reject(new Error(data.message));
                    return;
                  }
                } catch (e) {
                  console.error('Failed to parse SSE data:', e);
                }
              }
            }
          }
        })
        .catch(reject);
    });
  },

  // Index a folder without progress (fallback)
  indexFolderSimple: async (request: FolderIndexRequest): Promise<FolderIndexResponse> => {
    const response = await api.post<FolderIndexResponse>('/gallery/index-folder', request, {
      timeout: 3000000, // 30 minutes for large folder processing
    });
    return response.data;
  },

  // Find person in gallery
  findPerson: async (
    file: File,
    limit?: number,
    threshold?: number
  ): Promise<FindPersonResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (threshold) params.append('threshold', threshold.toString());
    
    const response = await api.post<FindPersonResponse>(
      `/gallery/find-person?${params.toString()}`,
      formData
    );
    return response.data;
  },

  // Get gallery stats
  getStats: async (): Promise<GalleryStatsResponse> => {
    const response = await api.get<GalleryStatsResponse>('/gallery/stats');
    return response.data;
  },

  // Delete image from gallery
  deleteImage: async (imageId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/gallery/image/${imageId}`);
    return response.data;
  },

  // Clear gallery
  clear: async (): Promise<{ success: boolean; message: string; deleted_count: number }> => {
    const response = await api.delete('/gallery/clear');
    return response.data;
  },
};

// ================== Saved Galleries API ==================

export interface SavedGalleryInfo {
  name: string;
  created_at: string;
  total_faces: number;
  unique_images: number;
}

export interface SaveGalleryResponse {
  success: boolean;
  message: string;
  name: string;
  faces_saved: number;
  unique_images: number;
}

export interface LoadGalleryResponse {
  success: boolean;
  message: string;
  name: string;
  faces_loaded: number;
  previous_gallery_cleared: boolean;
}

export interface ListSavedGalleriesResponse {
  success: boolean;
  message: string;
  galleries: SavedGalleryInfo[];
}

export interface GalleryProgressEvent {
  status: 'starting' | 'progress' | 'complete' | 'error';
  message: string;
  progress: number;
  total?: number;
  copied?: number;
  loaded?: number;
  name?: string;
  faces_saved?: number;
  faces_loaded?: number;
  unique_images?: number;
  previous_gallery_cleared?: boolean;
}

export const savedGalleryApi = {
  // Save current gallery
  save: async (name: string): Promise<SaveGalleryResponse> => {
    const response = await api.post<SaveGalleryResponse>('/gallery/save', { name });
    return response.data;
  },

  // Save current gallery with progress streaming
  saveWithProgress: (
    name: string,
    onProgress: (event: GalleryProgressEvent) => void,
    onComplete: (event: GalleryProgressEvent) => void,
    onError: (error: string) => void
  ): (() => void) => {
    const controller = new AbortController();
    
    fetch(`${API_BASE_URL}/gallery/save/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }
        
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event: GalleryProgressEvent = JSON.parse(line.slice(6));
                if (event.status === 'complete') {
                  onComplete(event);
                } else if (event.status === 'error') {
                  onError(event.message);
                } else {
                  onProgress(event);
                }
              } catch (e) {
                console.error('Failed to parse SSE event:', e);
              }
            }
          }
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          onError(error.message || 'Failed to save album');
        }
      });
    
    return () => controller.abort();
  },

  // Load a saved gallery
  load: async (name: string, clearCurrent: boolean = true): Promise<LoadGalleryResponse> => {
    const response = await api.post<LoadGalleryResponse>(
      `/gallery/load/${encodeURIComponent(name)}?clear_current=${clearCurrent}`
    );
    return response.data;
  },

  // Load a saved gallery with progress streaming
  loadWithProgress: (
    name: string,
    onProgress: (event: GalleryProgressEvent) => void,
    onComplete: (event: GalleryProgressEvent) => void,
    onError: (error: string) => void,
    clearCurrent: boolean = true
  ): (() => void) => {
    const controller = new AbortController();
    
    fetch(`${API_BASE_URL}/gallery/load/${encodeURIComponent(name)}/stream?clear_current=${clearCurrent}`, {
      method: 'POST',
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }
        
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event: GalleryProgressEvent = JSON.parse(line.slice(6));
                if (event.status === 'complete') {
                  onComplete(event);
                } else if (event.status === 'error') {
                  onError(event.message);
                } else {
                  onProgress(event);
                }
              } catch (e) {
                console.error('Failed to parse SSE event:', e);
              }
            }
          }
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          onError(error.message || 'Failed to load album');
        }
      });
    
    return () => controller.abort();
  },

  // List all saved galleries
  list: async (): Promise<ListSavedGalleriesResponse> => {
    const response = await api.get<ListSavedGalleriesResponse>('/gallery/saved');
    return response.data;
  },

  // Delete a saved gallery
  delete: async (name: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete(`/gallery/saved/${encodeURIComponent(name)}`);
    return response.data;
  },
};

// Helper function to get image URL from file path
export const getImageUrl = (filePath: string): string => {
  return `${API_BASE_URL}/images?path=${encodeURIComponent(filePath)}`;
};

export const faceApi = {
  // Detect faces in image
  detect: async (file: File): Promise<DetectionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<DetectionResponse>('/detect', formData);
    return response.data;
  },

  // Verify two faces
  verify: async (
    file1: File,
    file2: File,
    threshold?: number
  ): Promise<VerifyResponse> => {
    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);
    
    const params = new URLSearchParams();
    if (threshold) params.append('threshold', threshold.toString());
    
    const response = await api.post<VerifyResponse>(
      `/verify?${params.toString()}`,
      formData
    );
    return response.data;
  },
};

export const systemApi = {
  // Health check
  health: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },

  // Clear collection
  clearCollection: async (): Promise<{ success: boolean; message: string }> => {
    const response = await api.delete('/collection');
    return response.data;
  },
};

export default api;
