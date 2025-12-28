import { useState, useEffect } from 'react';
import { FolderOpen, Save, Trash2, RefreshCw, Plus, X, Upload, Images } from 'lucide-react';
import { LoadingSpinner } from '../components';
import { savedGalleryApi, galleryApi, type SavedGalleryInfo, type GalleryProgressEvent } from '../services/api';

interface ProgressState {
  active: boolean;
  type: 'save' | 'load' | null;
  message: string;
  progress: number;
  albumName: string;
}

export function AlbumsPage() {
  const [savedGalleries, setSavedGalleries] = useState<SavedGalleryInfo[]>([]);
  const [loadingGalleries, setLoadingGalleries] = useState(true);
  const [newGalleryName, setNewGalleryName] = useState('');
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [deletingGallery, setDeletingGallery] = useState<string | null>(null);
  
  // Progress state for save/load operations
  const [progressState, setProgressState] = useState<ProgressState>({
    active: false,
    type: null,
    message: '',
    progress: 0,
    albumName: ''
  });

  const fetchSavedGalleries = async () => {
    setLoadingGalleries(true);
    try {
      const response = await savedGalleryApi.list();
      if (response.success) {
        setSavedGalleries(response.galleries);
      }
    } catch (error) {
      console.error('Failed to fetch saved galleries:', error);
    } finally {
      setLoadingGalleries(false);
    }
  };

  useEffect(() => {
    fetchSavedGalleries();
  }, []);

  const handleSaveGallery = () => {
    if (!newGalleryName.trim()) {
      alert('Please enter a name for the album');
      return;
    }

    const albumName = newGalleryName.trim();
    setShowSaveModal(false);
    setNewGalleryName('');
    
    setProgressState({
      active: true,
      type: 'save',
      message: 'Starting save...',
      progress: 0,
      albumName
    });

    savedGalleryApi.saveWithProgress(
      albumName,
      (event: GalleryProgressEvent) => {
        setProgressState(prev => ({
          ...prev,
          message: event.message,
          progress: event.progress
        }));
      },
      (event: GalleryProgressEvent) => {
        setProgressState({
          active: false,
          type: null,
          message: '',
          progress: 0,
          albumName: ''
        });
        alert(`Album saved! ${event.faces_saved} faces from ${event.unique_images} images.`);
        fetchSavedGalleries();
      },
      (error: string) => {
        setProgressState({
          active: false,
          type: null,
          message: '',
          progress: 0,
          albumName: ''
        });
        alert(`Failed to save: ${error}`);
      }
    );
  };

  const handleLoadGallery = (name: string) => {
    if (!confirm(`Load album "${name}"? This will replace the current gallery.`)) {
      return;
    }

    setProgressState({
      active: true,
      type: 'load',
      message: 'Starting load...',
      progress: 0,
      albumName: name
    });

    savedGalleryApi.loadWithProgress(
      name,
      (event: GalleryProgressEvent) => {
        setProgressState(prev => ({
          ...prev,
          message: event.message,
          progress: event.progress
        }));
      },
      (event: GalleryProgressEvent) => {
        setProgressState({
          active: false,
          type: null,
          message: '',
          progress: 0,
          albumName: ''
        });
        alert(`Album loaded! ${event.faces_loaded} faces restored.`);
      },
      (error: string) => {
        setProgressState({
          active: false,
          type: null,
          message: '',
          progress: 0,
          albumName: ''
        });
        alert(`Failed to load: ${error}`);
      }
    );
  };

  const handleDeleteSavedGallery = async (name: string) => {
    if (!confirm(`Delete saved album "${name}"? This cannot be undone.`)) {
      return;
    }

    setDeletingGallery(name);
    try {
      const result = await savedGalleryApi.delete(name);
      if (result.success) {
        fetchSavedGalleries();
      } else {
        alert(`Failed to delete: ${result.message}`);
      }
    } catch (error) {
      console.error('Failed to delete saved album:', error);
      alert('Failed to delete saved album');
    } finally {
      setDeletingGallery(null);
    }
  };

  const handleClearGallery = async () => {
    if (!confirm('Clear the current gallery? Saved albums will not be affected.')) {
      return;
    }

    try {
      const result = await galleryApi.clear();
      if (result.success) {
        alert(`Gallery cleared! ${result.deleted_count} faces removed.`);
      }
    } catch (error) {
      console.error('Failed to clear gallery:', error);
      alert('Failed to clear gallery');
    }
  };

  return (
    <div className="space-y-6">
      {/* Progress Overlay */}
      {progressState.active && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-8 max-w-md w-full mx-4">
            <div className="text-center mb-6">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 rounded-full flex items-center justify-center">
                {progressState.type === 'save' ? (
                  <Save className="w-8 h-8 text-blue-600 animate-pulse" />
                ) : (
                  <Upload className="w-8 h-8 text-green-600 animate-pulse" />
                )}
              </div>
              <h3 className="text-lg font-semibold text-gray-900">
                {progressState.type === 'save' ? 'Saving Album' : 'Loading Album'}
              </h3>
              <p className="text-sm text-gray-500 mt-1">{progressState.albumName}</p>
            </div>
            
            {/* Progress Bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>{progressState.message}</span>
                <span>{progressState.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    progressState.type === 'save' ? 'bg-blue-600' : 'bg-green-600'
                  }`}
                  style={{ width: `${progressState.progress}%` }}
                />
              </div>
            </div>
            
            <p className="text-xs text-gray-400 text-center">
              Please wait, this may take a moment...
            </p>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Albums</h1>
          <p className="text-gray-500 mt-1">
            Save and manage your indexed photo collections.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowSaveModal(true)}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Save Current Gallery
          </button>
          <button
            onClick={fetchSavedGalleries}
            disabled={loadingGalleries}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${loadingGalleries ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Save Modal */}
      {showSaveModal && (
        <div className="bg-white rounded-xl shadow-sm border border-blue-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Save className="w-5 h-5 text-blue-600" />
            Save Current Gallery as Album
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            This will save all currently indexed faces so you can load them later without reprocessing.
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={newGalleryName}
              onChange={(e) => setNewGalleryName(e.target.value)}
              placeholder="Enter album name (e.g., Wedding 2024, Birthday Party)"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              onKeyDown={(e) => e.key === 'Enter' && handleSaveGallery()}
              autoFocus
            />
            <button
              onClick={handleSaveGallery}
              disabled={!newGalleryName.trim() || progressState.active}
              className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Save Album
            </button>
            <button
              onClick={() => { setShowSaveModal(false); setNewGalleryName(''); }}
              className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Info Card */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-100">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-white rounded-lg shadow-sm">
            <Images className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">How Albums Work</h3>
            <ul className="mt-2 text-sm text-gray-600 space-y-1">
              <li>• <strong>Save:</strong> Creates a snapshot of your current indexed gallery</li>
              <li>• <strong>Load:</strong> Restores a saved album (replaces current gallery)</li>
              <li>• <strong>Switch:</strong> Easily switch between different events or collections</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Saved Albums List */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <FolderOpen className="w-5 h-5" />
          Saved Albums
        </h2>

        {loadingGalleries ? (
          <div className="flex justify-center py-12">
            <LoadingSpinner message="Loading saved albums..." />
          </div>
        ) : savedGalleries.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
              <FolderOpen className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="font-medium text-gray-900 mb-1">No saved albums yet</h3>
            <p className="text-sm text-gray-500 mb-4">
              Index some images and save them as an album to get started.
            </p>
            <button
              onClick={() => setShowSaveModal(true)}
              className="px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 inline-flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Save Current Gallery
            </button>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {savedGalleries.map((gallery) => (
              <div
                key={gallery.name}
                className="bg-gray-50 rounded-xl p-5 hover:bg-gray-100 transition-colors border border-gray-200"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="p-2 bg-white rounded-lg shadow-sm">
                    <FolderOpen className="w-5 h-5 text-blue-600" />
                  </div>
                  <button
                    onClick={() => handleDeleteSavedGallery(gallery.name)}
                    disabled={deletingGallery === gallery.name}
                    className="p-1.5 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    title="Delete album"
                  >
                    {deletingGallery === gallery.name ? (
                      <LoadingSpinner size="sm" />
                    ) : (
                      <Trash2 className="w-4 h-4" />
                    )}
                  </button>
                </div>
                
                <h3 className="font-semibold text-gray-900 mb-1">{gallery.name}</h3>
                <p className="text-sm text-gray-500 mb-4">
                  {gallery.total_faces} faces • {gallery.unique_images} images
                  {gallery.created_at && (
                    <>
                      <br />
                      Saved {new Date(gallery.created_at).toLocaleDateString()}
                    </>
                  )}
                </p>
                
                <button
                  onClick={() => handleLoadGallery(gallery.name)}
                  disabled={progressState.active}
                  className="w-full px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  Load Album
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Clear Current Gallery */}
      <div className="bg-white rounded-xl shadow-sm border border-orange-200 p-6">
        <h2 className="text-lg font-semibold text-orange-700 mb-2 flex items-center gap-2">
          <Trash2 className="w-5 h-5" />
          Clear Current Gallery
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Remove all indexed images from the current gallery. Your saved albums will not be affected.
        </p>
        <button
          onClick={handleClearGallery}
          className="px-4 py-2 text-sm font-medium text-orange-700 bg-orange-50 rounded-lg hover:bg-orange-100 flex items-center gap-2"
        >
          <Trash2 className="w-4 h-4" />
          Clear Gallery
        </button>
      </div>
    </div>
  );
}
