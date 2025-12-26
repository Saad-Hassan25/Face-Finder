import { useEffect, useState } from 'react';
import { ImageIcon, Users, Trash2 } from 'lucide-react';
import { StatsCard, LoadingSpinner } from '../components';
import { galleryApi, systemApi, type GalleryStatsResponse } from '../services/api';

export function GalleryPage() {
  const [stats, setStats] = useState<GalleryStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);

  const fetchStats = async () => {
    try {
      const data = await galleryApi.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const handleClearGallery = async () => {
    if (!confirm('Are you sure you want to clear the entire gallery? This cannot be undone.')) {
      return;
    }

    setClearing(true);
    try {
      await systemApi.clearCollection();
      await fetchStats();
    } catch (error) {
      console.error('Failed to clear gallery:', error);
    } finally {
      setClearing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner message="Loading gallery stats..." />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Gallery</h1>
          <p className="text-gray-500 mt-1">
            View and manage your indexed photo collection.
          </p>
        </div>
        <button
          onClick={handleClearGallery}
          disabled={clearing || (stats?.total_images === 0)}
          className="px-4 py-2 text-sm font-medium text-red-700 bg-red-50 rounded-lg hover:bg-red-100 disabled:opacity-50 flex items-center gap-2"
        >
          {clearing ? (
            <LoadingSpinner size="sm" />
          ) : (
            <Trash2 className="w-4 h-4" />
          )}
          Clear Gallery
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <StatsCard
          title="Total Images"
          value={stats?.total_images || 0}
          icon={<ImageIcon className="w-6 h-6 text-blue-600" />}
          subtitle="Unique images indexed"
        />
        <StatsCard
          title="Total Faces"
          value={stats?.total_faces || 0}
          icon={<Users className="w-6 h-6 text-blue-600" />}
          subtitle="Faces detected and indexed"
        />
      </div>

      {/* Info Card */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">1️⃣</span>
            </div>
            <h3 className="font-medium text-gray-900">Upload Photos</h3>
            <p className="text-sm text-gray-500 mt-1">
              Upload your event or collection photos to the gallery
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">2️⃣</span>
            </div>
            <h3 className="font-medium text-gray-900">Faces Indexed</h3>
            <p className="text-sm text-gray-500 mt-1">
              AI automatically detects and indexes all faces
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl">3️⃣</span>
            </div>
            <h3 className="font-medium text-gray-900">Search & Find</h3>
            <p className="text-sm text-gray-500 mt-1">
              Upload a reference photo to find all images of that person
            </p>
          </div>
        </div>
      </div>

      {/* Empty State */}
      {stats?.total_images === 0 && (
        <div className="text-center py-12 bg-gray-50 rounded-xl border-2 border-dashed border-gray-300">
          <ImageIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900">No images yet</h3>
          <p className="text-gray-500 mt-1">
            Start by uploading some photos to your gallery
          </p>
          <a
            href="/upload"
            className="inline-block mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Upload Photos
          </a>
        </div>
      )}
    </div>
  );
}
