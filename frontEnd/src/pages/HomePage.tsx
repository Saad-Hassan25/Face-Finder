import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Upload, Search, Users, ImageIcon, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { StatsCard, LoadingSpinner } from '../components';
import { galleryApi, systemApi, type GalleryStatsResponse, type HealthResponse } from '../services/api';

export function HomePage() {
  const [stats, setStats] = useState<GalleryStatsResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, healthData] = await Promise.all([
          galleryApi.getStats(),
          systemApi.health(),
        ]);
        setStats(statsData);
        setHealth(healthData);
      } catch (error) {
        console.error('Failed to fetch data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'unhealthy':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner message="Loading dashboard..." />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">Welcome to Face Finder</h1>
        <p className="text-blue-100 text-lg max-w-2xl">
          Upload your photos and find yourself or anyone in your photo collection.
          Powered by SCRFD and LVFace AI models.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          title="Total Images"
          value={stats?.total_images || 0}
          icon={<ImageIcon className="w-6 h-6 text-blue-600" />}
          subtitle="Images in gallery"
        />
        <StatsCard
          title="Total Faces"
          value={stats?.total_faces || 0}
          icon={<Users className="w-6 h-6 text-blue-600" />}
          subtitle="Indexed faces"
        />
        <StatsCard
          title="System Status"
          value={health?.status === 'healthy' ? 'Online' : 'Issues'}
          icon={getStatusIcon(health?.status || 'unknown')}
          subtitle={health?.status || 'Unknown'}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link
          to="/upload"
          className="group bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-blue-300 transition-all"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-50 rounded-xl group-hover:bg-blue-100 transition-colors">
              <Upload className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Upload Photos</h3>
              <p className="text-sm text-gray-500">Add images to your gallery</p>
            </div>
          </div>
        </Link>

        <Link
          to="/search"
          className="group bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-green-300 transition-all"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-green-50 rounded-xl group-hover:bg-green-100 transition-colors">
              <Search className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Find Person</h3>
              <p className="text-sm text-gray-500">Search for someone in photos</p>
            </div>
          </div>
        </Link>

        <Link
          to="/verify"
          className="group bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md hover:border-purple-300 transition-all"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-50 rounded-xl group-hover:bg-purple-100 transition-colors">
              <Users className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Verify Faces</h3>
              <p className="text-sm text-gray-500">Compare two photos</p>
            </div>
          </div>
        </Link>
      </div>

      {/* System Health */}
      {health && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">System Components</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(health.components).map(([name, component]) => (
              <div key={name} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                {getStatusIcon(component.status)}
                <div>
                  <p className="font-medium text-gray-900 capitalize">
                    {name.replace('_', ' ')}
                  </p>
                  <p className="text-xs text-gray-500">{component.message}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
