import { useState, useEffect } from 'react';
import { Settings, Trash2, RefreshCw, CheckCircle, XCircle } from 'lucide-react';
import { LoadingSpinner } from '../components';
import { systemApi, type HealthResponse } from '../services/api';

export function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const data = await systemApi.health();
      setHealth(data);
    } catch (error) {
      console.error('Failed to fetch health:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  const handleClearCollection = async () => {
    if (!confirm('Are you sure you want to delete ALL data? This cannot be undone.')) {
      return;
    }

    setClearing(true);
    try {
      await systemApi.clearCollection();
      alert('Collection cleared successfully!');
    } catch (error) {
      console.error('Failed to clear collection:', error);
      alert('Failed to clear collection');
    } finally {
      setClearing(false);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'healthy':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-full">
            <CheckCircle className="w-3 h-3" />
            Healthy
          </span>
        );
      case 'unhealthy':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded-full">
            <XCircle className="w-3 h-3" />
            Unhealthy
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-yellow-100 text-yellow-700 text-xs font-medium rounded-full">
            Degraded
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-500 mt-1">
            System configuration and health status.
          </p>
        </div>
        <button
          onClick={fetchHealth}
          disabled={loading}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* System Health */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          System Health
        </h2>

        {loading ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner message="Checking system health..." />
          </div>
        ) : health ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <span className="font-medium text-gray-900">Overall Status</span>
              {getStatusBadge(health.status)}
            </div>

            <div className="divide-y divide-gray-200">
              {Object.entries(health.components).map(([name, component]) => (
                <div key={name} className="py-4 flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900 capitalize">
                      {name.replace('_', ' ')}
                    </p>
                    <p className="text-sm text-gray-500">{component.message}</p>
                    {component.latency_ms && (
                      <p className="text-xs text-gray-400">
                        Latency: {component.latency_ms.toFixed(2)}ms
                      </p>
                    )}
                  </div>
                  {getStatusBadge(component.status)}
                </div>
              ))}
            </div>

            <p className="text-xs text-gray-400 text-right">
              Last checked: {new Date(health.timestamp).toLocaleString()}
            </p>
          </div>
        ) : (
          <p className="text-gray-500">Failed to load health status</p>
        )}
      </div>

      {/* API Configuration */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">API Configuration</h2>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Backend URL</p>
              <p className="font-mono text-gray-900">http://localhost:8000</p>
            </div>
            <div>
              <p className="text-gray-500">API Docs</p>
              <a 
                href="http://localhost:8000/docs" 
                target="_blank" 
                rel="noopener noreferrer"
                className="font-mono text-blue-600 hover:underline"
              >
                http://localhost:8000/docs
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="bg-white rounded-xl shadow-sm border border-red-200 p-6">
        <h2 className="text-lg font-semibold text-red-600 mb-4 flex items-center gap-2">
          <Trash2 className="w-5 h-5" />
          Danger Zone
        </h2>
        <div className="space-y-4">
          <div className="p-4 bg-red-50 rounded-lg">
            <h3 className="font-medium text-red-800">Clear All Data</h3>
            <p className="text-sm text-red-600 mt-1">
              This will permanently delete all indexed images and face data from the database.
              This action cannot be undone.
            </p>
            <button
              onClick={handleClearCollection}
              disabled={clearing}
              className="mt-4 px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50 flex items-center gap-2"
            >
              {clearing ? (
                <>
                  <LoadingSpinner size="sm" />
                  Clearing...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4" />
                  Clear All Data
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* About */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">About</h2>
        <div className="space-y-2 text-sm text-gray-600">
          <p><strong>Face Finder</strong> - Find yourself in photos</p>
          <p>
            Powered by:
          </p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>SCRFD - Face Detection</li>
            <li>LVFace - Face Embeddings</li>
            <li>Qdrant - Vector Database</li>
            <li>FastAPI - Backend API</li>
            <li>React + Vite - Frontend</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
