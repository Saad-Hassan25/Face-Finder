import { useState, useCallback } from 'react';
import { Search, ImageIcon, AlertCircle } from 'lucide-react';
import { Dropzone, ResultsGrid, LoadingSpinner } from '../components';
import { galleryApi, type FindPersonResponse, type ImageMatch } from '../services/api';

export function SearchPage() {
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<FindPersonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.6);
  const [limit, setLimit] = useState(50);

  const handleFileAccepted = useCallback((files: File[]) => {
    if (files.length > 0) {
      const file = files[0];
      setReferenceImage(file);
      setReferencePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const handleSearch = async () => {
    if (!referenceImage) return;

    setSearching(true);
    setError(null);

    try {
      const response = await galleryApi.findPerson(referenceImage, limit, threshold);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleClear = () => {
    setReferenceImage(null);
    if (referencePreview) {
      URL.revokeObjectURL(referencePreview);
    }
    setReferencePreview(null);
    setResult(null);
    setError(null);
  };

  const handleImageClick = (image: ImageMatch) => {
    // Could open a modal or navigate to image details
    console.log('Clicked image:', image);
    alert(`Image: ${image.image_name}\nSimilarity: ${(image.similarity * 100).toFixed(1)}%`);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Find Person</h1>
        <p className="text-gray-500 mt-1">
          Upload a reference photo to find all images where that person appears.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel - Reference Image */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Reference Photo</h2>
            
            {!referenceImage ? (
              <Dropzone
                onFilesAccepted={handleFileAccepted}
                multiple={false}
                className="min-h-[200px]"
              >
                <div className="flex flex-col items-center gap-3">
                  <div className="p-4 bg-blue-50 rounded-full">
                    <Search className="w-8 h-8 text-blue-500" />
                  </div>
                  <p className="text-sm text-gray-600">
                    Drop a photo of the person to find
                  </p>
                </div>
              </Dropzone>
            ) : (
              <div className="space-y-4">
                <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                  <img
                    src={referencePreview!}
                    alt="Reference"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleClear}
                    disabled={searching}
                    className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
                  >
                    Change Photo
                  </button>
                  <button
                    onClick={handleSearch}
                    disabled={searching}
                    className="flex-1 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {searching ? (
                      <LoadingSpinner size="sm" />
                    ) : (
                      <Search className="w-4 h-4" />
                    )}
                    {searching ? 'Searching...' : 'Search'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Search Settings */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Settings</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Similarity Threshold: {(threshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.3"
                  max="0.9"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>More results</span>
                  <span>Higher accuracy</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Results: {limit}
                </label>
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="10"
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 min-h-[500px]">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">
                {result ? `Found ${result.total_images_found} Images` : 'Search Results'}
              </h2>
              {result && (
                <span className="text-sm text-gray-500">
                  {(result.processing_time_ms / 1000).toFixed(2)}s
                </span>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                <div>
                  <h3 className="font-medium text-red-800">Search Failed</h3>
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              </div>
            )}

            {/* Loading */}
            {searching && (
              <div className="flex items-center justify-center h-64">
                <LoadingSpinner message="Searching for matching faces..." />
              </div>
            )}

            {/* Empty State */}
            {!searching && !result && !error && (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <ImageIcon className="w-12 h-12 mb-4 text-gray-300" />
                <p>Upload a reference photo to start searching</p>
              </div>
            )}

            {/* Results */}
            {result && !searching && (
              <ResultsGrid
                images={result.images}
                onImageClick={handleImageClick}
                emptyMessage="No matching images found. Try lowering the threshold."
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
