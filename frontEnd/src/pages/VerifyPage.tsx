import { useState, useCallback } from 'react';
import { Users, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { Dropzone, LoadingSpinner } from '../components';
import { faceApi, type VerifyResponse } from '../services/api';

export function VerifyPage() {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [preview1, setPreview1] = useState<string | null>(null);
  const [preview2, setPreview2] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.6);

  const handleImage1Accepted = useCallback((files: File[]) => {
    if (files.length > 0) {
      const file = files[0];
      setImage1(file);
      if (preview1) URL.revokeObjectURL(preview1);
      setPreview1(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, [preview1]);

  const handleImage2Accepted = useCallback((files: File[]) => {
    if (files.length > 0) {
      const file = files[0];
      setImage2(file);
      if (preview2) URL.revokeObjectURL(preview2);
      setPreview2(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, [preview2]);

  const handleVerify = async () => {
    if (!image1 || !image2) return;

    setVerifying(true);
    setError(null);
    setResult(null);

    try {
      const response = await faceApi.verify(image1, image2, threshold);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Verification failed');
    } finally {
      setVerifying(false);
    }
  };

  const handleClear = () => {
    setImage1(null);
    setImage2(null);
    if (preview1) URL.revokeObjectURL(preview1);
    if (preview2) URL.revokeObjectURL(preview2);
    setPreview1(null);
    setPreview2(null);
    setResult(null);
    setError(null);
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-green-600';
    if (similarity >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Verify Faces</h1>
        <p className="text-gray-500 mt-1">
          Compare two photos to check if they contain the same person.
        </p>
      </div>

      {/* Image Upload Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Image 1 */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">First Photo</h2>
          
          {!image1 ? (
            <Dropzone
              onFilesAccepted={handleImage1Accepted}
              multiple={false}
              className="min-h-[250px]"
            />
          ) : (
            <div className="space-y-4">
              <div className="aspect-square rounded-lg overflow-hidden bg-gray-100">
                <img
                  src={preview1!}
                  alt="First"
                  className="w-full h-full object-cover"
                />
              </div>
              <button
                onClick={() => {
                  setImage1(null);
                  if (preview1) URL.revokeObjectURL(preview1);
                  setPreview1(null);
                  setResult(null);
                }}
                className="w-full px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Change Photo
              </button>
            </div>
          )}
        </div>

        {/* Image 2 */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Second Photo</h2>
          
          {!image2 ? (
            <Dropzone
              onFilesAccepted={handleImage2Accepted}
              multiple={false}
              className="min-h-[250px]"
            />
          ) : (
            <div className="space-y-4">
              <div className="aspect-square rounded-lg overflow-hidden bg-gray-100">
                <img
                  src={preview2!}
                  alt="Second"
                  className="w-full h-full object-cover"
                />
              </div>
              <button
                onClick={() => {
                  setImage2(null);
                  if (preview2) URL.revokeObjectURL(preview2);
                  setPreview2(null);
                  setResult(null);
                }}
                className="w-full px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Change Photo
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Threshold Setting */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="max-w-md">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Verification Threshold: {(threshold * 100).toFixed(0)}%
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
            <span>Lenient</span>
            <span>Strict</span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={handleClear}
          disabled={verifying || (!image1 && !image2)}
          className="px-6 py-3 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
        >
          Clear All
        </button>
        <button
          onClick={handleVerify}
          disabled={verifying || !image1 || !image2}
          className="flex-1 px-6 py-3 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {verifying ? (
            <>
              <LoadingSpinner size="sm" />
              Comparing Faces...
            </>
          ) : (
            <>
              <Users className="w-5 h-5" />
              Compare Faces
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
          <div>
            <h3 className="font-medium text-red-800">Verification Failed</h3>
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </div>
      )}

      {/* Result */}
      {result && result.success && (
        <div
          className={`rounded-xl p-8 text-center ${
            result.is_same_person
              ? 'bg-green-50 border border-green-200'
              : 'bg-red-50 border border-red-200'
          }`}
        >
          <div className="flex justify-center mb-4">
            {result.is_same_person ? (
              <div className="p-4 bg-green-100 rounded-full">
                <CheckCircle className="w-12 h-12 text-green-600" />
              </div>
            ) : (
              <div className="p-4 bg-red-100 rounded-full">
                <XCircle className="w-12 h-12 text-red-600" />
              </div>
            )}
          </div>
          
          <h3 className={`text-2xl font-bold ${result.is_same_person ? 'text-green-800' : 'text-red-800'}`}>
            {result.is_same_person ? 'Same Person!' : 'Different People'}
          </h3>
          
          <div className="mt-4 space-y-2">
            <p className={`text-4xl font-bold ${getSimilarityColor(result.similarity || 0)}`}>
              {((result.similarity || 0) * 100).toFixed(1)}%
            </p>
            <p className="text-gray-600">Similarity Score</p>
          </div>

          <div className="mt-6 flex justify-center gap-8 text-sm text-gray-600">
            <div>
              <span className="font-medium">{result.faces_in_image1}</span> face(s) in first image
            </div>
            <div>
              <span className="font-medium">{result.faces_in_image2}</span> face(s) in second image
            </div>
          </div>

          <p className="mt-4 text-xs text-gray-500">
            Threshold: {((result.threshold || threshold) * 100).toFixed(0)}% • 
            Processing time: {(result.processing_time_ms / 1000).toFixed(2)}s
          </p>
        </div>
      )}

      {result && !result.success && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0" />
          <div>
            <h3 className="font-medium text-yellow-800">Verification Issue</h3>
            <p className="text-sm text-yellow-600">{result.message}</p>
          </div>
        </div>
      )}
    </div>
  );
}
