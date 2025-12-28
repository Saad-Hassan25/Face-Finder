import { useState, useEffect } from 'react';
import { Upload, CheckCircle, AlertCircle, FolderOpen, Cloud, LogOut, RefreshCw } from 'lucide-react';
import { Dropzone, ImagePreviewGrid, LoadingSpinner, ProgressBar, DriveFolderBrowser } from '../components';
import { galleryApi, googleDriveApi, type BulkIndexResponse, type FolderIndexResponse, type FolderIndexProgress, type DriveIndexProgress, type DriveStatusResponse } from '../services/api';

type UploadMode = 'files' | 'folder' | 'drive';

interface FolderProgress {
  current: number;
  total: number;
  percent: number;
  currentImage: string;
  facesFound: number;
  totalFacesSoFar: number;
}

export function UploadPage() {
  const [mode, setMode] = useState<UploadMode>('files');
  const [files, setFiles] = useState<File[]>([]);
  const [folderPath, setFolderPath] = useState('');
  const [recursive, setRecursive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [folderProgress, setFolderProgress] = useState<FolderProgress | null>(null);
  const [result, setResult] = useState<BulkIndexResponse | FolderIndexResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Google Drive state
  const [driveStatus, setDriveStatus] = useState<DriveStatusResponse | null>(null);
  const [driveLoading, setDriveLoading] = useState(false);
  const [showDriveBrowser, setShowDriveBrowser] = useState(false);
  const [selectedDriveFolder, setSelectedDriveFolder] = useState<{ id: string; name: string } | null>(null);
  const [driveRecursive, setDriveRecursive] = useState(true);

  // Check Drive connection status on mount
  useEffect(() => {
    checkDriveStatus();
  }, []);

  const checkDriveStatus = async () => {
    try {
      const status = await googleDriveApi.getStatus();
      setDriveStatus(status);
    } catch (err) {
      console.error('Failed to check Drive status:', err);
      setDriveStatus({ connected: false, user: null });
    }
  };

  const handleConnectDrive = async () => {
    setDriveLoading(true);
    try {
      const redirectUri = `${window.location.origin}/drive/callback`;
      const { auth_url } = await googleDriveApi.getAuthUrl(redirectUri);
      // Store current location for return
      sessionStorage.setItem('driveAuthReturn', window.location.pathname);
      // Redirect to Google OAuth
      window.location.href = auth_url;
    } catch (err) {
      setError('Failed to connect to Google Drive');
      setDriveLoading(false);
    }
  };

  const handleDisconnectDrive = async () => {
    setDriveLoading(true);
    try {
      await googleDriveApi.logout();
      setDriveStatus({ connected: false, user: null });
      setSelectedDriveFolder(null);
    } catch (err) {
      console.error('Failed to disconnect Drive:', err);
    } finally {
      setDriveLoading(false);
    }
  };

  const handleDriveFolderSelect = (folderId: string, folderName: string) => {
    setSelectedDriveFolder({ id: folderId, name: folderName });
  };

  const handleDriveProgress = (data: DriveIndexProgress) => {
    setFolderProgress({
      current: data.current || 0,
      total: data.total || data.total_images || 0,
      percent: data.percent || 0,
      currentImage: data.current_file || data.current_image || '',
      facesFound: data.faces_found || 0,
      totalFacesSoFar: data.total_faces_so_far || 0,
    });
  };

  const handleDriveIndex = async () => {
    if (!selectedDriveFolder) return;

    setUploading(true);
    setProgress(0);
    setFolderProgress(null);
    setError(null);
    setResult(null);

    try {
      const response = await googleDriveApi.indexFolder(
        selectedDriveFolder.id,
        handleDriveProgress,
        driveRecursive
      );
      // Set result with the folder name
      setResult({
        ...response,
        folder_path: `Google Drive: ${selectedDriveFolder.name}`,
      });
      setSelectedDriveFolder(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Drive indexing failed');
    } finally {
      setUploading(false);
      setProgress(0);
      setFolderProgress(null);
    }
  };

  const handleFilesAccepted = (acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
    setResult(null);
    setError(null);
  };

  const handleRemoveFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleFolderProgress = (data: FolderIndexProgress) => {
    setFolderProgress({
      current: data.current || 0,
      total: data.total || data.total_images || 0,
      percent: data.percent || 0,
      currentImage: data.current_image || '',
      facesFound: data.faces_found || 0,
      totalFacesSoFar: data.total_faces_so_far || 0,
    });
  };

  const handleUpload = async () => {
    if (mode === 'files' && files.length === 0) return;
    if (mode === 'folder' && !folderPath.trim()) return;

    setUploading(true);
    setProgress(0);
    setFolderProgress(null);
    setError(null);
    setResult(null);

    try {
      if (mode === 'files') {
        const response = await galleryApi.indexBulk(files, (p) => setProgress(p));
        setResult(response);
        setFiles([]);
      } else {
        const response = await galleryApi.indexFolder(
          {
            folder_path: folderPath.trim(),
            recursive: recursive,
          },
          handleFolderProgress
        );
        setResult(response);
        if (response.success) {
          setFolderPath('');
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      setProgress(0);
      setFolderProgress(null);
    }
  };

  const handleClear = () => {
    setFiles([]);
    setFolderPath('');
    setResult(null);
    setError(null);
    setFolderProgress(null);
  };

  const isFolderResult = (r: BulkIndexResponse | FolderIndexResponse): r is FolderIndexResponse => {
    return 'folder_path' in r;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Upload Photos</h1>
        <p className="text-gray-500 mt-1">
          Upload images or index a folder to add faces to your gallery for searching later.
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setMode('files')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            mode === 'files'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <Upload className="w-4 h-4 inline-block mr-2" />
          Upload Files
        </button>
        <button
          onClick={() => setMode('folder')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            mode === 'folder'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <FolderOpen className="w-4 h-4 inline-block mr-2" />
          Index Folder
        </button>
        <button
          onClick={() => setMode('drive')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            mode === 'drive'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <Cloud className="w-4 h-4 inline-block mr-2" />
          Google Drive
        </button>
      </div>

      {/* File Upload Mode */}
      {mode === 'files' && (
        <>
          {/* Dropzone */}
          <Dropzone
            onFilesAccepted={handleFilesAccepted}
            multiple={true}
            maxFiles={100}
            disabled={uploading}
          />

          {/* Selected Files Preview */}
          {files.length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">
                  Selected Images ({files.length})
                </h2>
                <div className="flex gap-2">
                  <button
                    onClick={handleClear}
                    disabled={uploading}
                    className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
                  >
                    Clear All
                  </button>
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
                  >
                    {uploading ? (
                      <>
                        <LoadingSpinner size="sm" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4" />
                        Upload & Index
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Progress Bar */}
              {uploading && (
                <div className="mb-4">
                  <ProgressBar progress={progress} label="Uploading and indexing faces..." />
                </div>
              )}

              {/* Preview Grid */}
              <ImagePreviewGrid
                files={files}
                onRemove={handleRemoveFile}
                maxDisplay={20}
              />
            </div>
          )}
        </>
      )}

      {/* Folder Index Mode */}
      {mode === 'folder' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Index Local Folder</h2>
          <p className="text-sm text-gray-500 mb-4">
            Enter the full path to a folder containing images. All images will be scanned and faces indexed.
          </p>
          
          <div className="space-y-4">
            <div>
              <label htmlFor="folderPath" className="block text-sm font-medium text-gray-700 mb-1">
                Folder Path
              </label>
              <input
                type="text"
                id="folderPath"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="D:\Photos\Event2024"
                disabled={uploading}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
              />
            </div>
            
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="recursive"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
                disabled={uploading}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <label htmlFor="recursive" className="text-sm text-gray-700">
                Include subfolders (recursive)
              </label>
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={handleClear}
                disabled={uploading}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
              >
                Clear
              </button>
              <button
                onClick={handleUpload}
                disabled={uploading || !folderPath.trim()}
                className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
              >
                {uploading ? (
                  <>
                    <LoadingSpinner size="sm" />
                    Indexing...
                  </>
                ) : (
                  <>
                    <FolderOpen className="w-4 h-4" />
                    Index Folder
                  </>
                )}
              </button>
            </div>

            {/* Folder Indexing Progress */}
            {uploading && folderProgress && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-blue-800">
                    Processing images...
                  </span>
                  <span className="text-sm text-blue-600">
                    {folderProgress.current} / {folderProgress.total}
                  </span>
                </div>
                <ProgressBar progress={folderProgress.percent} />
                <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Current file:</span>
                    <p className="font-medium text-gray-800 truncate" title={folderProgress.currentImage}>
                      {folderProgress.currentImage || 'Starting...'}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">Faces found so far:</span>
                    <p className="font-medium text-gray-800">{folderProgress.totalFacesSoFar}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Google Drive Mode */}
      {mode === 'drive' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Index from Google Drive</h2>
          <p className="text-sm text-gray-500 mb-4">
            Connect to your Google Drive and select a folder containing images. Images are processed 
            in memory and never stored - only face embeddings are saved.
          </p>

          {/* Connection Status */}
          {!driveStatus?.connected ? (
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4 text-center">
                <Cloud className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-600 mb-4">Connect your Google Drive to get started</p>
                <button
                  onClick={handleConnectDrive}
                  disabled={driveLoading}
                  className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2 mx-auto"
                >
                  {driveLoading ? (
                    <>
                      <LoadingSpinner size="sm" />
                      Connecting...
                    </>
                  ) : (
                    <>
                      <Cloud className="w-4 h-4" />
                      Connect to Google Drive
                    </>
                  )}
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Connected User Info */}
              <div className="flex items-center justify-between bg-green-50 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <div>
                    <p className="text-sm font-medium text-green-800">Connected</p>
                    <p className="text-xs text-green-600">{driveStatus.user?.email}</p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={checkDriveStatus}
                    disabled={driveLoading}
                    className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
                    title="Refresh status"
                  >
                    <RefreshCw className={`w-4 h-4 ${driveLoading ? 'animate-spin' : ''}`} />
                  </button>
                  <button
                    onClick={handleDisconnectDrive}
                    disabled={driveLoading}
                    className="px-3 py-1 text-sm text-red-600 hover:bg-red-50 rounded-lg flex items-center gap-1"
                  >
                    <LogOut className="w-4 h-4" />
                    Disconnect
                  </button>
                </div>
              </div>

              {/* Folder Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Selected Folder
                </label>
                <div className="flex gap-2">
                  <div className="flex-1 px-4 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-700">
                    {selectedDriveFolder ? selectedDriveFolder.name : 'No folder selected'}
                  </div>
                  <button
                    onClick={() => setShowDriveBrowser(true)}
                    disabled={uploading}
                    className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    Browse Drive
                  </button>
                </div>
              </div>

              {/* Recursive Option */}
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="driveRecursive"
                  checked={driveRecursive}
                  onChange={(e) => setDriveRecursive(e.target.checked)}
                  disabled={uploading}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <label htmlFor="driveRecursive" className="text-sm text-gray-700">
                  Include subfolders (recursive)
                </label>
              </div>

              {/* Index Button */}
              <div className="flex gap-2">
                <button
                  onClick={() => setSelectedDriveFolder(null)}
                  disabled={uploading || !selectedDriveFolder}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
                >
                  Clear
                </button>
                <button
                  onClick={handleDriveIndex}
                  disabled={uploading || !selectedDriveFolder}
                  className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
                >
                  {uploading ? (
                    <>
                      <LoadingSpinner size="sm" />
                      Indexing...
                    </>
                  ) : (
                    <>
                      <Cloud className="w-4 h-4" />
                      Index Drive Folder
                    </>
                  )}
                </button>
              </div>

              {/* Drive Indexing Progress */}
              {uploading && folderProgress && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-blue-800">
                      Processing images from Drive...
                    </span>
                    <span className="text-sm text-blue-600">
                      {folderProgress.current} / {folderProgress.total}
                    </span>
                  </div>
                  <ProgressBar progress={folderProgress.percent} />
                  <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Current file:</span>
                      <p className="font-medium text-gray-800 truncate" title={folderProgress.currentImage}>
                        {folderProgress.currentImage || 'Starting...'}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-500">Faces found so far:</span>
                      <p className="font-medium text-gray-800">{folderProgress.totalFacesSoFar}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Drive Folder Browser Modal */}
      <DriveFolderBrowser
        isOpen={showDriveBrowser}
        onClose={() => setShowDriveBrowser(false)}
        onSelectFolder={handleDriveFolderSelect}
      />

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-medium text-red-800">Upload Failed</h3>
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </div>
      )}

      {/* Success Result */}
      {result && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-6">
          <div className="flex items-start gap-3">
            <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="font-semibold text-green-800 text-lg">
                {isFolderResult(result) ? 'Folder Indexed!' : 'Upload Complete!'}
              </h3>
              {isFolderResult(result) && (
                <p className="text-sm text-green-700 mt-1">Folder: {result.folder_path}</p>
              )}
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-gray-900">{result.total_images}</p>
                  <p className="text-xs text-gray-500">Images Processed</p>
                </div>
                <div className="bg-white rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-gray-900">{result.total_faces_indexed}</p>
                  <p className="text-xs text-gray-500">Faces Indexed</p>
                </div>
                <div className="bg-white rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-gray-900">{result.failed_images.length}</p>
                  <p className="text-xs text-gray-500">Failed</p>
                </div>
                <div className="bg-white rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-gray-900">
                    {(result.processing_time_ms / 1000).toFixed(1)}s
                  </p>
                  <p className="text-xs text-gray-500">Processing Time</p>
                </div>
              </div>

              {/* Processed Images List */}
              {result.images_processed.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-green-800 mb-2">Processed Images:</h4>
                  <div className="max-h-40 overflow-y-auto bg-white rounded-lg p-3">
                    <div className="space-y-1">
                      {result.images_processed.map((img, i) => (
                        <div key={i} className="flex justify-between text-sm">
                          <span className="text-gray-700 truncate flex-1">{img.image_name}</span>
                          <span className="text-gray-500 ml-4">
                            {img.faces_indexed} face{img.faces_indexed !== 1 ? 's' : ''}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Failed Images */}
              {result.failed_images.length > 0 && (
                <div className="mt-4 bg-red-50 rounded-lg p-3">
                  <h4 className="text-sm font-medium text-red-800 mb-2">Failed Images:</h4>
                  <div className="space-y-1">
                    {result.failed_images.map((img, i) => (
                      <div key={i} className="text-sm text-red-700">
                        {img.image_name}: {img.error}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
