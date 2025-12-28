import { useState, useEffect } from 'react';
import { FolderOpen, ArrowLeft, Cloud, Check, Image, Loader2, X } from 'lucide-react';
import { googleDriveApi, type DriveFolder, type DriveFile } from '../services/api';

interface DriveFolderBrowserProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectFolder: (folderId: string, folderName: string) => void;
}

interface FolderStackItem {
  id: string;
  name: string;
}

export function DriveFolderBrowser({ isOpen, onClose, onSelectFolder }: DriveFolderBrowserProps) {
  const [currentFolderId, setCurrentFolderId] = useState('root');
  const [currentFolderName, setCurrentFolderName] = useState('My Drive');
  const [folderStack, setFolderStack] = useState<FolderStackItem[]>([]);
  const [folders, setFolders] = useState<DriveFolder[]>([]);
  const [files, setFiles] = useState<DriveFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [imageCount, setImageCount] = useState(0);

  const loadContents = async (folderId: string) => {
    setLoading(true);
    try {
      const contents = await googleDriveApi.listContents(folderId);
      setFolders(contents.folders);
      setFiles(contents.files.filter(f => f.mimeType.startsWith('image/')));
      setImageCount(contents.files.filter(f => f.mimeType.startsWith('image/')).length);
    } catch (error) {
      console.error('Failed to load folder contents:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadContents(currentFolderId);
    }
  }, [isOpen, currentFolderId]);

  const navigateToFolder = (folder: DriveFolder) => {
    setFolderStack([...folderStack, { id: currentFolderId, name: currentFolderName }]);
    setCurrentFolderId(folder.id);
    setCurrentFolderName(folder.name);
  };

  const goBack = () => {
    if (folderStack.length > 0) {
      const previous = folderStack[folderStack.length - 1];
      setFolderStack(folderStack.slice(0, -1));
      setCurrentFolderId(previous.id);
      setCurrentFolderName(previous.name);
    }
  };

  const handleSelect = () => {
    onSelectFolder(currentFolderId, currentFolderName);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-3">
            <Cloud className="w-6 h-6 text-blue-500" />
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Select Google Drive Folder</h2>
              <p className="text-sm text-gray-500">Choose a folder containing your images</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Breadcrumb / Navigation */}
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-50 border-b">
          {folderStack.length > 0 && (
            <button
              onClick={goBack}
              className="p-1.5 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-4 h-4 text-gray-600" />
            </button>
          )}
          <div className="flex items-center gap-1 text-sm">
            <span className="text-gray-500">📁</span>
            <span className="font-medium text-gray-700">{currentFolderName}</span>
          </div>
          {imageCount > 0 && (
            <span className="ml-auto text-sm text-gray-500 flex items-center gap-1">
              <Image className="w-4 h-4" />
              {imageCount} images in this folder
            </span>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            </div>
          ) : folders.length === 0 && files.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <FolderOpen className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>This folder is empty</p>
            </div>
          ) : (
            <div className="space-y-1">
              {/* Folders */}
              {folders.map((folder) => (
                <button
                  key={folder.id}
                  onClick={() => navigateToFolder(folder)}
                  className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 transition-colors text-left"
                >
                  <FolderOpen className="w-5 h-5 text-yellow-500" />
                  <span className="font-medium text-gray-900">{folder.name}</span>
                </button>
              ))}

              {/* Show preview of images */}
              {files.length > 0 && (
                <div className="mt-4 pt-4 border-t">
                  <p className="text-sm text-gray-500 mb-2">Images in this folder:</p>
                  <div className="grid grid-cols-4 gap-2">
                    {files.slice(0, 8).map((file) => (
                      <div key={file.id} className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                        {file.thumbnailLink ? (
                          <img
                            src={file.thumbnailLink}
                            alt={file.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <Image className="w-6 h-6 text-gray-400" />
                          </div>
                        )}
                      </div>
                    ))}
                    {files.length > 8 && (
                      <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center text-gray-500 text-sm">
                        +{files.length - 8} more
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t bg-gray-50">
          <p className="text-sm text-gray-500">
            {folders.length} folders, {imageCount} images
          </p>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleSelect}
              disabled={imageCount === 0}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Check className="w-4 h-4" />
              Select This Folder ({imageCount} images)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
