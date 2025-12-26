import { useCallback } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

interface DropzoneProps {
  onFilesAccepted: (files: File[]) => void;
  multiple?: boolean;
  maxFiles?: number;
  disabled?: boolean;
  accept?: Record<string, string[]>;
  className?: string;
  children?: React.ReactNode;
}

export function Dropzone({
  onFilesAccepted,
  multiple = true,
  maxFiles = 100,
  disabled = false,
  accept = {
    'image/*': ['.jpeg', '.jpg', '.png', '.webp', '.gif'],
  },
  className = '',
  children,
}: DropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      if (rejectedFiles.length > 0) {
        console.warn('Rejected files:', rejectedFiles);
      }
      if (acceptedFiles.length > 0) {
        onFilesAccepted(acceptedFiles);
      }
    },
    [onFilesAccepted]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    multiple,
    maxFiles,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
        transition-all duration-200 ease-in-out
        ${isDragActive && !isDragReject ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
        ${isDragReject ? 'border-red-500 bg-red-50' : ''}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-400 hover:bg-gray-50'}
        ${className}
      `}
    >
      <input {...getInputProps()} />
      
      {children || (
        <div className="flex flex-col items-center gap-4">
          <div className={`p-4 rounded-full ${isDragActive ? 'bg-blue-100' : 'bg-gray-100'}`}>
            {isDragActive ? (
              <ImageIcon className="w-8 h-8 text-blue-500" />
            ) : (
              <Upload className="w-8 h-8 text-gray-400" />
            )}
          </div>
          
          <div>
            <p className="text-lg font-medium text-gray-700">
              {isDragActive
                ? 'Drop the images here...'
                : 'Drag & drop images here'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or click to select files
            </p>
          </div>
          
          <p className="text-xs text-gray-400">
            Supports: JPEG, PNG, WebP, GIF
            {multiple && ` • Max ${maxFiles} files`}
          </p>
        </div>
      )}
    </div>
  );
}

interface ImagePreviewProps {
  files: File[];
  onRemove: (index: number) => void;
  maxDisplay?: number;
}

export function ImagePreviewGrid({ files, onRemove, maxDisplay = 20 }: ImagePreviewProps) {
  const displayFiles = files.slice(0, maxDisplay);
  const remainingCount = files.length - maxDisplay;

  return (
    <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
      {displayFiles.map((file, index) => (
        <div
          key={`${file.name}-${index}`}
          className="relative group aspect-square rounded-lg overflow-hidden bg-gray-100"
        >
          <img
            src={URL.createObjectURL(file)}
            alt={file.name}
            className="w-full h-full object-cover"
            onLoad={(e) => URL.revokeObjectURL((e.target as HTMLImageElement).src)}
          />
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRemove(index);
            }}
            className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <X className="w-3 h-3" />
          </button>
          <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs p-1 truncate">
            {file.name}
          </div>
        </div>
      ))}
      
      {remainingCount > 0 && (
        <div className="aspect-square rounded-lg bg-gray-200 flex items-center justify-center">
          <span className="text-gray-600 font-medium">+{remainingCount} more</span>
        </div>
      )}
    </div>
  );
}
