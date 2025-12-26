import { useState } from 'react';
import { type ImageMatch, type BoundingBox, getImageUrl } from '../services/api';

interface ResultsGridProps {
  images: ImageMatch[];
  onImageClick?: (image: ImageMatch) => void;
  emptyMessage?: string;
}

export function ResultsGrid({ images, onImageClick, emptyMessage = 'No images found' }: ResultsGridProps) {
  if (images.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
      {images.map((image, index) => (
        <ImageCard
          key={`${image.image_id}-${index}`}
          image={image}
          onClick={() => onImageClick?.(image)}
        />
      ))}
    </div>
  );
}

interface ImageCardProps {
  image: ImageMatch;
  onClick?: () => void;
}

function ImageCard({ image, onClick }: ImageCardProps) {
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  
  const hasImage = image.file_path && !imageError;
  const imageUrl = image.file_path ? getImageUrl(image.file_path) : null;

  return (
    <div
      onClick={onClick}
      className="group relative bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden cursor-pointer hover:shadow-md transition-shadow"
    >
      <div className="aspect-square bg-gray-100 relative overflow-hidden">
        {hasImage && imageUrl ? (
          <>
            {!imageLoaded && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
              </div>
            )}
            <img
              src={imageUrl}
              alt={image.image_name || 'Image'}
              className={`w-full h-full object-cover transition-opacity ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageError(true)}
            />
          </>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center p-4">
              <div className="w-16 h-16 mx-auto bg-gray-200 rounded-lg flex items-center justify-center mb-2">
                <span className="text-2xl">🖼️</span>
              </div>
              <p className="text-xs text-gray-500 truncate max-w-full px-2">
                {image.image_name || image.image_id}
              </p>
            </div>
          </div>
        )}
      </div>
      
      <div className="p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-700 truncate flex-1">
            {image.image_name || 'Unknown'}
          </span>
          <SimilarityBadge similarity={image.similarity} />
        </div>
      </div>
    </div>
  );
}

interface SimilarityBadgeProps {
  similarity: number;
}

export function SimilarityBadge({ similarity }: SimilarityBadgeProps) {
  const percentage = Math.round(similarity * 100);
  
  let colorClass = 'bg-red-100 text-red-700';
  if (percentage >= 80) {
    colorClass = 'bg-green-100 text-green-700';
  } else if (percentage >= 60) {
    colorClass = 'bg-yellow-100 text-yellow-700';
  } else if (percentage >= 40) {
    colorClass = 'bg-orange-100 text-orange-700';
  }

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${colorClass}`}>
      {percentage}%
    </span>
  );
}

interface FaceBoundingBoxProps {
  bbox: BoundingBox;
  imageWidth: number;
  imageHeight: number;
  label?: string;
}

export function FaceBoundingBox({ bbox, imageWidth, imageHeight, label }: FaceBoundingBoxProps) {
  const style = {
    left: `${(bbox.x / imageWidth) * 100}%`,
    top: `${(bbox.y / imageHeight) * 100}%`,
    width: `${(bbox.width / imageWidth) * 100}%`,
    height: `${(bbox.height / imageHeight) * 100}%`,
  };

  return (
    <div
      className="absolute border-2 border-blue-500 bg-blue-500/10"
      style={style}
    >
      {label && (
        <span className="absolute -top-6 left-0 bg-blue-500 text-white text-xs px-2 py-0.5 rounded">
          {label}
        </span>
      )}
    </div>
  );
}

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  subtitle?: string;
}

export function StatsCard({ title, value, icon, subtitle }: StatsCardProps) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-3xl font-bold text-gray-900 mt-1">{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
          )}
        </div>
        <div className="p-3 bg-blue-50 rounded-xl">
          {icon}
        </div>
      </div>
    </div>
  );
}

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  message?: string;
}

export function LoadingSpinner({ size = 'md', message }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <div
        className={`${sizeClasses[size]} border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin`}
      />
      {message && (
        <p className="text-sm text-gray-500">{message}</p>
      )}
    </div>
  );
}

interface ProgressBarProps {
  progress: number;
  label?: string;
}

export function ProgressBar({ progress, label }: ProgressBarProps) {
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>{label}</span>
          <span>{progress}%</span>
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2.5">
        <div
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
