import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { googleDriveApi } from '../services/api';

export function DriveCallbackPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('Connecting to Google Drive...');

  useEffect(() => {
    const handleCallback = async () => {
      const code = searchParams.get('code');
      const error = searchParams.get('error');

      if (error) {
        setStatus('error');
        setMessage(`Authorization failed: ${error}`);
        setTimeout(() => navigate('/upload'), 3000);
        return;
      }

      if (!code) {
        setStatus('error');
        setMessage('No authorization code received');
        setTimeout(() => navigate('/upload'), 3000);
        return;
      }

      try {
        const redirectUri = `${window.location.origin}/drive/callback`;
        await googleDriveApi.handleCallback(code, redirectUri);
        setStatus('success');
        setMessage('Successfully connected to Google Drive!');
        
        // Navigate back to upload page after a brief delay
        const returnPath = sessionStorage.getItem('driveAuthReturn') || '/upload';
        sessionStorage.removeItem('driveAuthReturn');
        setTimeout(() => navigate(returnPath), 1500);
      } catch (err) {
        setStatus('error');
        setMessage(err instanceof Error ? err.message : 'Failed to complete authorization');
        setTimeout(() => navigate('/upload'), 3000);
      }
    };

    handleCallback();
  }, [searchParams, navigate]);

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center max-w-md">
        {status === 'loading' && (
          <>
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Connecting...</h2>
            <p className="text-gray-500">{message}</p>
          </>
        )}
        
        {status === 'success' && (
          <>
            <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-green-800 mb-2">Connected!</h2>
            <p className="text-gray-500">{message}</p>
            <p className="text-sm text-gray-400 mt-2">Redirecting...</p>
          </>
        )}
        
        {status === 'error' && (
          <>
            <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-red-800 mb-2">Connection Failed</h2>
            <p className="text-gray-500">{message}</p>
            <p className="text-sm text-gray-400 mt-2">Redirecting back...</p>
          </>
        )}
      </div>
    </div>
  );
}
