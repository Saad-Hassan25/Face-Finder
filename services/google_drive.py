"""
Google Drive Service for Face Finder
Handles OAuth authentication and file operations
"""

import os
import io
import logging
from typing import Optional, List, Dict, Any, Generator
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

# OAuth scopes - read-only access to Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Path to store credentials
CREDENTIALS_FILE = Path(__file__).parent.parent / 'credentials.json'
TOKEN_FILE = Path(__file__).parent.parent / 'google_token.json'


class GoogleDriveService:
    """
    Service for interacting with Google Drive API.
    Handles authentication and file operations.
    """
    
    def __init__(self):
        self.service = None
        self.credentials = None
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load saved credentials if they exist."""
        if TOKEN_FILE.exists():
            try:
                self.credentials = Credentials.from_authorized_user_file(
                    str(TOKEN_FILE), SCOPES
                )
                if self.credentials and self.credentials.valid:
                    self.service = build('drive', 'v3', credentials=self.credentials)
                    logger.info("Loaded existing Google Drive credentials")
                elif self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                    self._save_credentials()
                    self.service = build('drive', 'v3', credentials=self.credentials)
                    logger.info("Refreshed Google Drive credentials")
            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                self.credentials = None
                self.service = None
    
    def _save_credentials(self) -> None:
        """Save credentials to file."""
        if self.credentials:
            with open(TOKEN_FILE, 'w') as token:
                token.write(self.credentials.to_json())
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.service is not None and self.credentials is not None and self.credentials.valid
    
    def get_auth_url(self, redirect_uri: str) -> Optional[str]:
        """
        Get the Google OAuth authorization URL.
        
        Args:
            redirect_uri: The callback URL after authentication
            
        Returns:
            Authorization URL or None if credentials.json is missing
        """
        if not CREDENTIALS_FILE.exists():
            logger.error("credentials.json not found")
            return None
        
        try:
            flow = Flow.from_client_secrets_file(
                str(CREDENTIALS_FILE),
                scopes=SCOPES,
                redirect_uri=redirect_uri
            )
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            return auth_url
        except Exception as e:
            logger.error(f"Failed to create auth URL: {e}")
            return None
    
    def handle_callback(self, code: str, redirect_uri: str) -> bool:
        """
        Handle the OAuth callback and exchange code for tokens.
        
        Args:
            code: Authorization code from Google
            redirect_uri: The callback URL used in the initial request
            
        Returns:
            True if authentication successful
        """
        if not CREDENTIALS_FILE.exists():
            return False
        
        try:
            flow = Flow.from_client_secrets_file(
                str(CREDENTIALS_FILE),
                scopes=SCOPES,
                redirect_uri=redirect_uri
            )
            flow.fetch_token(code=code)
            
            self.credentials = flow.credentials
            self._save_credentials()
            self.service = build('drive', 'v3', credentials=self.credentials)
            
            logger.info("Google Drive authentication successful")
            return True
        except Exception as e:
            logger.error(f"Failed to handle callback: {e}")
            return False
    
    def logout(self) -> bool:
        """Clear stored credentials."""
        try:
            if TOKEN_FILE.exists():
                os.remove(TOKEN_FILE)
            self.service = None
            self.credentials = None
            logger.info("Logged out from Google Drive")
            return True
        except Exception as e:
            logger.error(f"Failed to logout: {e}")
            return False
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the authenticated user."""
        if not self.is_authenticated:
            return None
        
        try:
            about = self.service.about().get(fields="user").execute()
            return about.get('user', {})
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None
    
    def list_folders(self, parent_id: str = 'root') -> List[Dict[str, Any]]:
        """
        List folders in a given parent folder.
        
        Args:
            parent_id: ID of parent folder ('root' for top-level)
            
        Returns:
            List of folder metadata
        """
        if not self.is_authenticated:
            return []
        
        try:
            query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(
                q=query,
                pageSize=100,
                fields="files(id, name, mimeType)",
                orderBy="name"
            ).execute()
            return results.get('files', [])
        except Exception as e:
            logger.error(f"Failed to list folders: {e}")
            return []
    
    def list_files(self, folder_id: str = 'root', images_only: bool = True) -> List[Dict[str, Any]]:
        """
        List files in a given folder.
        
        Args:
            folder_id: ID of folder to list
            images_only: If True, only return image files
            
        Returns:
            List of file metadata
        """
        if not self.is_authenticated:
            return []
        
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            if images_only:
                query += " and (mimeType contains 'image/')"
            
            results = self.service.files().list(
                q=query,
                pageSize=1000,
                fields="files(id, name, mimeType, size, thumbnailLink)",
                orderBy="name"
            ).execute()
            return results.get('files', [])
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def list_all_contents(self, folder_id: str = 'root') -> List[Dict[str, Any]]:
        """
        List all contents (files and folders) in a given folder.
        
        Args:
            folder_id: ID of folder to list
            
        Returns:
            List of file/folder metadata
        """
        if not self.is_authenticated:
            return []
        
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                pageSize=100,
                fields="files(id, name, mimeType, size, thumbnailLink)",
                orderBy="folder, name"
            ).execute()
            return results.get('files', [])
        except Exception as e:
            logger.error(f"Failed to list contents: {e}")
            return []
    
    def download_file(self, file_id: str) -> Optional[bytes]:
        """
        Download a file's content.
        
        Args:
            file_id: ID of file to download
            
        Returns:
            File content as bytes, or None on error
        """
        if not self.is_authenticated:
            return None
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while not done:
                _, done = downloader.next_chunk()
            
            file_buffer.seek(0)
            return file_buffer.read()
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return None
    
    def get_folder_image_count(self, folder_id: str, recursive: bool = True) -> int:
        """
        Count images in a folder.
        
        Args:
            folder_id: ID of folder
            recursive: Whether to count in subfolders
            
        Returns:
            Number of images
        """
        if not self.is_authenticated:
            return 0
        
        count = len(self.list_files(folder_id, images_only=True))
        
        if recursive:
            for folder in self.list_folders(folder_id):
                count += self.get_folder_image_count(folder['id'], recursive=True)
        
        return count
    
    def iterate_images(
        self, 
        folder_id: str, 
        recursive: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate over all images in a folder.
        
        Args:
            folder_id: ID of folder
            recursive: Whether to include subfolders
            
        Yields:
            Dict with file metadata and content
        """
        if not self.is_authenticated:
            return
        
        # Get images in current folder
        images = self.list_files(folder_id, images_only=True)
        for image in images:
            content = self.download_file(image['id'])
            if content:
                yield {
                    'id': image['id'],
                    'name': image['name'],
                    'mime_type': image['mimeType'],
                    'content': content
                }
        
        # Recurse into subfolders
        if recursive:
            for folder in self.list_folders(folder_id):
                yield from self.iterate_images(folder['id'], recursive=True)


# Global instance
_drive_service: Optional[GoogleDriveService] = None


def get_drive_service() -> GoogleDriveService:
    """Get or create the Google Drive service instance."""
    global _drive_service
    if _drive_service is None:
        _drive_service = GoogleDriveService()
    return _drive_service
