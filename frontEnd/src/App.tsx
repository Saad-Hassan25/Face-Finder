import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components';
import {
  HomePage,
  UploadPage,
  GalleryPage,
  SearchPage,
  VerifyPage,
  AlbumsPage,
  SettingsPage,
} from './pages';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/gallery" element={<GalleryPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/verify" element={<VerifyPage />} />
            <Route path="/albums" element={<AlbumsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
