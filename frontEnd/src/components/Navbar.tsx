import { NavLink } from 'react-router-dom';
import { Upload, Search, Users, Settings, Home, ImageIcon } from 'lucide-react';

export function Navbar() {
  const navItems = [
    { to: '/', icon: Home, label: 'Home' },
    { to: '/upload', icon: Upload, label: 'Upload' },
    { to: '/gallery', icon: ImageIcon, label: 'Gallery' },
    { to: '/search', icon: Search, label: 'Find Person' },
    { to: '/verify', icon: Users, label: 'Verify' },
    { to: '/settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Search className="w-5 h-5 text-white" />
              </div>
              <span className="ml-2 text-xl font-bold text-gray-900">Face Finder</span>
            </div>
          </div>

          <div className="flex items-center space-x-1">
            {navItems.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors
                  ${isActive 
                    ? 'bg-blue-50 text-blue-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
