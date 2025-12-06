import React from 'react';
import { Link } from 'react-router-dom';
import { Settings } from 'lucide-react';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-black bg-opacity-50 backdrop-blur-md text-cyan-300 shadow-lg border-b border-cyan-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <span className="text-2xl font-bold text-cyan-400">PADCOM</span>
              <span className="ml-2 text-xs text-gray-500">System Online</span>
            </Link>
          </div>
          <div className="flex items-center space-x-4">
            <button className="p-2 rounded-full hover:bg-gray-800 transition">
              <Settings size={20} className="text-cyan-400" />
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 