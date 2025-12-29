import { Home, MessageSquare, Folder, Image, BookOpen } from 'lucide-react';
import { Link } from 'react-router-dom';

const Sidebar = () => (
  <aside className="w-64 bg-blue-600 p-4 h-screen shadow-lg">  {/* Always fixed, no translate */}
    <nav className="space-y-2">
      <Link to="/" className="flex items-center space-x-3 p-3 rounded-lg hover:bg-blue-500 ark:hover:bg-blue-900 transition-colors">
        <Home className="w-5 h-5" />
        <span className="text-white font-medium">Home</span>
      </Link>
      <div className="p-3 rounded-lg hover:bg-blue-500 transition-colors space-y-1">
        <span className="text-xs text-blue-200">Modes</span>
        <Link to="/converter?mode=pdf-to-video" className="flex items-center space-x-2 ark:hover:bg-blue-900 text-white font-medium">
          <span>📄→🎥 PDF to Video</span>
        </Link>
        <Link to="/converter?mode=video-to-pdf" className="flex items-center space-x-2 text-white ark:hover:bg-blue-900 font-medium">
          <span>🎥→📄 Video to PDF</span>
        </Link>
      </div>
      <Link to="/files" className="flex items-center space-x-3 p-3 rounded-lg hover:bg-blue-500 ark:hover:bg-blue-900 transition-colors">
        <Folder className="w-5 h-5" />
        <span className="text-white font-medium">Files</span>
      </Link>
      <Link to="/gallery" className="flex items-center space-x-3 p-3 rounded-lg hover:bg-blue-500 ark:hover:bg-blue-900 transition-colors">
        <Image className="w-5 h-5" />
        <span className="text-white font-medium">Gallery</span>
      </Link>
      <Link to="/resources" className="flex items-center space-x-3 p-3 rounded-lg hover:bg-blue-500 ark:hover:bg-blue-900 transition-colors">
        <BookOpen className="w-5 h-5" />
        <span className="text-white font-medium">Resources</span>
      </Link>
    </nav>
  </aside>
);

export default Sidebar;