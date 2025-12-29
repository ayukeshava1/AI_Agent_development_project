import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Folder, Video, FileText, Download } from 'lucide-react';
import axios from 'axios';
import SignInModal from '../components/SignInModal';
import ThemeToggle from '../components/ThemeToggle';

const Files = () => {
  const { isLoggedIn } = useAuth();
  const [showModal, setShowModal] = useState(!isLoggedIn);
  const [files, setFiles] = useState([]);
  const [activeTab, setActiveTab] = useState('all');  // Videos, PDFs, Conversions, All
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (!isLoggedIn) setShowModal(true);
    else {
      setShowModal(false);
      // Fetch files
      axios.get('http://localhost:8000/user/files')
        .then(res => setFiles(res.data.files))
        .catch(err => console.error(err));
    }
  }, [isLoggedIn]);

  const filteredFiles = files.filter(f => 
    f.name.toLowerCase().includes(search.toLowerCase()) &&
    (activeTab === 'all' || f.type === activeTab)
  );

  const tabs = [
    { key: 'all', icon: Folder, label: 'All' },
    { key: 'video', icon: Video, label: 'Videos' },
    { key: 'pdf', icon: FileText, label: 'PDFs' },
    { key: 'conversion', icon: Download, label: 'Conversions' }
  ];

  if (showModal) {
    return (
      <div className="flex items-center justify-center h-96">
        <SignInModal isOpen={true} onClose={() => setShowModal(false)} />
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-4xl font-bold mb-4">Files</h2>
      {/* Tabs */}
      <div className="flex space-x-4 mb-4">
        {tabs.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center space-x-1 px-4 py-2 rounded ${activeTab === tab.key ? 'bg-purple-600' : 'bg-gray-200 text-gray-700'}`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>
      {/* Search */}
      <input
        type="text"
        placeholder="Search files..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full p-2 mb-4 border rounded bg-white text-black"
      />
      {/* Grid Cards */}
      <div className="grid grid-cols-3 gap-4">
        {filteredFiles.map((file, idx) => (
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow text-black dark:text-white">  
            <img src={file.thumb} className="w-full h-32 object-cover rounded mb-2" />
            <h3 className="font-bold dark:text-white">{file.name}</h3>
            <p className="text-gray-600 dark:text-gray-300 text-sm">{file.type} - {file.date} - {file.size}</p>
            <button className="mt-2 bg-purple-600 text-white px-4 py-1 rounded hover:bg-purple-700">Download</button>
          </div>
        ))}
      </div>
      {filteredFiles.length === 0 && <p className="text-gray-500 mt-4">No files found.</p>}
    </div>
  );
};



export default Files;