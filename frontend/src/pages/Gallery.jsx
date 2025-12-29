import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Image as ImageIcon } from 'lucide-react';
import axios from 'axios';
import SignInModal from '../components/SignInModal';

const Gallery = () => {
  const { isLoggedIn } = useAuth();
  const [showModal, setShowModal] = useState(!isLoggedIn);
  const [gallery, setGallery] = useState({ public_samples: [], user_feed: [] });

  useEffect(() => {
    if (!isLoggedIn) setShowModal(true);
    else {
      setShowModal(false);
      axios.get('http://localhost:8000/user/gallery')
        .then(res => setGallery(res.data))
        .catch(err => console.error(err));
    }
  }, [isLoggedIn]);

  if (showModal) {
    return (
      <div className="flex items-center justify-center h-96">
        <SignInModal isOpen={true} onClose={() => setShowModal(false)} />
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-4xl font-bold mb-4">Gallery</h2>
      <div className="grid grid-cols-3 gap-4">
        {gallery.public_samples.map((item, idx) => (
          <div key={idx} className="bg-white p-4 rounded-lg shadow">
            <img src={item.thumb} alt={item.title} className="w-full h-48 object-cover rounded mb-2" />
            <h3 className="font-bold text-black">{item.title}</h3>
            <p className="text-gray-600 text-sm">{item.before} → {item.after}</p>
            <button className="mt-2 bg-purple-600 text-white px-4 py-1 rounded">Remix in Chat</button>
          </div>
        ))}
        {gallery.user_feed.map((item, idx) => (
          <div key={idx} className="bg-gray-100 p-4 rounded-lg shadow"> {/* User style */}
            <img src={item.thumb} alt={item.title} className="w-full h-48 object-cover rounded mb-2" />
            <h3 className="font-bold text-black">{item.title}</h3>
            <p className="text-gray-600 text-sm">Your creation</p>
          </div>
        ))}
      </div>
      {gallery.public_samples.length === 0 && <p className="text-gray-500 mt-4">Loading samples...</p>}
    </div>
  );
};

export default Gallery;