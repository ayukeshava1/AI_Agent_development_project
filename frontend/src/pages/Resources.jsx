import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import UpgradeModal from '../components/UpgradeModal';

const Resources = () => {
  const { isLoggedIn } = useAuth();
  const [openSections, setOpenSections] = useState({});  // Toggle per section
  const [openUpgrade, setOpenUpgrade] = useState(false);

  const sections = [
    {
      title: "Best Practices",
      content: "Upload short videos (<5min) for faster PDFs. Use clear audio for best transcription."
    },
    {
      title: "Limits",
      content: isLoggedIn ? "Free: 3 conversions/day. Premium: Unlimited (upgrade below)." : "...",
    },
    {
      title: "Commands",
      content: "/pdf-to-video: Switch mode. /video-to-pdf: Reverse."
    },
    {
      title: "FAQ",
      content: "Q: How accurate is transcription? A: ~80% on clear audio—improve with real data."
    }
  ];

  const toggleSection = (key) => {
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div>
      <h2 className="text-4xl font-bold mb-4">Resources</h2>
      <div className="space-y-4">
        {sections.map((section, idx) => (
          <div key={idx} className="bg-white p-4 rounded-lg shadow">
            <button
              onClick={() => toggleSection(idx)}
              className="w-full flex justify-between items-center text-left"
            >
              <h3 className="text-xl font-semibold">{section.title}</h3>
              {openSections[idx] ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            {openSections[idx] && <p className="mt-2 text-gray-700">{section.content}</p>}
          </div>
        ))}
        {/* // After {sections.map...} */}
        {isLoggedIn && (
          <button onClick={() => setOpenUpgrade(true)} className="mt-4 bg-orange-500 text-white px-6 py-2 rounded">
            Upgrade Now
          </button>
        )}
        <UpgradeModal isOpen={openUpgrade} onClose={() => setOpenUpgrade(false)} />
      </div>
    </div>
  );
};

export default Resources;