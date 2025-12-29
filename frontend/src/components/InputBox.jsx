import { useState } from 'react';
import { Plus, Send } from 'lucide-react';

const InputBox = ({ onSend, mode }) => {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFile = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.size < 50 * 1024 * 1024 && (selected.type.includes('pdf') || selected.type.includes('video'))) {
      setFile(selected);
      const url = URL.createObjectURL(selected);
      setPreview(url);
    } else {
      alert('File too big or wrong type (PDF/Video only, <50MB)!');
    }
  };

  const handleSend = () => {
    if (text.trim() || file) {
      onSend({ text, file, isUser: true });  // Only call parent—no local setMessages
      setText('');
      setFile(null);
      setPreview(null);
      if (preview) URL.revokeObjectURL(preview);
    }
  };

  return (
    <div className="flex space-x-2 p-4 bg-white border-t rounded-b-lg">
      <input type="file" onChange={handleFile} className="hidden" id="file-upload" accept=".pdf,.mp4" />
      <label htmlFor="file-upload" className="p-2 bg-gray-200 rounded hover:bg-gray-300 cursor-pointer">
        <Plus className="w-5 h-5 text-gray-600" />
      </label>
      {preview && <img src={preview} alt="Preview" className="w-8 h-8 rounded" />}
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={`Type message or /${mode.split('-')[0]} to switch mode...`}
        className="flex-1 p-2 border rounded resize-none"
        rows="1"
        onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
      />
      <button onClick={handleSend} className="p-2 bg-purple-600 text-white rounded hover:bg-purple-700">
        <Send className="w-5 h-5" />
      </button>
    </div>
  );
};

export default InputBox;