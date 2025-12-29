import { Download } from 'lucide-react';
import { Document, Page, pdfjs } from 'react-pdf';
import toast from 'react-hot-toast';
import { useState, useEffect } from 'react';


pdfjs.GlobalWorkerOptions.workerSrc = 'https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js';  // Stable v3.11

const MessageBubble = ({ msg, isUser, mode }) => {
  const [typing, setTyping] = useState(false);
  const [numPages, setNumPages] = useState(null);

  useEffect(() => {
    if (msg.progress !== undefined) setTyping(false);
    else if (!isUser && !msg.output) {
      setTyping(true);
      setTimeout(() => setTyping(false), 1000);
    }
  }, [msg]);

  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);

  if (typing) {
    return (
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
        <div className={`max-w-md p-3 rounded-lg ${isUser ? 'bg-blue-500' : 'bg-purple-500'}`}>
          <div className="flex space-x-1">
            <span className="w-2 h-2 bg-white rounded-full animate-bounce"></span>
            <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></span>
            <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></span>
          </div>
        </div>
      </div>
    );
  }

  if (msg.progress !== undefined) {
    return (
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
        <div className={`max-w-md p-3 rounded-lg ${isUser ? 'bg-blue-500' : 'bg-purple-500'}`}>
          <p className="text-white mb-2">{msg.text}</p>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-green-600 h-2 rounded-full transition-all" style={{ width: `${msg.progress * 100}%` }}></div>
          </div>
        </div>
      </div>
    );
  }

  const isOutput = msg.output;
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-md p-3 rounded-lg ${isUser ? 'bg-blue-500' : 'bg-purple-500'}`}>
        <p className="text-white mb-2">{msg.text}</p>
        {isOutput && (
          <div className="space-y-2">
            {mode === 'video-to-pdf' ? (
              <Document file={msg.output.previewUrl} onLoadSuccess={onDocumentLoadSuccess}>
                {numPages && Array.from(new Array(1), (el, index) => (
                  <Page key={`page_${index + 1}`} pageNumber={index + 1} width={200} />
                ))}
              </Document>
            ) : (
              <video src={msg.output.previewUrl} controls className="w-full rounded max-h-48">
                Your browser does not support the video tag. (Stub URL)
              </video>
            )}
            <button 
              onClick={() => {
                const link = document.createElement('a');
                link.href = msg.output.downloadUrl;
                link.download = 'conversion.zip';
                link.click();
                toast.success('Downloaded ZIP!');
              }}
              className="flex items-center space-x-2 text-white bg-green-600 p-2 rounded hover:bg-green-700 w-full justify-center"
            >
              <Download className="w-4 h-4" />
              <span>Download ZIP</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;