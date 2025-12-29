import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useSearchParams } from 'react-router-dom';
import SignInModal from '../components/SignInModal';
import ChatContainer from '../components/ChatContainer';
import InputBox from '../components/InputBox';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import io from 'socket.io-client';


const Converter = () => {
  const { isLoggedIn } = useAuth();
  const [searchParams] = useSearchParams();
  const [showModal, setShowModal] = useState(!isLoggedIn);
  const [messages, setMessages] = useState([
    { text: "Hi! Upload a file or type /pdf-to-video or /video-to-pdf to start.", isUser: false }
  ]);
  const [mode, setMode] = useState(searchParams.get('mode') || 'pdf-to-video');
  const [socket, setSocket] = useState(null);  // Socket state
  const [currentJob, setCurrentJob] = useState(null);

  useEffect(() => {
    if (!isLoggedIn) setShowModal(true);
    else setShowModal(false);
  }, [isLoggedIn]);

  // Socket setup (inside component)
  useEffect(() => {
    const newSocket = io('http://localhost:8000');
    setSocket(newSocket);
    newSocket.on('progress_start', (data) => {
      setCurrentJob(data.job_id);
      setMessages(prev => [...prev, { text: data.message, isUser: false }]);
    });
    newSocket.on('progress_update', (data) => {
      if (data.job_id === currentJob) {
        setMessages(prev => [...prev, { text: data.step, isUser: false, progress: data.percent / 100 }]);
      }
    });
    newSocket.on('conversion_complete', (data) => {
      if (data.job_id === currentJob) {
        setMessages(prev => [...prev, { text: "Complete! Preview ready.", isUser: false, output: data.output }]);
        setCurrentJob(null);
      }
    });
    return () => newSocket.close();
  }, []);

  const handleSend = (newMsg) => {
    setMessages(prev => [...prev, newMsg]);  // Add user msg once

    // Commands parser
    if (newMsg.text && newMsg.text.startsWith('/')) {
      let response = { text: "Unknown command.", isUser: false };
      if (newMsg.text.includes('pdf-to-video')) {
        setMode('pdf-to-video');
        response.text = "Switched to PDF to Video mode. Upload a PDF!";
      } else if (newMsg.text.includes('video-to-pdf')) {
        setMode('video-to-pdf');
        response.text = "Switched to Video to PDF mode. Upload a video!";
      }
      setTimeout(() => setMessages(prev => [...prev, response]), 500);
      return;
    }

    // Echo non-file
    if (!newMsg.file) {
      setTimeout(() => setMessages(prev => [...prev, { text: `Echo: ${newMsg.text} in ${mode} mode.`, isUser: false }]), 500);
      return;
    }

    // File flow
    const formData = new FormData();
    formData.append('file', newMsg.file);
    formData.append('mode', mode);

    axios.post('http://localhost:8000/convert', formData)
      .then(res => {
        if (res.data.limit_exceeded) {
          toast.error('Daily limit reached (3/3 free)—upgrade to premium!');
          return;
        }
        // Initial
        setMessages(prev => [...prev, { text: "Processing your request...", isUser: false }]);

        // Steps (Socket fallback to timer)
        if (socket) {
          // Backend emits to this client (job_id match if needed)
        } else {
          res.data.progress.forEach((step, i) => {
            setTimeout(() => {
              setMessages(prev => [...prev, { text: step, isUser: false, progress: (i + 1) / res.data.progress.length }]);
            }, i * 1500);
          });
          setTimeout(() => {
            setMessages(prev => [...prev, { text: "Conversion complete!", isUser: false, output: res.data.output }]);
          }, res.data.progress.length * 1500);
        }
      })
      .catch(err => {
        setMessages(prev => [...prev, { text: `Error: ${err.message}`, isUser: false }]);
      });
  };

  if (showModal) {
    return (
      <div className="flex items-center justify-center h-96">
        <SignInModal isOpen={true} onClose={() => setShowModal(false)} />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <h2 className="text-4xl font-bold mb-4">Converter - {mode.toUpperCase()} Mode</h2>
      <ChatContainer messages={messages} mode={mode} />
      <InputBox onSend={handleSend} mode={mode} />
      <Toaster position="top-right" />
    </div>
  );
};

export default Converter;