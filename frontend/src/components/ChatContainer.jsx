import { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';  // We'll create next

const ChatContainer = ({ messages, mode }) => {
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-100 rounded-lg max-h-96">
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} msg={msg} isUser={msg.isUser} mode={mode} />
      ))}
      <div ref={scrollRef} />  {/* Auto-scroll anchor */}
    </div>
  );
};

export default ChatContainer;