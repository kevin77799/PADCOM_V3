import React, { useState, useRef, useEffect } from 'react';
import { Mic, Send, Volume2 } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  audio?: string;
}

const PadcomChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send text message
  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: inputText };
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, audioMode: false })
      });

      const data = await response.json();
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.text
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Start recording audio
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await sendAudioMessage(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Send audio message
  const sendAudioMessage = async (audioBlob: Blob) => {
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      // Add transcribed user message
      setMessages(prev => [...prev, {
        role: 'user',
        content: `🎤 ${data.text || 'Voice message'}`
      }]);
      
      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.text,
        audio: data.audio
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
      // Auto-play audio response
      if (data.audio) {
        playAudio(data.audio);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I could not process your voice message.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Play audio response
  const playAudio = (base64Audio: string) => {
    const audio = new Audio(`data:audio/mp3;base64,${base64Audio}`);
    audio.play();
  };

  return (
    <div className="min-h-screen bg-gray-900 text-cyan-400 font-mono">
      {/* Header */}
      <div className="bg-black bg-opacity-50 backdrop-blur-md border-b border-cyan-700 p-4">
        <h1 className="text-3xl font-bold text-cyan-300">
          PADCOM <span className="text-sm text-gray-500">v3.0</span>
        </h1>
        <p className="text-xs text-gray-400">System Online • Agentic AI Companion</p>
      </div>

      {/* Messages Container */}
      <div className="max-w-4xl mx-auto p-6 h-[calc(100vh-180px)] overflow-y-auto">
        {messages.length === 0 ? (
          <div className="text-center mt-20">
            <div className="text-6xl mb-4">🤖</div>
            <h2 className="text-2xl mb-2 text-cyan-300">PADCOM Ready</h2>
            <p className="text-gray-500">Ask me anything. I have access to your personal knowledge base.</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[70%] p-4 rounded-lg ${
                  msg.role === 'user'
                    ? 'bg-cyan-800 text-white'
                    : 'bg-gray-800 text-cyan-300 border border-cyan-700'
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
                {msg.audio && (
                  <button
                    onClick={() => playAudio(msg.audio!)}
                    className="mt-2 text-xs flex items-center gap-1 text-cyan-400 hover:text-cyan-300"
                  >
                    <Volume2 size={14} /> Play Audio
                  </button>
                )}
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-800 p-4 rounded-lg border border-cyan-700">
              <p className="text-cyan-400 animate-pulse">PADCOM is thinking...</p>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="fixed bottom-0 left-0 right-0 bg-black bg-opacity-70 backdrop-blur-md border-t border-cyan-700 p-4">
        <div className="max-w-4xl mx-auto flex gap-2">
          {/* Text Input */}
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1 bg-gray-800 text-cyan-300 border border-cyan-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-cyan-500 placeholder-gray-600"
          />
          
          {/* Send Button */}
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputText.trim()}
            className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-700 text-white p-3 rounded-lg transition"
          >
            <Send size={20} />
          </button>
          
          {/* Voice Button */}
          <button
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
            disabled={isLoading}
            className={`p-3 rounded-lg transition ${
              isRecording
                ? 'bg-red-600 animate-pulse'
                : 'bg-cyan-600 hover:bg-cyan-700'
            } disabled:bg-gray-700 text-white`}
          >
            <Mic size={20} />
          </button>
        </div>
        <p className="text-center text-xs text-gray-600 mt-2">
          {isRecording ? 'Recording... Release to send' : 'Hold mic button to record voice message'}
        </p>
      </div>
    </div>
  );
};

export default PadcomChat;
