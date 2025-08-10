import React from 'react';
import { AuthProvider } from '../contexts/AuthContext';
import { ColorProvider } from '../contexts/ColorContext';
import ChatInterface from '../components/ChatInterface';

// Demo page to test the user profile functionality
export default function ChatDemo() {
  return (
    <AuthProvider>
      <ColorProvider>
        <div className="h-screen">
          <ChatInterface />
        </div>
      </ColorProvider>
    </AuthProvider>
  );
}
