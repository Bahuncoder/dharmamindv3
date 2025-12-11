import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Logo from '../components/Logo';

interface Message {
  id: number;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

interface ChatHistory {
  id: string;
  title: string;
  messages: Message[];
  lastUpdate: Date;
}

interface User {
  name: string;
  email: string;
  plan?: 'free' | 'professional' | 'enterprise';
  isGuest?: boolean;
}

const ConversationsPage: React.FC = () => {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatHistory | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'title'>('date');

  useEffect(() => {
    // Check if user came from demo mode
    const { demo } = router.query;
    
    if (demo === 'true') {
      // For demo users, show current session chat history only
      const demoUser: User = {
        name: 'Guest User',
        email: 'demo@dharmamind.com',
        isGuest: true,
        plan: 'free'
      };
      setUser(demoUser);
      
      // Load demo session chat history (temporary)
      const demoHistory = JSON.parse(sessionStorage.getItem('demo_chat_history') || '[]');
      setChatHistory(demoHistory);
      return;
    }

    // Check for authenticated user session first
    const userData = localStorage.getItem('dharma_user');
    if (!userData) {
      router.push('/');
      return;
    }

    const parsedUser = JSON.parse(userData);
    setUser(parsedUser);

    // Load persistent chat history for real users
    const history = JSON.parse(localStorage.getItem('dharma_chat_history') || '[]');
    setChatHistory(history);
  }, [router]);

  const deleteChatHistory = (chatId: string) => {
    if (confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
      
      // Save based on user type
      if (user?.isGuest) {
        // For demo users, use sessionStorage
        sessionStorage.setItem('demo_chat_history', JSON.stringify(updatedHistory));
      } else {
        // For real users, use localStorage
        localStorage.setItem('dharma_chat_history', JSON.stringify(updatedHistory));
      }
      
      setChatHistory(updatedHistory);
      
      if (selectedChat?.id === chatId) {
        setSelectedChat(null);
      }
    }
  };

  const exportChat = (chat: ChatHistory) => {
    const dataStr = JSON.stringify(chat, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `dharmamind-chat-${chat.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const openChatInMain = (chat: ChatHistory) => {
    // Navigate to chat page with the specific chat loaded
    const chatUrl = user?.isGuest 
      ? `/chat?demo=true&load=${chat.id}` 
      : `/chat?load=${chat.id}`;
    router.push(chatUrl);
  };

  const filteredAndSortedChats = chatHistory
    .filter(chat => 
      chat.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      chat.messages.some(msg => msg.content.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      if (sortBy === 'date') {
        return new Date(b.lastUpdate).getTime() - new Date(a.lastUpdate).getTime();
      } else {
        return a.title.localeCompare(b.title);
      }
    });

  if (!user) return null;

  return (
    <>
      <Head>
        <title>My Conversations - DharmaMind</title>
        <meta name="description" content="Manage and view your DharmaMind conversation history" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <Logo 
                  size="sm"
                  onClick={() => router.push('/')}
                />
                <span className="text-lg font-medium text-gray-600">My Conversations</span>
              </div>

              <div className="flex items-center space-x-4">
                <button
                  onClick={() => router.push(user?.isGuest ? '/chat?demo=true' : '/chat')}
                  className="bg-gradient-to-r from-amber-600 to-gold-600 hover:from-amber-700 hover:to-gold-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                >
                  New Chat
                </button>
                <button
                  onClick={() => router.push(user?.isGuest ? '/settings?demo=true' : '/settings')}
                  className="text-gray-600 hover:text-gray-900 text-sm font-medium"
                >
                  Settings
                </button>
              </div>
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Chat History List */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                <div className="p-6 border-b border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-gray-900">
                      Conversations ({chatHistory.length})
                    </h2>
                  </div>

                  {/* Search and Sort */}
                  <div className="space-y-3">
                    <input
                      type="text"
                      placeholder="Search conversations..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent text-sm"
                    />
                    
                    <select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value as 'date' | 'title')}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent text-sm"
                    >
                      <option value="date">Sort by Date</option>
                      <option value="title">Sort by Title</option>
                    </select>
                  </div>
                </div>

                <div className="max-h-96 overflow-y-auto">
                  {filteredAndSortedChats.length === 0 ? (
                    <div className="p-6 text-center">
                      <div className="w-12 h-12 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                        <span className="text-2xl">💬</span>
                      </div>
                      <p className="text-gray-500 text-sm">
                        {searchTerm ? 'No conversations match your search.' : 'No conversations yet. Start your first chat!'}
                      </p>
                      {!searchTerm && (
                        <button
                          onClick={() => router.push('/chat')}
                          className="mt-4 bg-gradient-to-r from-amber-600 to-gold-600 hover:from-amber-700 hover:to-gold-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                        >
                          Start New Chat
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="divide-y divide-gray-200">
                      {filteredAndSortedChats.map((chat, index) => (
                        <div
                          key={chat.id || index}
                          className={`p-4 hover:bg-gray-50 cursor-pointer transition-colors ${
                            selectedChat?.id === chat.id ? 'bg-neutral-100 border-r-2 border-neutral-1000' : ''
                          }`}
                          onClick={() => setSelectedChat(chat)}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1 min-w-0">
                              <h3 className="text-sm font-medium text-gray-900 truncate">
                                {chat.title || 'Untitled Conversation'}
                              </h3>
                              <p className="text-xs text-gray-500 mt-1">
                                {new Date(chat.lastUpdate).toLocaleDateString('en-US', {
                                  year: 'numeric',
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </p>
                              <p className="text-xs text-gray-400 mt-1">
                                {chat.messages?.length || 0} messages
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Chat Details */}
            <div className="lg:col-span-2">
              {selectedChat ? (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                  <div className="p-6 border-b border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-lg font-semibold text-gray-900">
                          {selectedChat.title || 'Untitled Conversation'}
                        </h2>
                        <p className="text-sm text-gray-500">
                          {new Date(selectedChat.lastUpdate).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })} • {selectedChat.messages?.length || 0} messages
                        </p>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => openChatInMain(selectedChat)}
                          className="px-3 py-1.5 bg-gradient-to-r from-amber-600 to-gold-600 hover:from-amber-700 hover:to-gold-700 text-white rounded-md text-sm font-medium transition-all duration-200"
                        >
                          Continue Chat
                        </button>
                        <button
                          onClick={() => exportChat(selectedChat)}
                          className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors"
                        >
                          Export
                        </button>
                        <button
                          onClick={() => deleteChatHistory(selectedChat.id)}
                          className="px-3 py-1.5 bg-red-100 text-red-700 rounded-md text-sm font-medium hover:bg-red-200 transition-colors"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Messages */}
                  <div className="p-6 max-h-96 overflow-y-auto">
                    <div className="space-y-4">
                      {selectedChat.messages?.map((message, index) => (
                        <div
                          key={index}
                          className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                              message.type === 'user'
                                ? 'bg-gold-600 text-white'
                                : 'bg-gray-100 text-gray-900'
                            }`}
                          >
                            <p className="text-sm">{message.content}</p>
                            <p className={`text-xs mt-1 ${
                              message.type === 'user' ? 'text-neutral-200' : 'text-gray-500'
                            }`}>
                              {new Date(message.timestamp).toLocaleTimeString('en-US', {
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
                  <div className="w-16 h-16 mx-auto mb-6 bg-gray-100 rounded-full flex items-center justify-center">
                    <span className="text-3xl">💬</span>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Select a Conversation
                  </h3>
                  <p className="text-gray-500">
                    Choose a conversation from the list to view its details and messages.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default ConversationsPage;
