import React, { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { useSession, signOut } from 'next-auth/react';
import { useAuth } from '../contexts/AuthContext';
import ProtectedRoute from '../components/ProtectedRoute';
import ChatInterface from '../components/ChatInterface';
import Logo from '../components/Logo';
import SidebarQuotes from '../components/SidebarQuotes';
import UserProfileMenu from '../components/UserProfileMenu';
import PersonalizedSuggestions from '../components/PersonalizedSuggestions';
import FeedbackButton from '../components/FeedbackButton';
import CentralizedSubscriptionModal from '../components/CentralizedSubscriptionModal';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  wisdom_score?: number;
  dharmic_alignment?: number;
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
  plan?: 'basic' | 'pro' | 'max' | 'enterprise';
  isGuest?: boolean;
}

const ChatPage: React.FC = () => {
  const router = useRouter();
  const { data: session, status } = useSession();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowUserDropdown(false);
      }
    };

    if (showUserDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [showUserDropdown]);

  // Check if mobile device
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    const { demo, welcome, subscription } = router.query;
    
    // Open subscription modal if query parameter is present
    if (subscription === 'true') {
      setShowSubscriptionModal(true);
      // Remove the query parameter after opening modal
      router.replace('/chat', undefined, { shallow: true });
    }

    // Redirect to login if not authenticated and not in demo mode
    if (status === 'loading') return; // Still loading
    
    // Handle authenticated users with NextAuth session (priority over demo mode)
    if (session?.user) {
      const authUser: User = {
        name: session.user.name || 'User',
        email: session.user.email || '',
        plan: 'pro' // Google users get pro access
      };
      setUser(authUser);
      
      // Add welcome message for authenticated users
      if (messages.length === 0) {
        setMessages([{
          id: '1',
          sender: 'ai',
          content: `Welcome back to DharmaMind — your AI companion powered by Dharma.
I'm here to support your personal growth and spiritual journey.

What part of your journey would you like to explore today?
DharmaMind is here to help you move forward with calm, clarity, and purpose.`,
          timestamp: new Date(),
          wisdom_score: 95,
          dharmic_alignment: 90
        }]);
      }
      return;
    }
    
    if (demo === 'true') {
      // Create a demo user
      const demoUser: User = {
        name: 'Guest User',
        email: 'demo@dharmamind.com',
        isGuest: true,
        plan: 'basic'
      };
      setUser(demoUser);
      
      // Add welcome message for demo
      if (welcome === 'true') {
        setMessages([{
          id: '1',
          sender: 'ai',
          content: `Welcome to the DharmaMind demo — your AI companion powered by Dharma.
This is a chance to experience an AI with soul, created to support your personal growth.

What part of your journey would you like to explore today?
DharmaMind is here to help you move forward with calm, clarity, and purpose.`,
          timestamp: new Date(),
          wisdom_score: 95,
          dharmic_alignment: 90
        }]);
      } else {
        setMessages([{
          id: '1',
          sender: 'ai',
          content: `Welcome to the DharmaMind demo — your AI companion powered by Dharma.
This is a chance to experience an AI with soul, created to support your personal growth.

What part of your journey would you like to explore today?
DharmaMind is here to help you move forward with calm, clarity, and purpose.`,
          timestamp: new Date(),
          wisdom_score: 85,
          dharmic_alignment: 80
        }]);
      }
      
      return;
    }

    // No session and not demo mode - redirect to demo mode
    if (!session && demo !== 'true') {
      router.push('/chat?demo=true');
      return;
    }
  }, [session, status, router]);

  useEffect(() => {
    if (user) {
      loadChatHistory();
    }
  }, [user]);

  useEffect(() => {
    // Handle loading specific conversation after user is set
    if (user && router.query.load && typeof router.query.load === 'string') {
      const chatId = router.query.load;
      
      if (router.query.demo === 'true') {
        setTimeout(() => loadSpecificDemoConversation(chatId), 100); // Small delay to ensure user is fully set
      } else {
        setTimeout(() => loadSpecificConversation(chatId), 100); // Small delay to ensure user is fully set
      }
    }
  }, [user, router.query.load, router.query.demo]);

  useEffect(() => {
    if (messages.length > 1) {
      saveChatHistory();
    }
  }, [messages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          user: user
        })
      });

      if (!response.ok) throw new Error('Failed to get response');

      const data = await response.json();
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        content: data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        content: 'I apologize, but I encountered an issue. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    // For demo users, go to landing page
    if (router.query.demo === 'true') {
      router.push('/');
      return;
    }
    
    // For authenticated users, use NextAuth signOut and redirect to landing
    if (session) {
      signOut({ redirect: false }).then(() => {
        // Clear localStorage and redirect to landing
        localStorage.removeItem('dharma_user');
        localStorage.removeItem('dharma_chat_history');
        router.push('/');
      });
    } else {
      // Fallback for any remaining localStorage-based auth
      localStorage.removeItem('dharma_user');
      localStorage.removeItem('dharma_chat_history');
      router.push('/');
    }
  };

  // Chat History Functions
  const saveChatHistory = () => {
    if (!user || messages.length <= 1) return;
    
    const chatId = currentChatId || `chat_${Date.now()}`;
    const chatTitle = messages.find(m => m.sender === 'user')?.content.slice(0, 50) + '...' || 'New Chat';
    
    const chatData: ChatHistory = {
      id: chatId,
      title: chatTitle,
      messages: messages,
      lastUpdate: new Date()
    };

    if (router.query.demo === 'true') {
      // For demo users, save to sessionStorage
      const existingHistory = JSON.parse(sessionStorage.getItem('demo_chat_history') || '[]');
      const updatedHistory = existingHistory.filter((chat: ChatHistory) => chat.id !== chatId);
      updatedHistory.unshift(chatData);
      
      // Keep only last 10 chats for demo
      const limitedHistory = updatedHistory.slice(0, 10);
      
      sessionStorage.setItem('demo_chat_history', JSON.stringify(limitedHistory));
      setChatHistory(limitedHistory);
    } else {
      // For real users, save to localStorage
      const existingHistory = JSON.parse(localStorage.getItem('dharma_chat_history') || '[]');
      const updatedHistory = existingHistory.filter((chat: ChatHistory) => chat.id !== chatId);
      updatedHistory.unshift(chatData);
      
      // Keep only last 50 chats
      const limitedHistory = updatedHistory.slice(0, 50);
      
      localStorage.setItem('dharma_chat_history', JSON.stringify(limitedHistory));
      setChatHistory(limitedHistory);
    }
    
    setCurrentChatId(chatId);
  };

  const loadChatHistory = () => {
    if (!user) return;
    
    if (router.query.demo === 'true') {
      // For demo users, load from sessionStorage
      const history = JSON.parse(sessionStorage.getItem('demo_chat_history') || '[]');
      setChatHistory(history);
    } else {
      // For real users, load from localStorage
      const history = JSON.parse(localStorage.getItem('dharma_chat_history') || '[]');
      setChatHistory(history);
    }
  };

  const loadSpecificConversation = (chatId: string) => {
    console.log('Loading specific conversation:', chatId, 'User:', user?.name);
    if (!user || router.query.demo === 'true') return;
    
    const history = JSON.parse(localStorage.getItem('dharma_chat_history') || '[]');
    setChatHistory(history);
    console.log('Loaded history:', history.length, 'conversations');
    
    const chat = history.find((c: ChatHistory) => c.id === chatId);
    console.log('Found chat:', chat ? 'Yes' : 'No', chat?.title);
    
    if (chat) {
      setMessages(chat.messages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp) // Ensure timestamps are Date objects
      })));
      setCurrentChatId(chatId);
      console.log('Loaded', chat.messages.length, 'messages');
    } else {
      // If chat not found, start new chat
      const welcomeMessage: Message = {
        id: '1',
        sender: 'ai',
        content: `Hello ${user?.name || 'there'}! I'm DharmaMind-AI with Soul, powered by Dharma. I'm here to help you with wisdom, clarity, and spiritual insights. How can I assist you today?`,
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
      setCurrentChatId(null);
      console.log('Chat not found, starting new chat');
    }
  };

  const loadSpecificDemoConversation = (chatId: string) => {
    console.log('Loading specific demo conversation:', chatId, 'User:', user?.name);
    if (!user || router.query.demo !== 'true') return;
    
    const history = JSON.parse(sessionStorage.getItem('demo_chat_history') || '[]');
    setChatHistory(history);
    console.log('Loaded demo history:', history.length, 'conversations');
    
    const chat = history.find((c: ChatHistory) => c.id === chatId);
    console.log('Found demo chat:', chat ? 'Yes' : 'No', chat?.title);
    
    if (chat) {
      setMessages(chat.messages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp) // Ensure timestamps are Date objects
      })));
      setCurrentChatId(chatId);
      console.log('Loaded', chat.messages.length, 'demo messages');
    } else {
      // If chat not found, start new demo chat
      const welcomeMessage: Message = {
        id: '1',
        sender: 'ai',
        content: `Welcome to the DharmaMind demo — your AI companion powered by Dharma.
This is a chance to experience an AI with soul, created to support your personal growth.

What part of your journey would you like to explore today?
DharmaMind is here to help you move forward with calm, clarity, and purpose.`,
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
      setCurrentChatId(null);
      console.log('Demo chat not found, starting new demo chat');
    }
  };

  const loadChat = (chatId: string) => {
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
      setMessages(chat.messages);
      setCurrentChatId(chatId);
    }
  };

  const startNewChat = () => {
    const welcomeMessage: Message = {
      id: '1',
      sender: 'ai',
      content: `Hello ${user?.name}! I'm DharmaMind-AI with Soul, powered by Dharma. I'm here to help you with wisdom, clarity, and spiritual insights. How can I assist you today?`,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
    setCurrentChatId(null);
  };

  const deleteChatHistory = (chatId: string) => {
    const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
    localStorage.setItem('dharma_chat_history', JSON.stringify(updatedHistory));
    setChatHistory(updatedHistory);
    
    if (currentChatId === chatId) {
      startNewChat();
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center" 
           style={{backgroundColor: 'var(--color-bg-primary)'}}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2" 
             style={{borderColor: 'var(--color-primary-saffron)'}}></div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>DharmaMind-AI with Soul. Powered by Dharma.</title>
        <meta name="description" content="Chat with DharmaMind AI for spiritual guidance" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="h-screen flex" style={{backgroundColor: 'var(--color-bg-white)'}}>
        
        {/* Sidebar */}
        <div className="hidden md:flex md:w-64 md:flex-col" style={{ backgroundColor: 'var(--color-bg-primary)' }}>
          <div className="flex flex-col h-full" 
               style={{borderRight: '1px solid var(--color-border-light)'}}>
            
            {/* Header */}
            <div className="flex items-center h-16 px-4" 
                 style={{
                   borderBottom: '1px solid var(--color-border-light)',
                   backgroundColor: 'var(--color-bg-secondary)'
                 }}>
              <div className="flex items-center space-x-3 w-full">
                <Logo 
                  size="sm"
                />
                {router.query.demo === 'true' && (
                  <div className="flex items-center space-x-1 px-2 py-1 rounded-md flex-1" 
                       style={{
                         backgroundColor: 'var(--color-border-emerald)',
                         color: 'white'
                       }}>
                    <span className="text-xs">🚀</span>
                    <span className="text-xs font-medium">Demo Mode</span>
                  </div>
                )}
              </div>
            </div>

            {/* Scrollable Content Area */}
            <div className="flex-1 overflow-y-auto" style={{ backgroundColor: 'var(--color-bg-primary)' }}>
              {/* New Chat Button */}
              <div className="p-3">
                <button
                  onClick={() => {
                    setMessages([{
                      id: '1',
                      sender: 'ai',
                      content: `Welcome back, ${user?.name || 'there'}! How can I guide you today?`,
                      timestamp: new Date()
                    }]);
                  }}
                  className="btn-secondary w-full flex items-center px-3 py-2 text-sm font-medium rounded-md"
                >
                  <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  New conversation
                </button>
              </div>

              {/* Content Sections */}
              <div className="px-3 pb-4">
              
              {/* Chat History Section (only for logged-in users) */}
              {user && !router.query.demo && chatHistory.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-xs font-semibold uppercase tracking-wider mb-3 px-2"
                      style={{color: 'var(--color-text-muted)'}}>
                    Recent Chats
                  </h3>
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {chatHistory.slice(0, 8).map((chat, index) => (
                      <button
                        key={chat.id || index}
                        onClick={() => loadChat(chat.id)}
                        className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors group ${
                          currentChatId === chat.id 
                            ? 'border' 
                            : ''
                        }`}
                        style={{
                          backgroundColor: currentChatId === chat.id ? 'var(--color-primary-gradient-light)' : 'transparent',
                          color: currentChatId === chat.id ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                          borderColor: currentChatId === chat.id ? 'var(--color-border-light)' : 'transparent'
                        }}
                        onMouseEnter={(e) => {
                          if (currentChatId !== chat.id) {
                            (e.target as HTMLElement).style.backgroundColor = 'var(--color-bg-secondary)';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (currentChatId !== chat.id) {
                            (e.target as HTMLElement).style.backgroundColor = 'transparent';
                          }
                        }}
                      >
                        <div className="flex items-center justify-between">
                          <span className="truncate flex-1" style={{ color: currentChatId === chat.id ? 'var(--color-text-primary)' : 'var(--color-text-secondary)' }}>
                            {chat.title || 'Untitled Chat'}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteChatHistory(chat.id);
                            }}
                            className="opacity-0 group-hover:opacity-100 ml-2 transition-all"
                            style={{ color: 'var(--color-text-muted)' }}
                            onMouseEnter={(e) => {
                              (e.target as HTMLElement).style.color = 'var(--color-error)';
                            }}
                            onMouseLeave={(e) => {
                              (e.target as HTMLElement).style.color = 'var(--color-text-muted)';
                            }}
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                        <div className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                          {new Date(chat.lastUpdate).toLocaleDateString()}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Spiritual Quotes */}
              <SidebarQuotes />
            </div>
            </div>

            {/* User Profile - Sticky to Bottom */}
            <div className="border-t" style={{ borderColor: 'var(--color-border-light)', backgroundColor: 'var(--color-bg-secondary)' }}>
              <div className="p-4">
                
                {/* Demo User Info - Non-clickable for demo users */}
                {router.query.demo === 'true' ? (
                  <div className="w-full flex items-center space-x-3 p-3 rounded-xl" 
                       style={{
                         backgroundColor: 'var(--color-success-light)',
                         borderColor: 'var(--color-success)',
                         borderWidth: '1px',
                         borderStyle: 'solid'
                       }}>
                    <div className="relative">
                      <div className="w-10 h-10 rounded-full flex items-center justify-center shadow-sm"
                           style={{
                             background: 'linear-gradient(135deg, var(--color-accent), var(--color-success))'
                           }}>
                        <span className="text-white text-sm font-semibold">
                          🚀
                        </span>
                      </div>
                    </div>
                    <div className="flex-1 min-w-0 text-left">
                      <p className="text-sm font-semibold" style={{ color: 'var(--color-success-dark)' }}>
                        Demo Mode
                      </p>
                      <p className="text-xs" style={{ color: 'var(--color-success)' }}>
                        Exploring DharmaMind AI wisdom
                      </p>
                    </div>
                    <div className="flex flex-col space-y-1">
                      <button
                        onClick={() => router.push('/auth?mode=signup')}
                        className="px-2 py-1 text-xs rounded-md transition-colors"
                        style={{
                          backgroundColor: 'var(--color-success)',
                          color: 'white'
                        }}
                        onMouseEnter={(e) => {
                          (e.target as HTMLElement).style.backgroundColor = 'var(--color-success-dark)';
                        }}
                        onMouseLeave={(e) => {
                          (e.target as HTMLElement).style.backgroundColor = 'var(--color-success)';
                        }}
                      >
                        Sign Up
                      </button>
                      <button
                        onClick={() => router.push('/auth?mode=login')}
                        className="px-2 py-1 text-xs rounded-md transition-colors"
                        style={{
                          borderColor: 'var(--color-success)',
                          color: 'var(--color-success)',
                          borderWidth: '1px',
                          borderStyle: 'solid',
                          backgroundColor: 'transparent'
                        }}
                        onMouseEnter={(e) => {
                          (e.target as HTMLElement).style.backgroundColor = 'var(--color-success-light)';
                        }}
                        onMouseLeave={(e) => {
                          (e.target as HTMLElement).style.backgroundColor = 'transparent';
                        }}
                      >
                        Sign In
                      </button>
                    </div>
                  </div>
                ) : (
                  /* Authenticated User Info - Clickable with dropdown */
                  <div className="relative" ref={dropdownRef}>
                    <button
                      onClick={() => setShowUserDropdown(!showUserDropdown)}
                      className="w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 group"
                      style={{
                        backgroundColor: 'transparent',
                        borderColor: 'var(--color-border-light)'
                      }}
                      onMouseEnter={(e) => {
                        (e.target as HTMLElement).style.backgroundColor = 'var(--color-bg-white)';
                        (e.target as HTMLElement).style.boxShadow = '0 1px 3px 0 rgba(0, 0, 0, 0.1)';
                      }}
                      onMouseLeave={(e) => {
                        (e.target as HTMLElement).style.backgroundColor = 'transparent';
                        (e.target as HTMLElement).style.boxShadow = 'none';
                      }}
                    >
                      <div className="relative">
                        <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-emerald-600 rounded-full flex items-center justify-center shadow-sm">
                          <span className="text-white text-sm font-semibold">
                            {user?.name?.charAt(0)?.toUpperCase() || 'U'}
                          </span>
                        </div>
                        {(user?.isGuest || user?.plan === 'basic') && (
                          <div className="absolute -top-1 -right-1 w-4 h-4 bg-yellow-400 rounded-full border-2 border-white flex items-center justify-center">
                            <span className="text-xs">⚡</span>
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0 text-left">
                        <p className="text-sm font-semibold truncate group-hover:text-gray-800" style={{ color: 'var(--color-text-primary)' }}>
                          {user?.name || 'User'}
                        </p>
                        <div className="flex items-center space-x-2">
                          <p className="text-xs capitalize" style={{ color: 'var(--color-text-secondary)' }}>
                            {user?.isGuest ? 'Guest User' : (user?.plan || 'Basic Plan')}
                          </p>
                          {session?.provider === 'google' && (
                            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">
                              Google
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="transition-colors" style={{ color: 'var(--color-text-secondary)' }}>
                        <svg 
                          className={`w-4 h-4 transition-transform duration-200 ${showUserDropdown ? 'rotate-180' : ''}`} 
                          fill="none" 
                          stroke="currentColor" 
                          viewBox="0 0 24 24"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </div>
                    </button>

                    {/* User Dropdown - Only for authenticated users */}
                    {showUserDropdown && (
                      <div className="absolute bottom-full left-0 right-0 mb-2 z-50">
                        <UserProfileMenu
                          user={user}
                          isDemo={false}
                          onUpgrade={() => setShowSubscriptionModal(true)}
                          onClose={() => setShowUserDropdown(false)}
                          onLogout={handleLogout}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          
          {/* Mobile Header */}
          <div className="md:hidden flex items-center justify-between h-16 px-4" 
               style={{
                 borderBottom: '1px solid var(--color-border-light)',
                 backgroundColor: 'var(--color-bg-secondary)'
               }}>
            <div className="flex items-center space-x-3">
              <Logo 
                size="sm"
              />
              {router.query.demo === 'true' && (
                <div className="flex items-center space-x-1 px-2 py-1 rounded-md" 
                     style={{
                       backgroundColor: 'var(--color-accent)',
                       color: 'white'
                     }}>
                  <span className="text-xs">🚀</span>
                  <span className="text-xs font-medium">Demo</span>
                </div>
              )}
            </div>
            <button
              onClick={handleLogout}
              className="transition-colors"
              style={{color: 'var(--color-text-muted)'}}
              onMouseEnter={(e) => {
                (e.target as HTMLElement).style.color = 'var(--color-text-primary)';
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLElement).style.color = 'var(--color-text-muted)';
              }}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto" style={{ backgroundColor: 'var(--color-bg-primary)' }}>
            {/* Demo Banner */}
            {router.query.demo === 'true' && (
              <div className="px-4 py-3" 
                   style={{
                     backgroundColor: 'var(--color-success)',
                     color: 'white',
                     borderBottom: '1px solid var(--color-border-light)'
                   }}>
                <div className="flex items-center justify-between max-w-6xl mx-auto">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">🚀</span>
                    <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-2">
                      <span className="font-medium text-white">Demo Mode</span>
                      <span className="text-sm text-white opacity-90">- Experience DharmaMind's AI wisdom</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={() => {
                        const currentUrl = router.asPath;
                        router.push(`/auth?mode=login&returnUrl=${encodeURIComponent(currentUrl)}`);
                      }}
                      className="btn-ghost px-4 py-2 rounded-md text-sm font-medium transition-colors"
                      style={{
                        backgroundColor: 'rgba(255, 255, 255, 0.2)',
                        color: 'var(--color-text-inverse)',
                        border: 'none'
                      }}
                      onMouseEnter={(e) => {
                        (e.target as HTMLElement).style.backgroundColor = 'rgba(255, 255, 255, 0.3)';
                      }}
                      onMouseLeave={(e) => {
                        (e.target as HTMLElement).style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
                      }}
                    >
                      Sign In
                    </button>
                    <button
                      onClick={() => {
                        const currentUrl = router.asPath;
                        router.push(`/auth?mode=signup&returnUrl=${encodeURIComponent(currentUrl)}`);
                      }}
                      className="px-4 py-2 rounded-md text-sm font-medium transition-colors"
                      style={{
                        backgroundColor: 'var(--color-bg-white)',
                        color: 'var(--color-primary-saffron)'
                      }}
                      onMouseEnter={(e) => {
                        (e.target as HTMLElement).style.backgroundColor = 'var(--color-bg-secondary)';
                      }}
                      onMouseLeave={(e) => {
                        (e.target as HTMLElement).style.backgroundColor = 'var(--color-bg-white)';
                      }}
                    >
                      Sign Up Free
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            <div className="max-w-3xl mx-auto" style={{ backgroundColor: 'var(--color-bg-primary)', minHeight: '100%' }}>
              
              {/* Welcome Screen */}
              {messages.length === 1 && (
                <div className="px-6 py-12 text-center" style={{ backgroundColor: 'var(--color-bg-primary)' }}>
                  <div className="flex justify-center mb-6">
                    <Logo size="lg" showText={false} />
                  </div>
                  <h2 className="text-2xl font-semibold mb-2" style={{color: 'var(--color-text-primary)'}}>
                    What part of your journey would you like to explore right now?
                  </h2>
                  <p className="mb-8" style={{color: 'var(--color-text-secondary)'}}>
                    DharmaMind is here to help you grow — with calm, clarity, and purpose.
                  </p>
                  
                  {/* Personalized Suggestions */}
                  <PersonalizedSuggestions 
                    onSuggestionClick={(suggestion) => setInputValue(suggestion)}
                    messages={messages}
                    className="max-w-4xl mx-auto"
                  />
                </div>
              )}

              {/* Messages */}
              {messages.map((message) => (
                <div
                  key={message.id}
                  className="px-6 py-6 border-b"
                  style={{
                    backgroundColor: message.sender === 'user' 
                      ? 'var(--color-bg-white)' 
                      : 'var(--color-bg-secondary)',
                    borderColor: 'var(--color-border-light)'
                  }}
                >
                  <div className="flex space-x-4">
                    <div className="flex-shrink-0">
                      {message.sender === 'user' ? (
                        <div className="w-8 h-8 rounded-full flex items-center justify-center"
                             style={{
                               background: 'linear-gradient(45deg, var(--color-primary-saffron), var(--color-primary-emerald))'
                             }}>
                          <span className="text-sm font-medium" style={{color: 'var(--color-text-inverse)'}}>
                            {user?.name?.charAt(0)?.toUpperCase() || 'U'}
                          </span>
                        </div>
                      ) : (
                        <Logo size="avatar" showText={false} />
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="text-sm whitespace-pre-wrap leading-relaxed" 
                           style={{color: 'var(--color-text-primary)'}}>
                        {message.content}
                      </div>
                      <div className="mt-2 text-xs text-gray-500">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div className="px-6 py-6" style={{ backgroundColor: 'var(--color-bg-secondary)' }}>
                  <div className="flex space-x-4">
                    <Logo size="avatar" showText={false} />
                    <div className="flex-1">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-75"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="border-t bg-white" style={{ borderColor: 'var(--color-border-light)' }}>
            <div className="max-w-3xl mx-auto px-6 py-4">
              {/* Personalized Suggestions for ongoing conversations */}
              {messages.length > 1 && (
                <PersonalizedSuggestions 
                  onSuggestionClick={(suggestion) => setInputValue(suggestion)}
                  messages={messages}
                  className="mb-4"
                />
              )}
              
              <form onSubmit={handleSubmit} className="flex space-x-4">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Message DharmaMind..."
                    className="w-full px-4 py-3 rounded-lg focus:outline-none text-sm transition-all duration-300"
                    style={{
                      border: '2px solid var(--color-border-light)',
                      backgroundColor: 'var(--color-bg-primary)',
                      color: 'var(--color-text-primary)'
                    }}
                    onFocus={(e) => {
                      (e.target as HTMLInputElement).style.borderColor = 'var(--color-logo-emerald)';
                      (e.target as HTMLInputElement).style.boxShadow = '0 0 0 2px rgba(16, 185, 129, 0.1)';
                    }}
                    onBlur={(e) => {
                      (e.target as HTMLInputElement).style.borderColor = 'var(--color-border-light)';
                      (e.target as HTMLInputElement).style.boxShadow = 'none';
                    }}
                    disabled={isLoading}
                  />
                </div>
                <button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading}
                  className="px-4 py-3 min-w-[48px] rounded-lg font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    backgroundColor: 'var(--color-success)',
                    borderColor: 'var(--color-success)',
                    color: 'white',
                    border: '2px solid var(--color-success)'
                  }}
                  onMouseEnter={(e) => {
                    if (!e.currentTarget.disabled) {
                      (e.target as HTMLElement).style.backgroundColor = 'var(--color-success-dark)';
                      (e.target as HTMLElement).style.borderColor = 'var(--color-success-dark)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!e.currentTarget.disabled) {
                      (e.target as HTMLElement).style.backgroundColor = 'var(--color-success)';
                      (e.target as HTMLElement).style.borderColor = 'var(--color-success)';
                    }
                  }}
                >
                  {isLoading ? (
                    <div 
                      className="w-5 h-5 border-2 rounded-full animate-spin"
                      style={{
                        borderColor: 'white',
                        borderTopColor: 'transparent'
                      }}
                    ></div>
                  ) : (
                    <svg 
                      className="w-5 h-5" 
                      fill="white" 
                      stroke="white" 
                      viewBox="0 0 24 24"
                      style={{ 
                        color: 'white',
                        stroke: 'white',
                        fill: 'white'
                      }}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  )}
                </button>
              </form>
              
              <p className="mt-2 text-xs text-center" style={{ color: 'var(--color-text-muted)' }}>
                DharmaMind can make mistakes. Consider checking important information.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Floating Feedback Button */}
      <FeedbackButton 
        variant="floating" 
        conversationId={currentChatId || undefined}
      />

      {/* Subscription Modal */}
      <CentralizedSubscriptionModal
        isOpen={showSubscriptionModal}
        onClose={() => setShowSubscriptionModal(false)}
      />
    </>
  );
};

export default ChatPage;
