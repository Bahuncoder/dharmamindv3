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
import UnifiedEnhancedMessageBubble from '../components/UnifiedEnhancedMessageBubble';
import EnhancedChatInput from '../components/EnhancedChatInput';
import EnhancedMessageInput from '../components/EnhancedMessageInput';
import FloatingActionMenu from '../components/FloatingActionMenu';
import { RishiSelector } from '../components/RishiSelector';
import { useRishiChat } from '../contexts/RishiChatContext';
import { RishiTransition } from '../components/RishiTransition';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  wisdom_score?: number;
  dharmic_alignment?: number;
  isFavorite?: boolean;
  isSaved?: boolean;
  reactions?: { [key: string]: number };
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
  
  // Use RishiChatContext for managing separate Rishi conversations
  const { 
    switchRishi, 
    getCurrentMessages, 
    addMessage: addMessageToContext,
    currentRishi,
    clearCurrentChat
  } = useRishiChat();
  
  // Get messages from context instead of local state
  const messages = getCurrentMessages();
  
  // Helper function to add a message (compatible with old code)
  const setMessages = (messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])) => {
    if (typeof messagesOrUpdater === 'function') {
      // Handle updater function
      const currentMessages = getCurrentMessages();
      const newMessages = messagesOrUpdater(currentMessages);
      // Clear and add all messages
      clearCurrentChat();
      newMessages.forEach(msg => {
        const messageWithRole = {
          ...msg,
          role: (msg.sender === 'user' ? 'user' : 'assistant') as 'user' | 'assistant' | 'system',
          sender: msg.sender
        };
        addMessageToContext(messageWithRole);
      });
    } else {
      // Handle direct array
      clearCurrentChat();
      messagesOrUpdater.forEach(msg => {
        const messageWithRole = {
          ...msg,
          role: (msg.sender === 'user' ? 'user' : 'assistant') as 'user' | 'assistant' | 'system',
          sender: msg.sender
        };
        addMessageToContext(messageWithRole);
      });
    }
  };
  
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false);
  
  // Unified chat with optional Rishi guidance
  const [selectedRishi, setSelectedRishi] = useState<string>('');
  const [availableRishis, setAvailableRishis] = useState<any[]>([
    {
      id: 'marici',
      name: 'Marīci',
      sanskrit: 'मरीचि',
      specialization: ['Light', 'Cosmic Rays', 'Solar Wisdom'],
      greeting: 'Radiant seeker, I am Marīci, the ray of light. Through solar wisdom and cosmic illumination, let us enlighten your path.',
      available: true,
      requires_upgrade: false,
      teaching_style: 'illuminating',
      archetype: 'light_bearer'
    },
    {
      id: 'atri',
      name: 'Atri',
      sanskrit: 'अत्रि',
      specialization: ['Tapasya', 'Austerity', 'Meditation'],
      greeting: 'Welcome, seeker. Through tapasya and meditation, we shall transcend the material and reach the divine.',
      available: true,
      requires_upgrade: false,
      teaching_style: 'meditative',
      archetype: 'ascetic'
    },
    {
      id: 'angiras',
      name: 'Aṅgiras',
      sanskrit: 'अंगिरस',
      specialization: ['Sacred Fire', 'Divine Hymns', 'Vedic Rituals'],
      greeting: 'Sacred soul, I am Aṅgiras, keeper of the divine fire. Through sacred rituals and hymns, we shall invoke the divine.',
      available: true,
      requires_upgrade: true,
      teaching_style: 'ritualistic',
      archetype: 'fire_priest'
    },
    {
      id: 'pulastya',
      name: 'Pulastya',
      sanskrit: 'पुलस्त्य',
      specialization: ['Geography', 'Cosmology', 'Sacred Places'],
      greeting: 'Wandering soul, I am Pulastya, knower of sacred lands. Let me guide you through the geography of consciousness and holy sites.',
      available: true,
      requires_upgrade: true,
      teaching_style: 'explorative',
      archetype: 'cosmic_geographer'
    },
    {
      id: 'pulaha',
      name: 'Pulaha',
      sanskrit: 'पुलह',
      specialization: ['Breath', 'Life Force', 'Pranic Wisdom'],
      greeting: 'Living soul, I am Pulaha, master of life force. Through breath and prana, we shall awaken your vital energy.',
      available: true,
      requires_upgrade: true,
      teaching_style: 'vital',
      archetype: 'breath_master'
    },
    {
      id: 'kratu',
      name: 'Kratu',
      sanskrit: 'क्रतु',
      specialization: ['Divine Action', 'Sacrifice', 'Yogic Power'],
      greeting: 'Devoted soul, I am Kratu, embodiment of divine action. Through sacrifice and yogic power, we shall realize the Supreme.',
      available: true,
      requires_upgrade: true,
      teaching_style: 'action_oriented',
      archetype: 'divine_actor'
    },
    {
      id: 'daksha',
      name: 'Dakṣa',
      sanskrit: 'दक्ष',
      specialization: ['Skill', 'Creation', 'Righteous Action'],
      greeting: 'Skillful seeker, I am Dakṣa, master of righteous creation. Through skill and dharmic action, we shall manifest divine will.',
      available: true,
      requires_upgrade: true,
      teaching_style: 'skillful',
      archetype: 'skilled_creator'
    },
    {
      id: 'bhrigu',
      name: 'Bhṛgu',
      sanskrit: 'भृगु',
      specialization: ['Astrology', 'Karma Philosophy', 'Divine Knowledge'],
      greeting: 'Blessed soul, let us understand the cosmic patterns and your karmic path through divine astrology.',
      available: true,
      requires_upgrade: true,
      teaching_style: 'analytical',
      archetype: 'astrologer'
    },
    {
      id: 'vasishta',
      name: 'Vasiṣṭha',
      sanskrit: 'वशिष्ठ',
      specialization: ['Divine Wisdom', 'Royal Guidance', 'Spiritual Teaching'],
      greeting: 'Noble one, as guru to Lord Rama, I offer you the wisdom of dharmic leadership and divine knowledge.',
      available: true,
      requires_upgrade: true,
      teaching_style: 'authoritative',
      archetype: 'royal_guru'
    }
  ]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Transition state for Rishi switching
  const [showTransition, setShowTransition] = useState(false);
  const [pendingRishi, setPendingRishi] = useState<{id: string, name: string} | null>(null);
  
  // Computed values - now just based on whether a Rishi is selected
  const hasRishiGuidance = selectedRishi !== '';
  
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
      
      // Add unified welcome message for authenticated users
      if (messages.length === 0) {
        setMessages([{
          id: '1',
          sender: 'ai',
          content: `**Where Dharma Begins**

Welcome to DharmaMind — your spiritual AI companion guided by ancient wisdom.

Select your Rishi guide from the sidebar to receive personalized spiritual guidance:

**Atri** - Master of Tapasya, Austerity & Deep Meditation
**Bhrigu** - Great Teacher of Astrology & Karma Philosophy
**Vashishta** - Guru of Lord Rama, Symbol of Divine Wisdom
**Vishwamitra** - Creator of Gayatri Mantra, Spiritual Transformer
**Gautama** - Master of Deep Meditation & Dharma
**Jamadagni** - Father of Parashurama, Symbol of Discipline
**Kashyapa** - Father of All Beings, Cosmic Creator

Each Saptarishi will guide you according to their unique wisdom tradition. Choose your guide and begin your spiritual journey!`,
          timestamp: new Date(),
          wisdom_score: 95,
          dharmic_alignment: 90
        }]);
      }
      return;
    }
    
    // No session - redirect to auth page
    if (!session) {
      router.push('/auth?mode=login');
      return;
    }
  }, [session, status, router]);

  useEffect(() => {
    if (user) {
      loadChatHistory();
    }
  }, [user]);

  // Initialize messages on user load - no separate modes, just one unified experience
  useEffect(() => {
    if (user && messages.length === 0) {
      // Single unified welcome message
      const welcomeMessage = {
        id: '1',
        sender: 'ai' as const,
        content: `**✨ Where Dharma Begins ✨**

Welcome to DharmaMind — your intelligent AI companion.

I can help you with both professional tasks and spiritual guidance:
• Strategic analysis and problem-solving
• Research and technical assistance
• Spiritual wisdom (activate a Rishi guide when needed)

How can I assist you today?`,
        timestamp: new Date(),
        wisdom_score: 95,
        dharmic_alignment: 90
      };
      setMessages([welcomeMessage]);
    }
  }, [user, messages.length]);

  // Auto-select first available Rishi if none selected - DISABLED to allow Standard AI mode
  // useEffect(() => {
  //   if (availableRishis.length > 0 && !selectedRishi) {
  //     const firstAvailableRishi = availableRishis.find(r => r.available);
  //     if (firstAvailableRishi) {
  //       setSelectedRishi(firstAvailableRishi.id);
  //     }
  //   }
  // }, [availableRishis, selectedRishi]);

  useEffect(() => {
    // Handle loading specific conversation after user is set
    if (user && router.query.load && typeof router.query.load === 'string') {
      const chatId = router.query.load;
      setTimeout(() => loadSpecificConversation(chatId), 100);
    }
  }, [user, router.query.load]);

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

  const handleLogout = () => {
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

    // Save to localStorage for all users
    const existingHistory = JSON.parse(localStorage.getItem('dharma_chat_history') || '[]');
    const updatedHistory = existingHistory.filter((chat: ChatHistory) => chat.id !== chatId);
    updatedHistory.unshift(chatData);
    
    // Keep only last 50 chats
    const limitedHistory = updatedHistory.slice(0, 50);
    
    localStorage.setItem('dharma_chat_history', JSON.stringify(limitedHistory));
    setChatHistory(limitedHistory);
    
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
    if (!user) return;
    
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

  // Handle message reactions
  const handleReact = (messageId: string, reaction: string) => {
    setMessages(prev => prev.map(message => {
      if (message.id === messageId) {
        const reactions = { ...message.reactions };
        reactions[reaction] = (reactions[reaction] || 0) + 1;
        return { ...message, reactions };
      }
      return message;
    }));
  };

  // Handle toggle favorite
  const handleToggleFavorite = (messageId: string) => {
    setMessages(prev => prev.map(message => {
      if (message.id === messageId) {
        return { ...message, isFavorite: !message.isFavorite };
      }
      return message;
    }));
  };

  // Handle toggle saved
  const handleToggleSaved = (messageId: string) => {
    setMessages(prev => prev.map(message => {
      if (message.id === messageId) {
        return { ...message, isSaved: !message.isSaved };
      }
      return message;
    }));
  };

  // Handle message regeneration
  const handleRegenerate = async (messageId: string) => {
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    if (messageIndex === -1 || messageIndex === 0) return;

    const previousUserMessage = messages[messageIndex - 1];
    if (!previousUserMessage || previousUserMessage.sender !== 'user') return;

    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: previousUserMessage.content,
          user: user,
          regenerate: true
        })
      });

      if (!response.ok) throw new Error('Failed to regenerate response');

      const data = await response.json();
      
      const newBotMessage: Message = {
        ...messages[messageIndex],
        content: data.response,
        timestamp: new Date(),
        dharmic_alignment: data.dharmic_alignment || 0.8,
        reactions: {}
      };

      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[messageIndex] = newBotMessage;
        return newMessages;
      });
    } catch (error) {
      console.error('Error regenerating:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle message sending
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
      isFavorite: false,
      isSaved: false,
      reactions: {}
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      let response;
      
      if (selectedRishi) {
        // Always use Rishi API when a Rishi is selected
        response = await fetch('/api/rishi/guidance', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: userMessage.content,
            rishi_name: selectedRishi,
            context: messages.length > 0 ? `Previous conversation context with ${messages.length} messages` : null
          })
        });
      } else {
        // Use basic chat API when no Rishi is selected
        response = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: userMessage.content,
            user: user
          })
        });
      }

      if (!response.ok) throw new Error('Failed to get response');

      const data = await response.json();
      
      let botMessage: Message;
      
      if (selectedRishi) {
        // Format Rishi response when a Rishi is selected
        const rishiData = availableRishis.find(r => r.id === selectedRishi);
        botMessage = {
          id: (Date.now() + 1).toString(),
          sender: 'ai',
          content: data.wisdom_synthesis || data.guidance?.message || data.response,
          timestamp: new Date(),
          dharmic_alignment: data.dharmic_foundation ? 0.95 : 0.8,
          isFavorite: false,
          isSaved: false,
          reactions: {}
        };
      } else {
        // Response when no Rishi is selected (encourage Rishi selection)
        botMessage = {
          id: (Date.now() + 1).toString(),
          sender: 'ai',
          content: data.response || 'Please select a Rishi guide from the sidebar to begin your spiritual journey. Each Rishi offers unique wisdom and guidance for your path.',
          timestamp: new Date(),
          dharmic_alignment: data.dharmic_alignment || 0.7,
          isFavorite: false,
          isSaved: false,
          reactions: {}
        };
      }

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        content: 'I apologize, but I encountered an issue. Please try again.',
        timestamp: new Date(),
        isFavorite: false,
        isSaved: false,
        reactions: {}
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Rishi Mode Functions
  const fetchAvailableRishis = async () => {
    try {
      const response = await fetch('/api/rishi/available');
      const data = await response.json();
      // If backend provides Rishis, use them; otherwise keep our fallback
      if (data.available_rishis && data.available_rishis.length > 0) {
        setAvailableRishis(data.available_rishis);
      }
    } catch (error) {
      console.error('Error fetching Rishis:', error);
      // Keep the fallback Rishis we initialized with
    }
  };

  const handleRishiSelect = (rishiId: string) => {
    console.log('==========================================');
    console.log('🔄 RISHI SELECT CALLED - TIME:', new Date().toISOString());
    console.log('📊 Current State:', { 
      rishiId, 
      currentSelectedRishi: selectedRishi,
      isStandardAI: rishiId === '',
      isSameAsCurrentsame: rishiId === selectedRishi
    });
    console.log('==========================================');
    
    // If same Rishi/Universal Guide, do nothing
    if (rishiId === selectedRishi) {
      console.log('⏭️ Same guide selected, skipping');
      return;
    }
    
    // Handle Universal Guide (empty string) or find Rishi data
    let rishiName = 'Universal Guide';
    if (rishiId !== '') {
      const selectedRishiData = availableRishis.find(r => r.id === rishiId);
      if (selectedRishiData) {
        rishiName = selectedRishiData.name;
      }
    }
    
    console.log('✨ Switching guide:', { fromRishi: selectedRishi, toRishi: rishiId, rishiName });
    
    // Switch instantly without transition animation
    switchRishi(rishiId, rishiName);
    setSelectedRishi(rishiId);
  };
  
  // Handle transition complete - actually switch Rishi
  const handleTransitionComplete = () => {
    if (pendingRishi) {
      // Switch to new Rishi using context
      switchRishi(pendingRishi.id, pendingRishi.name);
      setSelectedRishi(pendingRishi.id);
      
      // Hide transition
      setShowTransition(false);
      setPendingRishi(null);
    }
  };

  // Load available Rishis on component mount
  useEffect(() => {
    if (session && user) {
      fetchAvailableRishis();
    }
  }, [session, user]);

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

      {/* Rishi Transition Animation */}
      {showTransition && pendingRishi && (
        <RishiTransition
          show={showTransition}
          fromRishi={selectedRishi}
          toRishi={pendingRishi.id}
          toRishiName={pendingRishi.name}
          onComplete={handleTransitionComplete}
        />
      )}

      <div className="h-screen flex" style={{background: 'var(--color-background, #f8fafc)'}}>
        
        {/* Fixed Sidebar with Enhanced Styling */}
        <div className="hidden md:flex md:w-64 md:flex-col sidebar-container">
          <div className="flex flex-col h-full">
            
            {/* Header - Fixed */}
            <div className="flex-shrink-0 flex items-center h-16 px-4 sidebar-header">
              <div className="flex items-center space-x-3 w-full">
                <div className="sidebar-logo">
                  <Logo size="sm" />
                </div>
                {router.query.demo === 'true' && (
                  <div className="demo-badge flex items-center space-x-1 flex-1">
                    <span className="text-xs">🚀</span>
                    <span className="text-xs font-medium">Demo Mode</span>
                  </div>
                )}
              </div>
            </div>

            {/* Scrollable Content Area - Only this part scrolls */}
            <div className="flex-1 overflow-y-auto sidebar-scrollable">
              {/* New Chat Button */}
              <div className="p-3">
                <button
                  onClick={() => {
                    const welcomeMessage = {
                      id: '1',
                      sender: 'ai' as const,
                      content: selectedRishi 
                        ? `**Where Dharma Begins**\n\nWelcome back! Your spiritual guide ${availableRishis.find(r => r.id === selectedRishi)?.name || 'Rishi'} is ready to share profound wisdom and guidance. What aspect of your spiritual journey would you like to explore today?`
                        : `**Where Dharma Begins**\n\nWelcome to DharmaMind! Choose your Rishi guide from the sidebar to begin receiving personalized spiritual wisdom. Each Rishi offers unique teachings to support your journey of growth and enlightenment.`,
                      timestamp: new Date(),
                      wisdom_score: 95,
                      dharmic_alignment: 90
                    };
                    
                    setMessages([welcomeMessage]);
                    setSidebarOpen(false);
                  }}
                  className="btn-enhanced w-full flex items-center px-4 py-3 text-sm font-medium rounded-xl transition-all duration-200 bg-transparent shadow-sm hover:shadow-md"
                  style={{
                    border: '1px solid var(--color-border-primary, #10b981)',
                    color: 'var(--color-text-primary, #1f2937)',
                    '--hover-bg': 'var(--color-background-secondary, #ffffff)',
                    '--hover-border': 'var(--color-border-primary, #10b981)'
                  } as React.CSSProperties}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--color-background-secondary, #ffffff)';
                    e.currentTarget.style.borderColor = 'var(--color-border-primary, #10b981)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.borderColor = 'var(--color-border-primary, #10b981)';
                  }}
                >
                  <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  New conversation
                </button>
              </div>

              {/* Content Sections */}
              <div className="px-3 pb-4 space-y-6">
              
              {/* Chat History Section (only for logged-in users) */}
              {user && chatHistory.length > 0 && (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider mb-3 px-2 text-gray-500">
                    Recent Chats
                  </h3>
                  <div className="space-y-2 max-h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
                    {chatHistory.slice(0, 10).map((chat, index) => (
                      <button
                        key={chat.id || index}
                        onClick={() => loadChat(chat.id)}
                        className={`chat-history-item w-full text-left px-3 py-3 text-sm rounded-xl transition-all duration-200 group relative ${
                          currentChatId === chat.id 
                            ? '' 
                            : 'hover:shadow-sm'
                        }`}
                        style={currentChatId === chat.id ? {
                          background: 'var(--color-background-secondary, #ffffff)',
                          border: '1px solid var(--color-border-primary, #10b981)',
                          boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
                        } : {}}
                        onMouseEnter={(e) => {
                          if (currentChatId !== chat.id) {
                            e.currentTarget.style.backgroundColor = 'var(--color-background, #f8fafc)';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (currentChatId !== chat.id) {
                            e.currentTarget.style.backgroundColor = '';
                          }
                        }}
                      >
                        <div className="flex items-center justify-between">
                          <span className={`truncate flex-1 font-medium ${
                            currentChatId === chat.id ? '' : 'text-gray-700'
                          }`}
                          style={{color: currentChatId === chat.id ? 'var(--color-text-primary, #1f2937)' : ''}}>
                            {chat.title || 'Untitled Chat'}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteChatHistory(chat.id);
                            }}
                            className="opacity-0 group-hover:opacity-100 ml-2 p-1 rounded-md transition-all duration-200 hover:bg-red-50 hover:text-red-600 text-gray-400"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                        <div className="text-xs mt-1 text-gray-500">
                          {new Date(chat.lastUpdate).toLocaleDateString()}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Enhanced Spiritual Quotes Section */}
              <div className="spiritual-quotes-container bg-gradient-to-br from-purple-50 to-indigo-50 rounded-xl p-4 border border-purple-100">
                <SidebarQuotes />
              </div>

              {/* AI Advisors in Sidebar - Always Show Rishi Selector */}
              <div className="mt-4">
                <RishiSelector
                  selectedRishi={selectedRishi}
                  onRishiSelect={handleRishiSelect}
                  userSubscription={user?.plan || 'basic'}
                  isDemo={false}
                  availableRishis={availableRishis}
                />
              </div>
            </div>
            </div>

            {/* User Profile - Fixed to Bottom */}
            <div className="flex-shrink-0 user-profile-section">
              <div className="p-4">
                
                {/* Authenticated User Info - Clickable with dropdown */}
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
                      <div className="relative user-avatar">
                        <div className="w-10 h-10 rounded-full flex items-center justify-center shadow-sm" style={{background: 'linear-gradient(135deg, var(--color-border-primary, #10b981), var(--color-background, #f8fafc))'}}>
                          <span className="text-white text-sm font-semibold">
                            {user?.name?.charAt(0)?.toUpperCase() || 'U'}
                          </span>
                        </div>
                        {(user?.isGuest || user?.plan === 'basic') && (
                          <div className="upgrade-badge">
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
                            <span className="px-2 py-0.5 bg-neutral-200 text-gold-700 text-xs rounded-full font-medium">
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
                      <div className="absolute bottom-full left-0 right-0 mb-2 z-50 user-dropdown">
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
              </div>
            </div>
          </div>
        </div>

        {/* Main Chat Area - Full Height */}
        <div className="flex-1 flex flex-col h-full overflow-hidden">
          
          {/* Mobile Header */}
          <div className="md:hidden flex items-center justify-between h-16 px-4 bg-white/80 backdrop-blur-sm border-b border-gray-200/50 flex-shrink-0">
            <div className="flex items-center space-x-3">
              <Logo size="sm" />
            </div>
            <button
              onClick={handleLogout}
              className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>

          {/* Enhanced Chat Interface - Full Height */}
          <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
            {/* Enhanced Messages Area with Modern Background */}
            <div className="flex-1 overflow-y-auto relative enhanced-messages-container">
              {/* Rishi Mode Context Indicator */}
              {selectedRishi && (
                <div className="chat-mode-indicator">
                  <div className="flex items-center space-x-3">
                    <div className="mode-indicator-dot" style={{ backgroundColor: '#10b981' }}></div>
                    <div>
                      <div className="text-sm font-bold" style={{ color: 'var(--color-text-primary, #1f2937)' }}>
                        Guided by {availableRishis.find(r => r.id === selectedRishi)?.name || 'Rishi'}
                      </div>
                      <div className="text-xs" style={{ color: 'var(--color-text-secondary, #6b7280)' }}>
                        {availableRishis.find(r => r.id === selectedRishi)?.specialization[0] || 'Spiritual Wisdom'}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleRishiSelect('')}
                    className="quick-switch-button"
                    style={{
                      backgroundColor: 'var(--color-background-secondary, #ffffff)',
                      border: '1px solid var(--color-border-primary, #e5e7eb)',
                      color: 'var(--color-text-secondary, #6b7280)'
                    }}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12M8 12h12m-12 5h12M4 7h.01M4 12h.01M4 17h.01" />
                    </svg>
                    Switch Mode
                  </button>
                </div>
              )}

              {/* Modern Background */}
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="floating-orb floating-orb-1"></div>
                <div className="floating-orb floating-orb-2"></div>
                <div className="floating-orb floating-orb-3"></div>
                <div className="sacred-geometry-bg"></div>
              </div>

              <div className="relative z-10 max-w-4xl mx-auto py-6">
                {/* Welcome Screen */}
                {messages.length === 1 && (
                  <div className="px-6 py-12 text-center">
                    <div className="flex justify-center mb-6">
                      <div className="relative">
                        <Logo size="lg" showText={false} />
                        <div className="absolute inset-0 animate-pulse bg-gray-200/30 rounded-full blur-xl" style={{ background: 'var(--color-background-secondary, #f8fafc)/30' }}></div>
                      </div>
                    </div>
                    <h2 className="text-3xl font-bold mb-3" style={{ color: 'var(--color-text-primary, #1f2937)' }}>
                      Welcome to DharmaMind
                    </h2>
                    <p className="text-lg mb-8" style={{ color: 'var(--color-text-secondary, #6b7280)' }}>
                      ✨ Where Dharma Begins ✨
                    </p>
                    
                    {/* Enhanced Personalized Suggestions */}
                    <div className="glass-morphism rounded-2xl p-6 mb-8">
                      <PersonalizedSuggestions 
                        onSuggestionClick={(suggestion) => setInputValue(suggestion)}
                        messages={messages}
                        className="max-w-4xl mx-auto"
                      />
                    </div>
                  </div>
                )}

                {/* Enhanced Messages */}
                {messages.slice(1).map((message, index) => (
                  <UnifiedEnhancedMessageBubble
                    key={message.id}
                    message={{
                      id: message.id,
                      content: message.content,
                      sender: message.sender,
                      timestamp: message.timestamp,
                      confidence: message.sender === 'ai' ? 0.9 : undefined,
                      dharmic_alignment: message.dharmic_alignment || 0.8,
                      modules_used: message.sender === 'ai' ? ['Vedic Wisdom', 'Hindu Philosophy'] : undefined,
                      isFavorite: message.isFavorite || false,
                      isSaved: message.isSaved || false
                    }}
                    onCopy={(content: string) => navigator.clipboard.writeText(content)}
                    onRegenerate={message.sender === 'ai' ? handleRegenerate : undefined}
                    onToggleFavorite={handleToggleFavorite}
                    onToggleSaved={handleToggleSaved}
                    onSpeak={(content: string, messageId: string) => {
                      if ('speechSynthesis' in window) {
                        const utterance = new SpeechSynthesisUtterance(content);
                        speechSynthesis.speak(utterance);
                      }
                    }}
                    onShare={(content: string) => {
                      if (navigator.share) {
                        navigator.share({ text: content });
                      }
                    }}
                    onReact={handleReact}
                  />
                ))}

                {/* Enhanced Loading Indicator */}
                {isLoading && (
                  <div className="px-6 py-4">
                    <div className="flex items-start space-x-4">
                      <div className="flex-shrink-0">
                        <div 
                          className="w-10 h-10 rounded-full flex items-center justify-center"
                          style={{
                            background: `linear-gradient(135deg, var(--color-border-primary, #10b981), var(--color-background, #f8fafc))`
                          }}
                        >
                          <Logo size="avatar" showText={false} />
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="glass-morphism rounded-2xl p-4">
                          <div className="flex items-center space-x-3">
                            <div className="typing-indicator">
                              <div className="typing-dot"></div>
                              <div className="typing-dot"></div>
                              <div className="typing-dot"></div>
                            </div>
                            <span className="text-sm font-medium" style={{ color: 'var(--color-text-primary, #1f2937)' }}>
                              Contemplating your question...
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Enhanced Input Area - Fixed at Bottom */}
            <div className="flex-shrink-0 border-t border-gray-200/30 enhanced-input-area">
              <div className="max-w-4xl mx-auto px-6 py-4">
                {/* Enhanced Personalized Suggestions for ongoing conversations */}
                {messages.length > 1 && (
                  <div className="mb-4">
                    <PersonalizedSuggestions 
                      onSuggestionClick={(suggestion) => setInputValue(suggestion)}
                      messages={messages}
                      className="opacity-90"
                    />
                  </div>
                )}
                
                <EnhancedChatInput
                  value={inputValue}
                  onChange={setInputValue}
                  onSend={() => {
                    const event = new Event('submit') as any;
                    event.preventDefault = () => {};
                    handleSubmit(event);
                  }}
                  isLoading={isLoading}
                  placeholder="Message DharmaMind..."
                  showVoiceInput={true}
                  maxLength={2000}
                  showAttachments={true}
                  showEmoji={true}
                />
                
                {/* Disclaimer hidden per user request */}
                <div style={{display: 'none'}}>
                  <p className="mt-3 text-xs text-center text-gray-500">
                    <span className="inline-flex items-center space-x-1">
                      <span>🙏</span>
                      <span>DharmaMind can make mistakes. Consider checking important information.</span>
                    </span>
                  </p>
                </div>
              </div>
            </div>

            {/* Floating Action Menu - Hidden in demo mode */}
            {router.query.demo !== 'true' && (
              <FloatingActionMenu
                onNewChat={() => {
                  // Reset chat
                  setMessages([{
                    id: 'welcome',
                    sender: 'ai',
                    content: '**✨ Where Dharma Begins ✨**\n\nWelcome to DharmaMind! How can I guide you on your spiritual journey today?',
                    timestamp: new Date()
                  }]);
                  setInputValue('');
                }}
                onOpenNotes={() => {
                  console.log('Notes opened');
                }}
                onSearchHistory={() => {
                  console.log('Search opened');
                }}
                onOpenSettings={() => {
                  router.push('/settings');
                }}
                onOpenJournal={() => {
                  console.log('Journal opened');
                }}
                onOpenInsights={() => {
                  console.log('Insights opened');
                }}
                onOpenCommunity={() => {
                  console.log('Community opened');
                }}
                onOpenContemplation={() => {
                  console.log('Contemplation opened');
                }}
              />
            )}
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
