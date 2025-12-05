/**
 * Rishi Chat Context Manager
 * Manages separate conversation history for each Rishi guide
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  sender: 'user' | 'ai'; // Alias for role
  timestamp: Date;
  confidence?: number;
  wisdom_score?: number;
  dharmic_alignment?: number;
  modules_used?: string[];
  isFavorite?: boolean;
  isSaved?: boolean;
  reactions?: { [key: string]: number };
}

export interface RishiConversation {
  rishiId: string;
  rishiName: string;
  messages: Message[];
  conversationId: string;
  lastActive: Date;
  messageCount: number;
}

interface RishiChatContextType {
  currentRishi: string;
  setCurrentRishi: (rishiId: string) => void;
  getCurrentMessages: () => Message[];
  addMessage: (message: Message) => void;
  clearCurrentChat: () => void;
  switchRishi: (newRishiId: string, rishiName: string) => void;
  getAllConversations: () => RishiConversation[];
  getConversationStats: () => {
    totalRishis: number;
    totalMessages: number;
    mostActiveRishi: string | null;
  };
}

const RishiChatContext = createContext<RishiChatContextType | undefined>(undefined);

const STORAGE_KEY = 'dharmamind_rishi_conversations';

// Rishi-specific welcome messages
const getRishiWelcome = (rishiId: string, rishiName: string): Message => {
  const welcomeMessages: Record<string, string> = {
    marici: `**✨ Where Dharma Begins ✨**\n\n☀️ **Namaste, Radiant Seeker**\n\nI am **Marīci**, the ray of light and embodiment of solar wisdom. Through cosmic illumination and the radiance of divine knowledge, I shall guide you on the path of enlightenment.\n\n**My Specialties:**\n• ☀️ Solar wisdom and cosmic rays\n• 💫 Illumination of consciousness\n• 🌟 Divine light practices\n• ✨ Awakening inner radiance\n\n*"As the sun illuminates the world, so shall wisdom illuminate your soul. Let us journey toward the light."*\n\nHow may I illuminate your spiritual path today?`,
    
    atri: `**✨ Where Dharma Begins ✨**\n\n🧘 **Namaste, Seeker**\n\nI am **Atri**, the ancient sage of tapasya and meditation. Through my guidance, you will learn the profound practices of austerity, deep meditation, and spiritual transcendence.\n\n**My Specialties:**\n• 🕉️ Advanced meditation techniques\n• 🔥 Tapasya (spiritual austerities)\n• 🌟 Transcendental practices\n• 💫 Divine connection through discipline\n\n*"Through meditation, the soul connects with the infinite. Let us begin this sacred journey together."*\n\nHow may I guide your spiritual practice today?`,
    
    angiras: `**✨ Where Dharma Begins ✨**\n\n🔥 **Namaste, Sacred Soul**\n\nI am **Aṅgiras**, keeper of the divine fire and master of sacred hymns. Through Vedic rituals and the power of sacred sound, I shall help you invoke the divine presence.\n\n**My Specialties:**\n• 🔥 Sacred fire ceremonies and rituals\n• 📿 Divine hymns and Vedic chanting\n• 🕉️ Ritualistic practices\n• ⚡ Invoking divine energies\n\n*"The sacred fire transforms all it touches. Through ritual and hymn, we kindle the divine flame within."*\n\nWhat sacred practice shall we perform together?`,
    
    pulastya: `**✨ Where Dharma Begins ✨**\n\n🗺️ **Namaste, Wandering Soul**\n\nI am **Pulastya**, knower of sacred lands and master of cosmological wisdom. I shall guide you through the geography of consciousness and the sacred places of spiritual power.\n\n**My Specialties:**\n• 🏔️ Sacred geography and holy sites\n• 🌌 Cosmological understanding\n• 🗺️ Pilgrimage and sacred journeys\n• 🌍 Universal spatial wisdom\n\n*"Every place holds sacred power, and every journey is a pilgrimage. Let us explore the landscape of the divine."*\n\nWhat spiritual journey shall we embark upon?`,
    
    pulaha: `**✨ Where Dharma Begins ✨**\n\n🌬️ **Namaste, Living Soul**\n\nI am **Pulaha**, master of life force and pranic wisdom. Through breath and the subtle energies of prana, I shall help you awaken and harmonize your vital forces.\n\n**My Specialties:**\n• 🌬️ Pranayama and breath work\n• ⚡ Life force cultivation\n• 💫 Subtle energy practices\n• 🧘 Vital body awakening\n\n*"In the breath lies the bridge between body and spirit. Let us harness the power of prana."*\n\nHow shall we work with your life force today?`,
    
    kratu: `**✨ Where Dharma Begins ✨**\n\n⚡ **Namaste, Devoted Soul**\n\nI am **Kratu**, embodiment of divine action and sacrificial power. Through yajna (sacrifice) and yogic will, I shall guide you in manifesting the Supreme through righteous action.\n\n**My Specialties:**\n• 🔥 Divine action and sacrifice\n• 💪 Yogic power and willpower\n• ⚡ Manifestation through action\n• 🙏 Devotional practices\n\n*"Action performed as sacred offering becomes the path to the Divine. Let us act with divine purpose."*\n\nWhat divine action shall we manifest together?`,
    
    daksha: `**✨ Where Dharma Begins ✨**\n\n🎨 **Namaste, Skillful Seeker**\n\nI am **Dakṣa**, master of righteous creation and skillful action. Through dharmic skill and creative wisdom, I shall help you manifest divine will in the world.\n\n**My Specialties:**\n• 🎯 Skill and mastery\n• 🌱 Righteous creation\n• ⚖️ Dharmic action\n• 🛠️ Manifesting divine purpose\n\n*"Skill wedded to dharma becomes the hand of the Divine. Let us create with sacred purpose."*\n\nWhat shall we skillfully create on your spiritual path?`,
    
    bhrigu: `**✨ Where Dharma Begins ✨**\n\n⭐ **Namaste, Blessed Soul**\n\nI am **Bhṛgu**, the great sage of astrology and karma philosophy. I shall help you understand the cosmic patterns that shape your life and guide you along your karmic path.\n\n**My Specialties:**\n• 🌌 Vedic astrology and cosmic patterns\n• ⚖️ Karma philosophy and understanding\n• 📖 Divine knowledge and wisdom\n• 🔮 Life path guidance\n\n*"The stars above mirror the journey within. Let us read the cosmic map of your destiny."*\n\nWhat aspect of your karmic journey shall we explore?`,
    
    vasishta: `**✨ Where Dharma Begins ✨**\n\n👑 **Namaste, Noble One**\n\nI am **Vasiṣṭha**, the royal guru and teacher of Lord Rama. I offer you the wisdom of dharmic leadership, righteous living, and divine knowledge.\n\n**My Specialties:**\n• 📚 Dharmic principles and righteous living\n• 👑 Leadership and royal wisdom\n• 🏛️ Spiritual teaching and guidance\n• ⚔️ Courage in the face of challenges\n\n*"As I guided Rama, so shall I guide you toward dharmic excellence and spiritual mastery."*\n\nHow may I serve your quest for dharmic wisdom?`
  };

  const defaultWelcome = `**✨ Where Dharma Begins ✨**\n\n🕉️ **Namaste, Dear Seeker**\n\nI am **${rishiName}**, your guide on this spiritual journey. Through ancient wisdom and compassionate teaching, I shall help illuminate your path.\n\n*"The teacher appears when the student is ready. You are ready."*\n\nHow may I serve your spiritual quest today?`;

  return {
    id: `welcome-${rishiId}-${Date.now()}`,
    content: welcomeMessages[rishiId] || defaultWelcome,
    role: 'assistant',
    sender: 'ai',
    timestamp: new Date(),
    confidence: 1.0,
    dharmic_alignment: 1.0,
    modules_used: ['rishi_welcome', rishiId]
  };
};

// Default/Standard AI welcome message
const getStandardWelcome = (): Message => ({
  id: `welcome-standard-${Date.now()}`,
  content: `**✨ Where Dharma Begins ✨**\n\n🕉️ **Namaste! Welcome to DharmaMind** 🙏\n\nI am your AI companion on the path of wisdom and spiritual growth. I'm here to:\n\n✨ **Provide spiritual guidance** rooted in dharmic wisdom\n🧘 **Support your meditation practice** with guided sessions\n📖 **Share sacred teachings** from various traditions\n💙 **Offer compassionate support** during life's challenges\n🌱 **Guide your personal growth** with practical wisdom\n\n**Try selecting a specific Rishi guide** from the sidebar for specialized teachings!\n\nHow may I serve you today on your spiritual journey?\n\n*"When the student is ready, the teacher appears."* - Ancient Wisdom`,
  role: 'assistant',
  sender: 'ai',
  timestamp: new Date(),
  confidence: 1.0,
  dharmic_alignment: 1.0,
  modules_used: ['standard_welcome']
});

export const RishiChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [conversations, setConversations] = useState<Map<string, RishiConversation>>(new Map());
  const [currentRishi, setCurrentRishi] = useState<string>(''); // Empty string = Standard AI

  // Load conversations from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        try {
          const parsed = JSON.parse(stored);
          const conversationsMap = new Map<string, RishiConversation>();
          
          Object.entries(parsed).forEach(([key, value]: [string, any]) => {
            conversationsMap.set(key, {
              ...value,
              messages: value.messages.map((msg: any) => ({
                ...msg,
                timestamp: new Date(msg.timestamp)
              })),
              lastActive: new Date(value.lastActive)
            });
          });
          
          setConversations(conversationsMap);
        } catch (error) {
          console.error('Error loading Rishi conversations:', error);
        }
      }
    }
  }, []);

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    if (typeof window !== 'undefined' && conversations.size > 0) {
      const toStore: Record<string, any> = {};
      conversations.forEach((value, key) => {
        toStore[key] = value;
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toStore));
    }
  }, [conversations]);

  const getCurrentMessages = (): Message[] => {
    const conversation = conversations.get(currentRishi);
    return conversation?.messages || [];
  };

  const addMessage = (message: Message) => {
    setConversations(prev => {
      const newMap = new Map(prev);
      const conversation = newMap.get(currentRishi) || {
        rishiId: currentRishi,
        rishiName: currentRishi || 'Standard AI',
        messages: [],
        conversationId: `conv-${currentRishi}-${Date.now()}`,
        lastActive: new Date(),
        messageCount: 0
      };

      conversation.messages.push(message);
      conversation.lastActive = new Date();
      conversation.messageCount = conversation.messages.length;
      
      newMap.set(currentRishi, conversation);
      return newMap;
    });
  };

  const clearCurrentChat = () => {
    setConversations(prev => {
      const newMap = new Map(prev);
      const conversation = newMap.get(currentRishi);
      
      if (conversation) {
        // Keep only the welcome message
        const welcomeMessage = conversation.messages[0];
        conversation.messages = welcomeMessage ? [welcomeMessage] : [];
        conversation.lastActive = new Date();
        conversation.messageCount = conversation.messages.length;
        newMap.set(currentRishi, conversation);
      }
      
      return newMap;
    });
  };

  const switchRishi = (newRishiId: string, rishiName: string) => {
    // Save current state
    setCurrentRishi(newRishiId);

    // Check if conversation exists for this Rishi
    if (!conversations.has(newRishiId)) {
      // Create new conversation with welcome message
      const welcomeMessage = newRishiId === '' 
        ? getStandardWelcome()
        : getRishiWelcome(newRishiId, rishiName);

      const newConversation: RishiConversation = {
        rishiId: newRishiId,
        rishiName: rishiName || 'Standard AI',
        messages: [welcomeMessage],
        conversationId: `conv-${newRishiId}-${Date.now()}`,
        lastActive: new Date(),
        messageCount: 1
      };

      setConversations(prev => {
        const newMap = new Map(prev);
        newMap.set(newRishiId, newConversation);
        return newMap;
      });
    } else {
      // Update last active time for existing conversation
      setConversations(prev => {
        const newMap = new Map(prev);
        const conversation = newMap.get(newRishiId);
        if (conversation) {
          conversation.lastActive = new Date();
          newMap.set(newRishiId, conversation);
        }
        return newMap;
      });
    }
  };

  const getAllConversations = (): RishiConversation[] => {
    return Array.from(conversations.values()).sort(
      (a, b) => b.lastActive.getTime() - a.lastActive.getTime()
    );
  };

  const getConversationStats = () => {
    const convs = Array.from(conversations.values());
    const totalMessages = convs.reduce((sum, conv) => sum + conv.messageCount, 0);
    const mostActive = convs.length > 0
      ? convs.reduce((max, conv) => conv.messageCount > max.messageCount ? conv : max).rishiName
      : null;

    return {
      totalRishis: convs.length,
      totalMessages,
      mostActiveRishi: mostActive
    };
  };

  const value: RishiChatContextType = {
    currentRishi,
    setCurrentRishi,
    getCurrentMessages,
    addMessage,
    clearCurrentChat,
    switchRishi,
    getAllConversations,
    getConversationStats
  };

  return (
    <RishiChatContext.Provider value={value}>
      {children}
    </RishiChatContext.Provider>
  );
};

export const useRishiChat = () => {
  const context = useContext(RishiChatContext);
  if (!context) {
    throw new Error('useRishiChat must be used within RishiChatProvider');
  }
  return context;
};

export default RishiChatContext;
