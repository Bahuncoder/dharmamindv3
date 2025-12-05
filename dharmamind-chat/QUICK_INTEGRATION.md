# 🚀 Quick Integration Guide: Rishi-Specific Chats

## What You Need to Do

To integrate the new Rishi-specific chat system into your existing chat interface, follow these simple steps:

---

## Step 1: Update Your Chat Component

In `pages/chat.tsx` or wherever you handle Rishi selection:

### Replace the old Rishi selection handler:

**BEFORE:**
```tsx
const handleRishiSelect = (rishiId: string) => {
  setSelectedRishi(rishiId);
  // Messages just stay mixed together ❌
};
```

**AFTER:**
```tsx
import { useRishiChat } from '../contexts/RishiChatContext';
import { RishiTransition } from '../components/RishiTransition';

const {
  currentRishi,
  switchRishi,
  getCurrentMessages,
  addMessage
} = useRishiChat();

const [showTransition, setShowTransition] = useState(false);
const [pendingRishi, setPendingRishi] = useState({ id: '', name: '' });

const handleRishiSelect = (rishiId: string) => {
  const rishi = availableRishis.find(r => r.id === rishiId);
  const rishiName = rishi?.name || 'Standard AI';
  
  // Show transition animation
  setPendingRishi({ id: rishiId, name: rishiName });
  setShowTransition(true);
};

// Add the transition component to your JSX
<RishiTransition
  show={showTransition}
  fromRishi={currentRishi}
  toRishi={pendingRishi.id}
  toRishiName={pendingRishi.name}
  onComplete={() => {
    // Actually switch the Rishi after animation
    switchRishi(pendingRishi.id, pendingRishi.name);
    setShowTransition(false);
  }}
/>
```

---

## Step 2: Use Rishi-Specific Messages

### Replace message state management:

**BEFORE:**
```tsx
const [messages, setMessages] = useState<Message[]>([...]);

// Messages shared across all Rishis ❌
```

**AFTER:**
```tsx
// Get messages for current Rishi only ✅
const messages = getCurrentMessages();

// When adding a new message:
const userMessage = {
  id: `msg-${Date.now()}`,
  content: inputValue,
  role: 'user' as const,
  timestamp: new Date()
};

addMessage(userMessage); // Adds to current Rishi's conversation

// When receiving AI response:
const assistantMessage = {
  id: `msg-${Date.now() + 1}`,
  content: response.message,
  role: 'assistant' as const,
  timestamp: new Date(),
  confidence: response.confidence,
  dharmic_alignment: response.dharmic_alignment
};

addMessage(assistantMessage); // Also adds to current Rishi's conversation
```

---

## Step 3: Optional - Add Context Badge

Show which Rishi is active:

```tsx
import { RishiContextBadge } from '../components/RishiTransition';

<RishiContextBadge
  rishiId={currentRishi}
  rishiName={availableRishis.find(r => r.id === currentRishi)?.name || 'Standard AI'}
  messageCount={getCurrentMessages().length}
  onSwitch={() => setSidebarOpen(true)} // Open Rishi selector
/>
```

---

## Step 4: Optional - Add Conversation History

Show all Rishi conversations in sidebar:

```tsx
import { ConversationHistory } from '../components/RishiTransition';

const { getAllConversations } = useRishiChat();

<ConversationHistory
  conversations={getAllConversations().map(conv => ({
    rishiId: conv.rishiId,
    rishiName: conv.rishiName,
    messageCount: conv.messageCount,
    lastActive: conv.lastActive
  }))}
  currentRishi={currentRishi}
  onSelect={(rishiId) => {
    const rishi = availableRishis.find(r => r.id === rishiId);
    handleRishiSelect(rishiId);
  }}
/>
```

---

## Complete Example

Here's a minimal working example:

```tsx
import React, { useState } from 'react';
import { useRishiChat } from '../contexts/RishiChatContext';
import { RishiTransition, RishiContextBadge } from '../components/RishiTransition';
import { RishiSelector } from '../components/RishiSelector';

const ChatPage = () => {
  const {
    currentRishi,
    switchRishi,
    getCurrentMessages,
    addMessage
  } = useRishiChat();

  const [inputValue, setInputValue] = useState('');
  const [showTransition, setShowTransition] = useState(false);
  const [pendingRishi, setPendingRishi] = useState({ id: '', name: '' });

  const availableRishis = [
    { id: 'atri', name: 'Atri', specialization: ['Meditation'], available: true },
    { id: 'bhrigu', name: 'Bhrigu', specialization: ['Astrology'], available: true },
    // ... more Rishis
  ];

  const handleRishiSelect = (rishiId: string) => {
    const rishi = availableRishis.find(r => r.id === rishiId);
    setPendingRishi({ id: rishiId, name: rishi?.name || 'Standard AI' });
    setShowTransition(true);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    // Add user message
    addMessage({
      id: `msg-${Date.now()}`,
      content: inputValue,
      role: 'user',
      timestamp: new Date()
    });

    setInputValue('');

    // Get AI response (your existing logic)
    const response = await fetchAIResponse(inputValue, currentRishi);

    // Add AI response
    addMessage({
      id: `msg-${Date.now() + 1}`,
      content: response.message,
      role: 'assistant',
      timestamp: new Date()
    });
  };

  const messages = getCurrentMessages();

  return (
    <div className="chat-page">
      {/* Rishi Transition Animation */}
      <RishiTransition
        show={showTransition}
        fromRishi={currentRishi}
        toRishi={pendingRishi.id}
        toRishiName={pendingRishi.name}
        onComplete={() => {
          switchRishi(pendingRishi.id, pendingRishi.name);
          setShowTransition(false);
        }}
      />

      {/* Sidebar with Rishi Selector */}
      <aside className="sidebar">
        <RishiSelector
          selectedRishi={currentRishi}
          availableRishis={availableRishis}
          onRishiSelect={handleRishiSelect}
          userSubscription="basic"
        />
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Context Badge */}
        <div className="chat-header">
          <RishiContextBadge
            rishiId={currentRishi}
            rishiName={availableRishis.find(r => r.id === currentRishi)?.name || 'Standard AI'}
            messageCount={messages.length}
          />
        </div>

        {/* Messages */}
        <div className="messages">
          {messages.map(message => (
            <div key={message.id} className={`message ${message.role}`}>
              {message.content}
            </div>
          ))}
        </div>

        {/* Input */}
        <div className="chat-input">
          <input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          <button onClick={handleSendMessage}>Send</button>
        </div>
      </main>
    </div>
  );
};

export default ChatPage;
```

---

## Testing Checklist

After integration, test these scenarios:

- [ ] Click a Rishi → See transition animation
- [ ] Send message to Atri → Message appears
- [ ] Switch to Bhrigu → See Bhrigu's welcome, Atri's messages gone
- [ ] Send message to Bhrigu → Message appears
- [ ] Switch back to Atri → Original Atri messages are back!
- [ ] Refresh browser → Messages persist
- [ ] Context badge shows correct Rishi name
- [ ] Message counts are accurate

---

## Benefits You'll See Immediately

✅ **Clear Separation**: Each Rishi has their own conversation  
✅ **Beautiful Transitions**: Professional UI feedback  
✅ **No More Confusion**: Users know which Rishi they're talking to  
✅ **Persistent History**: All conversations saved automatically  
✅ **Easy Switching**: Jump between Rishis without losing context  

---

## Common Pitfalls to Avoid

❌ **DON'T** use separate `useState` for messages  
✅ **DO** use `getCurrentMessages()` from the hook

❌ **DON'T** manually manage Rishi state  
✅ **DO** use `switchRishi()` method

❌ **DON'T** forget to wrap app with `RishiChatProvider`  
✅ **DO** add provider in `_app.tsx` (already done!)

❌ **DON'T** clear all messages on Rishi switch  
✅ **DO** let the context handle message filtering

---

## Need Help?

Check these files for reference:
- `contexts/RishiChatContext.tsx` - Core logic
- `components/RishiTransition.tsx` - UI components
- `RISHI_CHAT_SYSTEM.md` - Full documentation

---

**That's it! The system is ready to use.** 🎉

The hard work is done - now you just need to use the hooks and components in your chat interface!

🕉️ **May your Rishi conversations be clear and enlightening!**
