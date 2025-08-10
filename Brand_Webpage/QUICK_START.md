# 🚀 DharmaMind Quick Start Guide

## Installation & Setup

### 1. Copy Color System
Copy the core CSS variables to your main stylesheet:

```css
/* styles/globals.css */
:root {
  /* Text Colors */
  --text-primary: #2C2C2C;     /* Dark charcoal for main text */
  --text-secondary: #6E6E6E;   /* Medium gray for secondary text */
  --text-muted: #9ca3af;       /* Light gray for muted text */
  --text-inverse: #ffffff;     /* White for dark backgrounds */
  
  /* UI Colors */
  --primary: #6b7280;          /* Gray-500 for main UI elements */
  --primary-hover: #4b5563;    /* Gray-600 for hover states */
  --accent: #10b981;           /* Emerald-500 for accents */
  --accent-hover: #059669;     /* Emerald-600 for accent hover */
  
  /* Backgrounds */
  --bg-primary: #ffffff;       /* Pure white */
  --bg-secondary: #f9fafb;     /* Very light gray */
  --bg-tertiary: #f3f4f6;      /* Light gray */
  
  /* Borders */
  --border-light: #e5e7eb;     /* Light gray borders */
  --border-medium: #d1d5db;    /* Medium gray borders */
  
  /* States */
  --success: #10b981;          /* Success green */
  --warning: #f59e0b;          /* Warning amber */
  --error: #ef4444;            /* Error red */
  --info: #3b82f6;             /* Info blue */
}

/* Apply to body */
body {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  font-family: 'Inter', system-ui, sans-serif;
}
```

### 2. Create Utility Classes

```css
/* Utility classes for quick styling */

/* Text Colors */
.text-primary { color: var(--text-primary); }
.text-secondary { color: var(--text-secondary); }
.text-muted { color: var(--text-muted); }
.text-inverse { color: var(--text-inverse); }

/* Backgrounds */
.bg-primary { background-color: var(--bg-primary); }
.bg-secondary { background-color: var(--bg-secondary); }
.bg-tertiary { background-color: var(--bg-tertiary); }

/* Accent colors */
.text-accent { color: var(--accent); }
.bg-accent { background-color: var(--accent); }
.border-accent { border-color: var(--accent); }

/* Button base styles */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 500;
  border-radius: 0.5rem;
  transition: all 0.2s ease;
  cursor: pointer;
  border: 2px solid;
  text-decoration: none;
}

.btn:focus {
  outline: none;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
}

/* Button variants */
.btn-primary {
  background-color: var(--primary);
  color: var(--text-primary);
  border-color: var(--accent);
  padding: 0.75rem 1.5rem;
}

.btn-primary:hover {
  background-color: var(--primary-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.btn-secondary {
  background-color: transparent;
  color: var(--text-primary);
  border-color: var(--accent);
  padding: 0.75rem 1.5rem;
}

.btn-secondary:hover {
  background-color: var(--bg-tertiary);
  transform: translateY(-1px);
}

/* Card styles */
.card {
  background-color: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.card:hover {
  box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
  transition: all 0.2s ease;
}

/* Form styles */
.form-input {
  width: 100%;
  padding: 0.75rem 1rem;
  background-color: var(--bg-primary);
  border: 2px solid var(--border-light);
  border-radius: 0.5rem;
  color: var(--text-primary);
  font-size: 1rem;
}

.form-input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.form-label {
  display: block;
  font-weight: 500;
  font-size: 0.875rem;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
}
```

---

## 📱 Basic Chat App Implementation

### HTML Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DharmaMind Chat</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="chat-app">
        <!-- Header -->
        <header class="chat-header">
            <div class="logo-container">
                <img src="logo.jpg" alt="Logo" class="logo-image">
                <div class="logo-accent"></div>
            </div>
            <h1>DharmaMind</h1>
        </header>

        <!-- Messages Container -->
        <div class="messages-container" id="messages">
            <!-- Messages will be added here -->
        </div>

        <!-- Input Area -->
        <div class="input-container">
            <input type="text" id="messageInput" class="form-input" placeholder="Type your message...">
            <button onclick="sendMessage()" class="btn btn-primary">Send</button>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>
```

### CSS Styles
```css
/* styles.css */

/* Include the CSS variables and utilities from above, then add: */

.chat-app {
  max-width: 800px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: var(--bg-secondary);
}

.chat-header {
  background-color: var(--bg-primary);
  padding: 1rem;
  border-bottom: 1px solid var(--border-light);
  display: flex;
  align-items: center;
  gap: 1rem;
}

.logo-container {
  width: 40px;
  height: 40px;
  border-radius: 0.5rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-light);
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.logo-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.logo-accent {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 4px;
  background-color: var(--accent);
}

.chat-header h1 {
  color: var(--text-primary);
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  max-width: 80%;
  padding: 0.75rem 1rem;
  border-radius: 1rem;
  font-size: 0.875rem;
  line-height: 1.5;
}

.message-user {
  background-color: var(--accent);
  color: var(--text-inverse);
  align-self: flex-end;
  border-bottom-right-radius: 0.25rem;
}

.message-ai {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  align-self: flex-start;
  border-bottom-left-radius: 0.25rem;
  border-left: 4px solid var(--accent);
}

.input-container {
  background-color: var(--bg-primary);
  padding: 1rem;
  border-top: 1px solid var(--border-light);
  display: flex;
  gap: 0.5rem;
}

.input-container .form-input {
  flex: 1;
  border-radius: 2rem;
}

.input-container .btn {
  border-radius: 2rem;
  padding: 0.75rem 1.5rem;
  white-space: nowrap;
}

/* Responsive design */
@media (max-width: 768px) {
  .chat-app {
    height: 100vh;
    margin: 0;
    border-radius: 0;
  }
  
  .message {
    max-width: 90%;
  }
  
  .input-container {
    padding: 0.75rem;
  }
}
```

### JavaScript Functionality
```javascript
// script.js

let messages = [];

function addMessage(content, type = 'user') {
    const message = {
        content: content,
        type: type,
        timestamp: new Date()
    };
    
    messages.push(message);
    renderMessages();
    scrollToBottom();
}

function renderMessages() {
    const container = document.getElementById('messages');
    container.innerHTML = '';
    
    messages.forEach(message => {
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${message.type}`;
        messageEl.textContent = message.content;
        container.appendChild(messageEl);
    });
}

function sendMessage() {
    const input = document.getElementById('messageInput');
    const content = input.value.trim();
    
    if (!content) return;
    
    // Add user message
    addMessage(content, 'user');
    input.value = '';
    
    // Simulate AI response (replace with actual API call)
    setTimeout(() => {
        const responses = [
            "Thank you for your message. How can I help you today?",
            "That's an interesting question. Let me think about that...",
            "I understand your concern. Here's what I would suggest...",
            "Based on your input, I would recommend..."
        ];
        
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        addMessage(randomResponse, 'ai');
    }, 1000);
}

function scrollToBottom() {
    const container = document.getElementById('messages');
    container.scrollTop = container.scrollHeight;
}

// Allow Enter key to send message
document.getElementById('messageInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Add some initial messages
addMessage("Hello! Welcome to DharmaMind. How can I assist you today?", 'ai');
```

---

## 🎨 Quick Component Examples

### Button Examples
```html
<!-- Primary button -->
<button class="btn btn-primary">Get Started</button>

<!-- Secondary button -->
<button class="btn btn-secondary">Learn More</button>

<!-- Small button -->
<button class="btn btn-primary" style="padding: 0.5rem 1rem; font-size: 0.875rem;">
    Small Action
</button>
```

### Card Examples
```html
<!-- Basic card -->
<div class="card">
    <h3 style="color: var(--text-primary); margin-bottom: 0.5rem;">Card Title</h3>
    <p style="color: var(--text-secondary);">Card content goes here...</p>
</div>

<!-- Feature card with accent -->
<div class="card" style="background-color: var(--primary); color: var(--text-inverse); border: 2px solid var(--accent);">
    <div style="font-size: 2rem; margin-bottom: 1rem;">🚀</div>
    <h3 style="margin-bottom: 0.5rem;">Feature Title</h3>
    <p>Feature description...</p>
</div>
```

### Form Examples
```html
<!-- Input field -->
<div style="margin-bottom: 1rem;">
    <label class="form-label">Your Name</label>
    <input type="text" class="form-input" placeholder="Enter your name">
</div>

<!-- Textarea -->
<div style="margin-bottom: 1rem;">
    <label class="form-label">Message</label>
    <textarea class="form-input" rows="4" placeholder="Enter your message..."></textarea>
</div>
```

---

## 🎯 Color Usage Guidelines

### Do's ✅
- Use `#2C2C2C` for all primary text
- Use `#6E6E6E` for secondary text  
- Use `#10b981` (emerald) for accents and highlights
- Use `#6b7280` (gray) for UI elements
- Use `#f9fafb` for page backgrounds
- Use white `#ffffff` for content areas

### Don'ts ❌
- Don't use orange or amber colors
- Don't use pure black `#000000` for text
- Don't mix different accent colors
- Don't use bright or neon colors

### Quick Reference
```css
/* Copy-paste color values */
Primary Text: #2C2C2C
Secondary Text: #6E6E6E  
Accent Color: #10b981
UI Elements: #6b7280
Page Background: #f9fafb
Content Background: #ffffff
Light Borders: #e5e7eb
```

---

## 📱 Mobile Optimization

Add these responsive utilities:

```css
/* Mobile-first approach */
@media (max-width: 640px) {
    .btn {
        width: 100%;
        justify-content: center;
    }
    
    .card {
        margin: 0.5rem;
        padding: 1rem;
    }
    
    .input-container {
        flex-direction: column;
        gap: 0.75rem;
    }
}

@media (max-width: 480px) {
    .chat-header h1 {
        font-size: 1.25rem;
    }
    
    .message {
        max-width: 95%;
        font-size: 0.8rem;
    }
}
```

---

## ✅ Checklist

- [ ] Copy CSS variables to your stylesheet
- [ ] Add utility classes for buttons, cards, forms
- [ ] Implement responsive design
- [ ] Test on mobile devices
- [ ] Verify color contrast accessibility
- [ ] Add hover and focus states
- [ ] Test with actual content

---

**Ready to build!** 🚀 

This quick start guide gives you everything needed to create a chat app with the same professional DharmaMind design system.

*Quick Start Guide v1.0 - Last updated: August 8, 2025*
