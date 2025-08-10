# 🧩 DharmaMind Component Library

## Overview
This document provides detailed examples and usage guidelines for all DharmaMind UI components.

---

## 🔘 Button Components

### Basic Button Usage

```jsx
// Primary button for main actions
<Button variant="primary" size="md">
  Get Started
</Button>

// Secondary button for alternative actions
<Button variant="secondary" size="md">
  Learn More
</Button>

// Outline button for subtle actions
<Button variant="outline" size="sm">
  Cancel
</Button>
```

### Button Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `variant` | `'primary' \| 'secondary' \| 'outline' \| 'ghost'` | `'primary'` | Button style variant |
| `size` | `'xs' \| 'sm' \| 'md' \| 'lg' \| 'xl'` | `'md'` | Button size |
| `disabled` | `boolean` | `false` | Disable button interaction |
| `loading` | `boolean` | `false` | Show loading spinner |
| `onClick` | `function` | - | Click handler |

### Button Implementation

```jsx
import React from 'react';
import { cn } from '../utils/cn';

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  disabled?: boolean;
  loading?: boolean;
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  children,
  className,
  onClick,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2';
  
  const variantClasses = {
    primary: 'bg-gray-500 text-charcoal border-2 border-emerald-500 hover:bg-gray-600 hover:transform hover:-translate-y-0.5 hover:shadow-lg',
    secondary: 'bg-transparent text-charcoal border-2 border-emerald-500 hover:bg-gray-100 hover:transform hover:-translate-y-0.5',
    outline: 'bg-transparent text-charcoal border border-gray-300 hover:bg-gray-50 hover:border-emerald-500',
    ghost: 'bg-transparent text-charcoal hover:bg-gray-100'
  };
  
  const sizeClasses = {
    xs: 'px-2 py-1 text-xs rounded',
    sm: 'px-3 py-1.5 text-sm rounded-md',
    md: 'px-4 py-2 text-base rounded-lg',
    lg: 'px-6 py-3 text-lg rounded-lg',
    xl: 'px-8 py-4 text-xl rounded-xl'
  };

  return (
    <button
      className={cn(
        baseClasses,
        variantClasses[variant],
        sizeClasses[size],
        disabled && 'opacity-50 cursor-not-allowed',
        className
      )}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
          <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" className="opacity-75" />
        </svg>
      )}
      {children}
    </button>
  );
};

export default Button;
```

---

## 💬 Chat Components

### Message Bubble

```jsx
interface MessageProps {
  type: 'user' | 'ai';
  content: string;
  timestamp?: Date;
  avatar?: string;
}

const MessageBubble: React.FC<MessageProps> = ({ type, content, timestamp, avatar }) => {
  const isUser = type === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} items-end max-w-[80%]`}>
        {avatar && (
          <div className="flex-shrink-0 mx-2">
            <img src={avatar} alt="Avatar" className="w-8 h-8 rounded-full" />
          </div>
        )}
        <div className={`px-4 py-2 rounded-2xl ${
          isUser 
            ? 'bg-emerald-500 text-white rounded-br-sm' 
            : 'bg-gray-100 text-charcoal border-l-4 border-emerald-500 rounded-bl-sm'
        }`}>
          <p className="text-sm leading-relaxed">{content}</p>
          {timestamp && (
            <p className={`text-xs mt-1 ${isUser ? 'text-emerald-100' : 'text-gray-500'}`}>
              {timestamp.toLocaleTimeString()}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
```

### Chat Input

```jsx
interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled, placeholder = "Type a message..." }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSend(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 bg-white border-t border-gray-200">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="flex-1 px-4 py-2 bg-white border-2 border-gray-200 rounded-full text-charcoal placeholder-gray-400 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500 focus:ring-opacity-20"
      />
      <Button
        type="submit"
        variant="primary"
        size="md"
        disabled={disabled || !message.trim()}
        className="rounded-full px-6"
      >
        Send
      </Button>
    </form>
  );
};
```

---

## 📦 Card Components

### Content Card

```jsx
interface CardProps {
  children: React.ReactNode;
  hover?: boolean;
  padding?: 'sm' | 'md' | 'lg';
  className?: string;
}

const Card: React.FC<CardProps> = ({ children, hover = true, padding = 'md', className }) => {
  const paddingClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };

  return (
    <div className={cn(
      'bg-white border border-gray-200 rounded-xl shadow-sm',
      hover && 'hover:shadow-lg hover:transform hover:-translate-y-1 transition-all duration-200',
      paddingClasses[padding],
      className
    )}>
      {children}
    </div>
  );
};
```

### Feature Card

```jsx
interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  accent?: boolean;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, description, accent = false }) => {
  return (
    <Card className={accent ? 'bg-gray-500 text-white border-emerald-500 border-2' : ''}>
      <div className="text-center">
        <div className="w-12 h-12 mx-auto mb-4 flex items-center justify-center text-2xl">
          {icon}
        </div>
        <h3 className={`text-xl font-semibold mb-2 ${accent ? 'text-white' : 'text-charcoal'}`}>
          {title}
        </h3>
        <p className={`leading-relaxed ${accent ? 'text-gray-100' : 'text-gray-600'}`}>
          {description}
        </p>
      </div>
    </Card>
  );
};
```

### Quote Card

```jsx
interface QuoteCardProps {
  quote: string;
  author?: string;
  role?: string;
}

const QuoteCard: React.FC<QuoteCardProps> = ({ quote, author, role }) => {
  return (
    <div className="bg-gray-50 border-l-4 border-emerald-500 rounded-lg p-6">
      <blockquote className="text-lg text-charcoal leading-relaxed mb-4">
        "{quote}"
      </blockquote>
      {author && (
        <div className="text-sm text-gray-600">
          <div className="font-medium">{author}</div>
          {role && <div>{role}</div>}
        </div>
      )}
    </div>
  );
};
```

---

## 📝 Form Components

### Input Field

```jsx
interface InputProps {
  label?: string;
  error?: string;
  help?: string;
  required?: boolean;
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
}

const Input: React.FC<InputProps> = ({ 
  label, 
  error, 
  help, 
  required, 
  type = 'text',
  placeholder,
  value,
  onChange,
  ...props 
}) => {
  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm font-medium text-charcoal">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <input
        type={type}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        placeholder={placeholder}
        className={cn(
          'w-full px-3 py-2 bg-white border rounded-lg text-charcoal placeholder-gray-400',
          'focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500',
          error ? 'border-red-500' : 'border-gray-300'
        )}
        {...props}
      />
      {help && !error && (
        <p className="text-sm text-gray-600">{help}</p>
      )}
      {error && (
        <p className="text-sm text-red-500">{error}</p>
      )}
    </div>
  );
};
```

### Textarea

```jsx
interface TextareaProps {
  label?: string;
  error?: string;
  help?: string;
  required?: boolean;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  rows?: number;
}

const Textarea: React.FC<TextareaProps> = ({ 
  label, 
  error, 
  help, 
  required,
  placeholder,
  value,
  onChange,
  rows = 4,
  ...props 
}) => {
  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm font-medium text-charcoal">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <textarea
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        className={cn(
          'w-full px-3 py-2 bg-white border rounded-lg text-charcoal placeholder-gray-400 resize-none',
          'focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500',
          error ? 'border-red-500' : 'border-gray-300'
        )}
        {...props}
      />
      {help && !error && (
        <p className="text-sm text-gray-600">{help}</p>
      )}
      {error && (
        <p className="text-sm text-red-500">{error}</p>
      )}
    </div>
  );
};
```

---

## 🎨 Logo Component

### Logo Implementation

```jsx
interface LogoProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'avatar';
  showText?: boolean;
  onClick?: () => void;
  className?: string;
}

const Logo: React.FC<LogoProps> = ({ 
  size = 'md', 
  showText = true, 
  onClick, 
  className = '' 
}) => {
  const sizeClasses = {
    xs: { container: 'w-6 h-6', text: 'text-xs' },
    sm: { container: 'w-8 h-8', text: 'text-sm' },
    md: { container: 'w-10 h-10', text: 'text-lg' },
    lg: { container: 'w-12 h-12', text: 'text-xl' },
    xl: { container: 'w-16 h-16', text: 'text-2xl' },
    avatar: { container: 'w-8 h-8', text: 'text-sm' }
  };

  const currentSize = sizeClasses[size];
  
  const LogoContent = () => (
    <>
      <div className={`${currentSize.container} rounded-lg overflow-hidden shadow-lg bg-white border border-gray-200 relative`}>
        <img
          src="/logo.jpeg"
          alt="DharmaMind Logo"
          className="w-full h-full object-contain"
        />
        {/* Emerald accent bar */}
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-emerald-500"></div>
      </div>
      {showText && (
        <span className={`font-bold text-charcoal ${currentSize.text} ml-3 tracking-tight`}>
          DharmaMind
        </span>
      )}
    </>
  );

  if (onClick) {
    return (
      <button 
        onClick={onClick}
        className={`flex items-center hover:opacity-80 transition-opacity ${className}`}
      >
        <LogoContent />
      </button>
    );
  }

  return (
    <div className={`flex items-center ${className}`}>
      <LogoContent />
    </div>
  );
};
```

---

## 🔄 Loading Components

### Spinner

```jsx
interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'accent' | 'white';
}

const Spinner: React.FC<SpinnerProps> = ({ size = 'md', color = 'accent' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  const colorClasses = {
    primary: 'border-gray-200 border-t-gray-600',
    accent: 'border-gray-200 border-t-emerald-500',
    white: 'border-gray-300 border-t-white'
  };

  return (
    <div className={`${sizeClasses[size]} ${colorClasses[color]} border-2 rounded-full animate-spin`} />
  );
};
```

### Loading Button

```jsx
const LoadingButton: React.FC<ButtonProps & { loading?: boolean }> = ({ 
  loading, 
  children, 
  disabled,
  ...props 
}) => {
  return (
    <Button disabled={disabled || loading} {...props}>
      {loading && <Spinner size="sm" color="white" className="mr-2" />}
      {children}
    </Button>
  );
};
```

---

## 🚨 Alert Components

### Alert

```jsx
interface AlertProps {
  type: 'success' | 'warning' | 'error' | 'info';
  title?: string;
  children: React.ReactNode;
  onClose?: () => void;
}

const Alert: React.FC<AlertProps> = ({ type, title, children, onClose }) => {
  const typeClasses = {
    success: 'bg-green-50 border-green-200 text-green-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    error: 'bg-red-50 border-red-200 text-red-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800'
  };

  const icons = {
    success: '✅',
    warning: '⚠️',
    error: '❌',
    info: 'ℹ️'
  };

  return (
    <div className={`${typeClasses[type]} border rounded-lg p-4`}>
      <div className="flex">
        <div className="flex-shrink-0">
          <span className="text-lg">{icons[type]}</span>
        </div>
        <div className="ml-3 flex-1">
          {title && (
            <h4 className="font-medium mb-1">{title}</h4>
          )}
          <div className="text-sm">{children}</div>
        </div>
        {onClose && (
          <div className="ml-auto pl-3">
            <button
              onClick={onClose}
              className="inline-flex text-gray-400 hover:text-gray-600"
            >
              <span className="sr-only">Close</span>
              ✕
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
```

---

## 📱 Layout Components

### Container

```jsx
interface ContainerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  children: React.ReactNode;
  className?: string;
}

const Container: React.FC<ContainerProps> = ({ size = 'lg', children, className }) => {
  const sizeClasses = {
    sm: 'max-w-2xl',
    md: 'max-w-4xl',
    lg: 'max-w-6xl',
    xl: 'max-w-7xl',
    full: 'max-w-full'
  };

  return (
    <div className={cn('mx-auto px-4 sm:px-6 lg:px-8', sizeClasses[size], className)}>
      {children}
    </div>
  );
};
```

### Section

```jsx
interface SectionProps {
  background?: 'white' | 'gray' | 'accent';
  padding?: 'sm' | 'md' | 'lg' | 'xl';
  children: React.ReactNode;
  className?: string;
}

const Section: React.FC<SectionProps> = ({ 
  background = 'white', 
  padding = 'lg', 
  children, 
  className 
}) => {
  const backgroundClasses = {
    white: 'bg-white',
    gray: 'bg-gray-50',
    accent: 'bg-gray-500 text-white'
  };

  const paddingClasses = {
    sm: 'py-8',
    md: 'py-12',
    lg: 'py-16',
    xl: 'py-24'
  };

  return (
    <section className={cn(backgroundClasses[background], paddingClasses[padding], className)}>
      {children}
    </section>
  );
};
```

---

## 🎯 Usage Examples

### Complete Chat Interface

```jsx
const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSendMessage = async (content: string) => {
    const userMessage = { type: 'user', content, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const aiMessage = { type: 'ai', content: 'Thank you for your message!', timestamp: new Date() };
      setMessages(prev => [...prev, aiMessage]);
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <Logo size="sm" />
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((message, index) => (
          <MessageBubble key={index} {...message} />
        ))}
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 rounded-2xl p-4">
              <Spinner size="sm" />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput onSend={handleSendMessage} disabled={loading} />
    </div>
  );
};
```

### Landing Page Hero

```jsx
const HeroSection = () => {
  return (
    <Section background="gray" padding="xl">
      <Container size="lg">
        <div className="text-center">
          <Logo size="xl" className="mx-auto mb-8" />
          <h1 className="text-4xl md:text-6xl font-bold text-charcoal mb-6">
            AI-Powered Spiritual Guidance
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Discover wisdom from ancient traditions with modern AI technology
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="primary" size="lg">
              Get Started Free
            </Button>
            <Button variant="secondary" size="lg">
              Learn More
            </Button>
          </div>
        </div>
      </Container>
    </Section>
  );
};
```

---

*Component Library v2.0.0 - Last updated: August 8, 2025*
