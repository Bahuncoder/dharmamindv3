# 🧘 DharmaMind Chat - Enterprise Security Edition

**🔐 MFA-Protected | 🧠 AI-Powered Spiritual Guidance | 📱 Progressive Web App**

A cutting-edge spiritual AI chat application with enterprise-grade security, featuring Multi-Factor Authentication, real-time threat protection, and advanced spiritual intelligence.

## 🌟 **Enterprise Features**

### 🔐 **Advanced Security**
- **Multi-Factor Authentication (MFA)**: TOTP support with QR code setup
- **Session Security**: Advanced token management with automatic expiration
- **Real-time Protection**: Connection to security monitoring system
- **Secure Headers**: Complete CSP and security header implementation
- **Input Validation**: Comprehensive XSS and injection prevention
- **Privacy First**: GDPR-compliant data handling and user controls

### 🧘 **Spiritual Intelligence**
- **Advanced AI Routing**: Intelligent spiritual guidance with multiple AI providers
- **Context Awareness**: Deep understanding of spiritual context and user journey
- **Sanskrit Integration**: Real-time Sanskrit translation and cultural context
- **Meditation Guidance**: Integrated meditation timers and practice tracking
- **Dharmic Wisdom**: Ancient wisdom teachings with modern accessibility
- **Personalized Experience**: Adaptive interface based on spiritual preferences

### 📱 **Modern User Experience**
- **Progressive Web App (PWA)**: Native app-like experience with offline support
- **Direct-to-Chat**: Zero friction access with optional MFA for security
- **Responsive Design**: Optimized for all devices with touch-friendly interface
- **Dark/Light Themes**: Intelligent theme switching for comfort
- **Accessibility**: WCAG 2.1 AA compliant with screen reader support
- **Performance Optimized**: Sub-second loading with intelligent caching

## 🔐 **Security-Enhanced Navigation Flow**

### **Authenticated Users:**
- **/** → **Landing Page** (main entry point with security status)
- **/chat** → **MFA Verification** → **Chat Interface** (secure access)
- **/profile** → **Security Settings** (MFA setup, device management)
- **/settings** → **Privacy Controls** (data management, preferences)

### **Demo Access:**
- **/?demo=true** → **Demo Chat** (limited access, no MFA required)
- **Demo Mode**: Read-only access with basic spiritual guidance

### **Security Flows:**
- **MFA Setup**: Guided wizard with QR code and backup code generation
- **Device Trust**: 30-day trusted device tokens for convenience
- **Session Management**: Automatic logout and re-authentication prompts

## Demo Link for Beta

```typescript
// Perfect beta sharing link
https://yourdomain.com/?demo=true

// Flow: Click link → Instant demo chat
/?demo=true → /chat?demo=true
```

## Architecture

- **Frontend**: Next.js with TypeScript
- **Styling**: Tailwind CSS
- **Authentication**: NextAuth.js with Google provider
- **State Management**: React Context
- **Chat Interface**: Real-time messaging with spiritual AI

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Environment Variables

Create a `.env.local` file with:

```env
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Chat Features

- **AI Wisdom Conversations**: Chat with dharma-guided AI
- **Chat History**: Save and access previous conversations
- **Demo Mode**: Try without registration
- **Mobile Responsive**: Works on all devices
- **Real-time Responses**: Immediate AI responses
- **Spiritual Guidance**: AI trained on dharmic principles

---

Built with ❤️ for conscious technology and spiritual growth.
