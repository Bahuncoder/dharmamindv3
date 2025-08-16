# DharmaMind Beta - Pure Chat App

A streamlined beta AI wisdom chat application - no onboarding friction, direct to chat experience.

## Beta Features

- **Direct-to-Chat**: Click demo link and start chatting immediately  
- **Zero Friction**: No setup, no onboarding, no barriers
- **Pure Chat Interface**: Focused AI wisdom conversations
- **Smart Routing**: Intelligent routing based on user status
- **Demo Mode**: Instant access without registration

## Beta Navigation Flow

- **/** → **Landing Page** (main entry point)
- **/?demo=true** → **Demo Chat** (instant access)
- **/landing** → Main marketing/entry page with demo and sign up options
- **/chat** → Main chat interface  
- **Logo Click** → Returns to `/landing`
- **Back to Home** → Returns to `/landing`

## Beta Architecture

This is a **pure chat application** with landing page as entry point:

1. **Demo Link**: `/?demo=true` → `/chat?demo=true` (instant access)
2. **Home/Logo**: All navigation goes to `/landing` (main entry point)
3. **Landing Page**: Shows demo and sign up options
4. **Beta Focus**: Landing page → Demo/Auth → Chat experience

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
