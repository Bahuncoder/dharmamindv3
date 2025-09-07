# DharmaMind Community Platform

A Next.js-powered community platform for DharmaMind users to connect, share wisdom, and grow together on their spiritual journey.

## 🌟 Features

### 🏠 **Community Hub**
- **Real-time Activity Feed**: Live updates on community discussions, member joins, achievements
- **Discussion Forums**: Organized by categories like Meditation, Dharma Wisdom, Daily Life
- **Member Profiles**: User profiles with badges, specialties, and contribution tracking
- **Online Status**: Real-time online/offline member indicators

### 🔔 **Notification System**
- **Smart Notifications**: Contextual alerts for community activities
- **Notification Center**: Centralized notification management
- **Read/Unread Tracking**: Keep track of notification status
- **Action Buttons**: Quick actions directly from notifications

### 📊 **Community Analytics**
- **Engagement Metrics**: Member activity, discussion participation
- **Growth Tracking**: New member statistics and trends
- **Content Analytics**: Post engagement, comment threads
- **Admin Dashboard**: Comprehensive management interface

### 🛡️ **Moderation & Security**
- **Content Moderation**: Pending review system for posts and comments
- **User Management**: Admin tools for member management
- **Secure Authentication**: Integration with main DharmaMind platform
- **Session Management**: Secure session handling and cleanup

### 🎨 **Enhanced UI/UX**
- **Centralized Color System**: Consistent design language
- **Responsive Design**: Mobile-first approach
- **Dark/Light Modes**: User preference system
- **Smooth Animations**: Framer Motion powered interactions

## 🚀 Getting Started

### Prerequisites
- Node.js 18.0.0 or higher
- npm 8.0.0 or higher

### Installation

1. **Install Dependencies**
   ```bash
   cd DhramaMind_Community
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```
   The community platform will be available at `http://localhost:3002`

3. **Build for Production**
   ```bash
   npm run build
   npm start
   ```

### Available Scripts

- `npm run dev` - Start development server on port 3002
- `npm run build` - Build the application for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint for code quality
- `npm run type-check` - Run TypeScript type checking

## 📁 Project Structure

```
DhramaMind_Community/
├── components/           # Reusable React components
│   ├── Navigation.tsx           # Main navigation component
│   ├── CommunityDiscussions.tsx # Discussion forum interface
│   ├── CommunityActivityFeed.tsx # Real-time activity feed
│   ├── NotificationCenter.tsx   # Notification management
│   ├── EnhancedCommunityDashboard.tsx # Main dashboard
│   └── ...
├── contexts/            # React Context providers
│   ├── ColorContext.tsx        # Theme and color management
│   ├── ToastContext.tsx        # Toast notifications
│   └── NotificationContext.tsx # Community notifications
├── pages/              # Next.js pages
│   ├── index.tsx              # Home page
│   ├── community.tsx          # Main community page
│   ├── feed.tsx              # Community feed with sections
│   ├── insights.tsx          # Analytics and insights
│   ├── blog.tsx              # Community blog
│   └── admin/
│       └── dashboard.tsx     # Admin management panel
├── services/           # API and external services
│   └── communityAuth.ts      # Authentication service
├── styles/            # CSS and styling
│   ├── globals.css           # Global styles
│   └── colors.css           # Centralized color system
├── utils/             # Utility functions
│   ├── cn.ts                # Class name utilities
│   └── secureStorage.ts     # Secure storage management
└── js/               # Configuration
    └── config.js            # Application configuration
```

## 🎯 Key Pages

### **Community Dashboard** (`/community`)
- Main community hub with activity overview
- Member statistics and engagement metrics
- Quick access to discussions and events
- Real-time community updates

### **Discussion Feed** (`/feed`)
- Interactive discussion interface
- Category-based filtering
- Search functionality
- Member engagement features

### **Community Insights** (`/insights`)
- Real-time activity monitoring
- Member activity tracking
- Community growth analytics
- Online member directory

### **Admin Dashboard** (`/admin/dashboard`)
- User management interface
- Content moderation tools
- Community analytics
- System administration

## 🔧 Configuration

### Environment Variables
Create a `.env.local` file with:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_CENTRAL_AUTH_URL=https://dharmamind.ai
NEXT_PUBLIC_COMMUNITY_API_URL=http://localhost:8000
```

### Integration with Main Platform
The community platform integrates seamlessly with the main DharmaMind application:
- **Shared Authentication**: Single sign-on with main platform
- **User Sync**: Automatic user profile synchronization
- **Cross-Platform Navigation**: Smooth transitions between applications
- **Unified Design**: Consistent user experience

## 🎨 Design System

### Color Palette
- **Primary**: Saffron Orange (#F2A300) - Spiritual warmth
- **Secondary**: Emerald Green (#32A370) - Growth and wisdom
- **Neutrals**: Professional grays for content
- **Status Colors**: Success, warning, error, and info states

### Typography
- **Font Family**: Inter (Google Fonts)
- **Font Weights**: 300-700 for various hierarchies
- **Font Sizes**: Responsive scale for all devices

## 🔐 Security Features

- **Authentication Integration**: Secure auth with main platform
- **Session Management**: Automatic cleanup and validation
- **Input Sanitization**: XSS protection for user content
- **Rate Limiting**: API protection against abuse
- **Secure Storage**: Encrypted local storage for sensitive data

## 📱 Responsive Design

- **Mobile-First**: Optimized for mobile devices
- **Tablet Support**: Enhanced layouts for tablets
- **Desktop Experience**: Full-featured desktop interface
- **Cross-Browser**: Compatible with all modern browsers

## 🚀 Performance

- **Next.js Optimization**: Automatic code splitting and optimization
- **Image Optimization**: Built-in Next.js image optimization
- **Caching Strategy**: Efficient caching for better performance
- **Bundle Analysis**: Optimized bundle sizes

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for all new components
3. Add proper error handling and loading states
4. Test responsive design across devices
5. Document new features and components

## 📄 License

MIT License - see the main project LICENSE file for details.

---

## 🌸 About DharmaMind

DharmaMind Community is part of the larger DharmaMind ecosystem - an AI platform that combines ancient dharma wisdom with modern artificial intelligence to provide ethical guidance and spiritual insights.

**Connect with us:**
- 🌐 Main Platform: [dharmamind.ai](https://dharmamind.ai)
- 💬 Chat Interface: Available in the main application
- 👥 Community: This platform for connection and growth
- 📚 Knowledge Base: Extensive dharma and mindfulness resources

Join our community of mindful practitioners and experience the integration of wisdom and technology! 🙏
