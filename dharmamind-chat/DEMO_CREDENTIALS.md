# DharmaMind Demo & Test Accounts

## Quick Access URLs

### Demo Mode (No Login Required)
- **URL**: `http://localhost:3000/chat?demo=true`
- **Features**: Full access to all 7 Rishis, Standard AI, conversation history
- **Limitations**: Data stored in sessionStorage (cleared on browser close)
- **Perfect for**: Quick testing, showcasing features

### Landing Page
- **URL**: `http://localhost:3000`
- **Auto-redirects to**: `/chat?demo=true` for unauthenticated users

## Test Accounts for Login Testing

### Basic Plan User
```
Email: demo@dharmamind.com
Password: DharmaDemo2024!
Plan: Basic
Features:
- Standard AI access
- 1 Rishi (Atri) available
- Limited features
```

### Pro Plan User
```
Email: pro@dharmamind.com
Password: DharmaPro2024!
Plan: Pro
Features:
- Standard AI access
- 3 Rishis available (Atri, Bhrigu, Vashishta)
- Advanced analytics
- Priority support
```

### Max Plan User
```
Email: max@dharmamind.com
Password: DharmaMax2024!
Plan: Max
Features:
- Standard AI access
- All 7 Rishis available
- Full analytics dashboard
- Premium support
- Unlimited conversations
```

### Enterprise Plan User
```
Email: enterprise@dharmamind.com
Password: DharmaEnt2024!
Plan: Enterprise
Features:
- Everything in Max
- Custom Rishi training
- API access
- White-label options
- Dedicated support
```

## Testing Scenarios

### 1. Test Subscription Tiers
1. Login with Basic account → See limited Rishi access
2. Login with Pro account → See 3 Rishis unlocked
3. Login with Max account → See all 7 Rishis
4. Click on locked Rishi → See upgrade prompt

### 2. Test Rishi Switching
1. Select Atri → Chat about meditation
2. Select Bhrigu → See separate conversation
3. Switch back to Atri → Original conversation restored
4. Click "← Standard" → Return to Standard AI mode

### 3. Test Conversation Persistence
1. Chat with a Rishi
2. Refresh the page
3. Verify conversation is restored from localStorage
4. Switch to different Rishi
5. Return to first Rishi → Conversation still there

### 4. Test Welcome Messages
1. Open fresh session
2. Verify "✨ Where Dharma Begins ✨" appears
3. Select each Rishi
4. Verify unique welcome message for each

### 5. Test Standard AI Mode
1. Start in Standard AI mode (default)
2. Select a Rishi
3. Click "← Standard" button
4. Verify transition back to Standard AI
5. Check that Standard AI conversation is separate

## Creating Real Test Accounts

### Option 1: Direct Database Insert (If you have DB access)
```sql
INSERT INTO users (email, password_hash, first_name, last_name, subscription_plan, verified)
VALUES 
  ('demo@dharmamind.com', '$2b$10$...', 'Demo', 'User', 'basic', true),
  ('pro@dharmamind.com', '$2b$10$...', 'Pro', 'User', 'pro', true),
  ('max@dharmamind.com', '$2b$10$...', 'Max', 'User', 'max', true),
  ('enterprise@dharmamind.com', '$2b$10$...', 'Enterprise', 'User', 'enterprise', true);
```

### Option 2: Through API (Recommended)
Use the registration API endpoint:

```bash
# Create Basic user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@dharmamind.com",
    "password": "DharmaDemo2024!",
    "firstName": "Demo",
    "lastName": "User",
    "plan": "basic"
  }'

# Create Pro user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "pro@dharmamind.com",
    "password": "DharmaPro2024!",
    "firstName": "Pro",
    "lastName": "User",
    "plan": "pro"
  }'
```

### Option 3: Through UI
1. Go to `/auth?mode=signup`
2. Fill registration form
3. Use email verification code from backend logs
4. Manually update subscription plan in database

## Environment Setup

### Required Environment Variables
```env
# Frontend (.env.local)
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DHARMALLM_API_URL=http://localhost:8001

# Backend
DATABASE_URL=postgresql://user:password@localhost:5432/dharmamind
JWT_SECRET=your-jwt-secret
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## Quick Test Workflow

### Morning Checklist ☀️
```bash
# 1. Start development server
npm run dev

# 2. Open demo mode
open http://localhost:3000/chat?demo=true

# 3. Test Rishi switching
# - Select Atri
# - Chat: "Tell me about meditation"
# - Select Bhrigu
# - Chat: "What's my karmic path?"
# - Click "← Standard"
# - Verify separate conversations
```

### Feature Testing 🧪
1. **Standard AI ↔ Rishi Switching**
   - Verify transition animation plays
   - Check conversation separation
   - Test "← Standard" button

2. **Subscription Features**
   - Login with different plans
   - Verify Rishi access restrictions
   - Test upgrade prompts

3. **Conversation Persistence**
   - Chat, refresh, verify restoration
   - Test localStorage data
   - Multiple Rishis same session

4. **UI/UX Elements**
   - Gradient cards for active Rishi
   - Welcome messages
   - Transition animations
   - Mobile responsiveness

## Troubleshooting

### Standard AI button not working?
- Check browser console for errors
- Verify `handleRishiSelect('')` is being called
- Check if transition animation is blocking interaction
- Inspect `selectedRishi` state in React DevTools

### Login not working?
- Verify backend is running on port 8000
- Check NEXTAUTH_SECRET is set
- Verify database connection
- Check backend logs for errors

### Rishi conversations mixing?
- Clear localStorage: `localStorage.clear()`
- Refresh page
- Check RishiChatContext is properly wrapped in _app.tsx
- Verify unique conversation keys per Rishi

### Transition animation stuck?
- Check `showTransition` state
- Verify `pendingRishi` is set correctly
- Look for JavaScript errors in console
- Try clearing component state

## Support

For issues or questions:
1. Check browser console for errors
2. Review backend logs
3. Verify all environment variables are set
4. Test in incognito mode (fresh session)
5. Check React DevTools for state issues

---

**Last Updated**: November 12, 2025
**DharmaMind Version**: 2.0 (Rishi Integration)
