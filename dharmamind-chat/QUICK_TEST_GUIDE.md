# Quick Testing Guide

## 1. Standard AI Button Issue - DEBUGGING

I've added console logging to help debug. When you click "← Standard":

**Test Steps:**
1. Go to: http://localhost:3000/chat?demo=true
2. Open DevTools Console (F12)
3. Select any Rishi (e.g., Atri)
4. Click "← Standard" button in header
5. Look for console messages:
   - 🔄 handleRishiSelect called
   - ✨ Triggering transition
6. Share what you see!

## 2. Demo Login Accounts

### No Login Required (Works Now)
**Demo Mode**: http://localhost:3000/chat?demo=true

### Login Page (Requires Backend)
**URL**: http://localhost:3000/auth?mode=login

You'll see a blue banner with test accounts:
- Basic: demo@dharmamind.com
- Pro: pro@dharmamind.com  
- Max: max@dharmamind.com
- Password (all): DharmaDemo2024!

⚠️ **Note**: Accounts must exist in your database first!

## 3. What's New

✅ Professional Rishi selector with gradients
✅ "← Standard" button (top-right of selector)
✅ Demo credentials banner on login page
✅ Debug logging for button clicks
✅ Full documentation in DEMO_CREDENTIALS.md

## 4. Next Step

Please test the "← Standard" button and share the console output!
