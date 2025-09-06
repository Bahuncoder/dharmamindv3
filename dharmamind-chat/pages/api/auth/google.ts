import { NextApiRequest, NextApiResponse } from 'next';

interface UserResponse {
  id: string;
  name: string;
  email: string;
  plan: 'free' | 'professional' | 'enterprise';
  provider: string;
  avatar?: string;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<UserResponse | { error: string }>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // TODO: Implement actual Google OAuth flow
    // For now, simulate Google authentication
    
    // In production, this would:
    // 1. Verify Google OAuth token
    // 2. Extract user information from Google
    // 3. Create or update user in database
    // 4. Generate JWT session token
    
    // Simulate Google user data
    const userData: UserResponse = {
      id: `google_${Date.now()}`,
      name: 'Google User',
      email: 'user@gmail.com',
      plan: 'professional', // Google users get pro trial
      provider: 'google',
      avatar: 'https://via.placeholder.com/40'
    };

    return res.status(200).json(userData);

  } catch (error) {
    console.error('Google authentication error:', error);
    return res.status(500).json({ error: 'Google authentication failed' });
  }
}
