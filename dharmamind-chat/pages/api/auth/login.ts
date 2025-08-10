import { NextApiRequest, NextApiResponse } from 'next';

interface LoginRequest {
  email: string;
  password: string;
  name?: string;
  isLogin: boolean;
}

interface UserResponse {
  id: string;
  name: string;
  email: string;
  plan: 'free' | 'professional' | 'enterprise';
  provider?: string;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<UserResponse | { error: string }>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { email, password, name, isLogin }: LoginRequest = req.body;

    // TODO: Connect to actual backend/database
    // For now, simulate authentication
    
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    if (!isLogin && !name) {
      return res.status(400).json({ error: 'Name is required for registration' });
    }

    // Simulate different user types based on email domain
    let plan: 'free' | 'professional' | 'enterprise' = 'free';
    if (email.includes('enterprise') || email.includes('company')) {
      plan = 'enterprise';
    } else if (email.includes('pro') || email.includes('business')) {
      plan = 'professional';
    }

    // Simulate successful authentication
    const userData: UserResponse = {
      id: `user_${Date.now()}`,
      name: name || email.split('@')[0],
      email,
      plan,
      provider: 'email'
    };

    // TODO: In production, implement:
    // - Password hashing and verification
    // - Database user lookup/creation
    // - JWT token generation
    // - Session management
    // - Email verification
    // - Rate limiting

    return res.status(200).json(userData);

  } catch (error) {
    console.error('Authentication error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
