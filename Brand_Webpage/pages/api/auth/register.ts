import type { NextApiRequest, NextApiResponse } from 'next';

// TODO: Replace with actual database
const users: { id: string; name: string; email: string; password: string; plan: string }[] = [];

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { name, email, password } = req.body;

    // Validation
    if (!name || !email || !password) {
      return res.status(400).json({ message: 'All fields are required' });
    }

    if (password.length < 8) {
      return res.status(400).json({ message: 'Password must be at least 8 characters' });
    }

    // Check if user already exists (in real app, check database)
    const existingUser = users.find(u => u.email === email);
    if (existingUser) {
      return res.status(400).json({ message: 'Email already registered' });
    }

    // Create new user (in real app, hash password and save to database)
    const newUser = {
      id: `user-${Date.now()}`,
      name,
      email,
      password, // TODO: Hash this!
      plan: 'free',
    };
    users.push(newUser);

    // Return success (don't include password)
    return res.status(201).json({
      user: {
        id: newUser.id,
        name: newUser.name,
        email: newUser.email,
        plan: newUser.plan,
      },
    });
  } catch (error) {
    console.error('Registration error:', error);
    return res.status(500).json({ message: 'Internal server error' });
  }
}
