import type { NextApiRequest, NextApiResponse } from 'next';

type Data = {
  success: boolean;
  message: string;
  reset_token?: string;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Data>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, message: 'Method not allowed' });
  }

  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ success: false, message: 'Email is required' });
    }

    // Forward request to backend
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/auth/forgot-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email }),
    });

    const data = await response.json();

    if (response.ok) {
      return res.status(200).json(data);
    } else {
      return res.status(response.status).json({
        success: false,
        message: data.detail || 'Failed to send reset email'
      });
    }
  } catch (error) {
    console.error('Forgot password API error:', error);
    return res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
}
