import { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Forward the request to the backend API
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    const response = await fetch(`${backendUrl}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();

    // Forward the response status and data
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('Registration proxy error:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Internal server error' 
    });
  }
}
