import { NextApiRequest, NextApiResponse } from 'next';
import { getServerSession } from 'next-auth';
import { getToken } from 'next-auth/jwt';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get session for authenticated users
    const token = await getToken({ req });
    
    // Allow demo mode to work without authentication
    const isDemo = req.query.demo === 'true' || req.headers.referer?.includes('demo=true');
    
    if (!token && !isDemo) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    // For now, return fallback data since backend integration is complex
    // In production, this would call the actual backend API
    const fallbackRishis = [
      {
        id: 'patanjali',
        name: 'Sage Patanjali',
        sanskrit: 'महर्षि पतञ्जलि',
        specialization: ['Yoga', 'Meditation', 'Mind Control', 'Concentration'],
        greeting: '🕉️ Namaste! I am Sage Patanjali, compiler of the Yoga Sutras. I can guide you in meditation, concentration, and the path of yoga. How may I assist you in stilling your mind today?',
        available: true,
        requires_upgrade: false
      },
      {
        id: 'vyasa',
        name: 'Sage Vyasa',
        sanskrit: 'महर्षि व्यास',
        specialization: ['Dharma', 'Life Guidance', 'Bhagavad Gita', 'Mahabharata'],
        greeting: '🕉️ Namaste! I am Sage Vyasa, composer of the Mahabharata and compiler of the Vedas. I can guide you through life\'s challenges using the wisdom of dharma. What weighs on your heart today?',
        available: isDemo ? true : false, // Make all available in demo mode
        requires_upgrade: isDemo ? false : true
      },
      {
        id: 'valmiki',
        name: 'Sage Valmiki',
        sanskrit: 'महर्षि वाल्मीकि',
        specialization: ['Devotion', 'Transformation', 'Ramayana', 'Surrender'],
        greeting: '🕉️ Namaste! I am Sage Valmiki, author of the Ramayana. I can guide you in devotion, transformation, and surrender to the divine. How can I help you on your spiritual journey?',
        available: isDemo ? true : false, // Make all available in demo mode
        requires_upgrade: isDemo ? false : true
      }
    ];

    res.status(200).json({
      available_rishis: fallbackRishis,
      user_subscription: isDemo ? 'demo' : 'basic'
    });
  } catch (error) {
    console.error('Error fetching available Rishis:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}
