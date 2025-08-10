import { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Mock subscription plans for now
  const plans = [
    {
      id: 'free',
      name: 'Free Dharma',
      description: 'Basic spiritual guidance',
      tier: 'free',
      price: { monthly: 0, yearly: 0 },
      currency: 'USD',
      features: [
        {
          feature_id: 'basic_chat',
          name: 'Basic Chat',
          description: 'Basic spiritual conversations',
          included: true,
          usage_limit: 50
        }
      ],
      limits: {
        messages_per_month: 50,
        wisdom_modules: 3,
        api_requests_per_month: 100
      }
    },
    {
      id: 'professional',
      name: 'Professional Seeker',
      description: 'Enhanced spiritual guidance with advanced features',
      tier: 'professional',
      price: { monthly: 19.99, yearly: 199.99 },
      currency: 'USD',
      popular: true,
      features: [
        {
          feature_id: 'unlimited_chat',
          name: 'Unlimited Chat',
          description: 'Unlimited spiritual conversations',
          included: true
        },
        {
          feature_id: 'advanced_modules',
          name: 'Advanced Wisdom Modules',
          description: 'Access to all 32 wisdom modules',
          included: true
        }
      ],
      limits: {
        messages_per_month: -1,
        wisdom_modules: 32,
        api_requests_per_month: 10000
      }
    },
    {
      id: 'max',
      name: 'Dharma Max',
      description: 'Ultimate spiritual experience',
      tier: 'max',
      price: { monthly: 49.99, yearly: 499.99 },
      currency: 'USD',
      features: [
        {
          feature_id: 'everything',
          name: 'Everything Included',
          description: 'All features plus priority support',
          included: true
        }
      ],
      limits: {
        messages_per_month: -1,
        wisdom_modules: 32,
        api_requests_per_month: -1
      }
    }
  ];

  res.status(200).json({ success: true, data: plans });
}
