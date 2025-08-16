import { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const { subscriptionId } = req.query;

  if (req.method === 'GET') {
    // Mock subscription details
    const subscription = {
      id: subscriptionId,
      subscription_id: subscriptionId,
      user_id: 'user_123',
      plan_id: 'free',
      plan_name: 'Free Dharma',
      status: 'active',
      billing_interval: 'monthly',
      amount: 0,
      currency: 'USD',
      current_period_start: new Date().toISOString(),
      current_period_end: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
      cancel_at_period_end: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    res.status(200).json({ success: true, data: subscription });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
