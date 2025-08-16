import { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const { subscriptionId } = req.query;

  if (req.method === 'DELETE') {
    // Mock subscription cancellation
    res.status(200).json({ 
      success: true, 
      message: 'Subscription cancelled successfully',
      subscription: {
        id: subscriptionId,
        status: 'cancelled',
        cancelled_at: new Date().toISOString()
      }
    });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
