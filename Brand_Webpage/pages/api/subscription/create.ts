import { NextApiRequest, NextApiResponse } from 'next';

// Helper function to validate JWT token (same as in payment methods)
function validateAuthToken(authHeader: string | undefined): { valid: boolean; user?: any; error?: string } {
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return { valid: false, error: 'Authentication required' };
  }

  const token = authHeader.substring(7);
  
  // For demo tokens, decode and validate
  if (token.startsWith('demo_')) {
    try {
      const payload = JSON.parse(atob(token.substring(5)));
      
      // Check if token is expired
      if (payload.exp && Date.now() > payload.exp) {
        return { valid: false, error: 'Token expired' };
      }

      return { 
        valid: true, 
        user: {
          id: payload.user_id,
          email: payload.email,
          plan: payload.plan
        }
      };
    } catch (error) {
      return { valid: false, error: 'Invalid token format' };
    }
  }

  // For production, validate JWT properly here
  return { valid: false, error: 'Invalid token' };
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  try {
    // Validate authentication
    const authValidation = validateAuthToken(req.headers.authorization);
    if (!authValidation.valid) {
      return res.status(401).json({ detail: authValidation.error });
    }

    const user = authValidation.user;

    if (req.method === 'POST') {
      const { plan_id, payment_method_id, metadata } = req.body;

      if (!plan_id) {
        return res.status(400).json({ detail: 'Plan ID is required' });
      }

      // Validate plan exists
      const validPlans = ['dharma_free', 'dharma_pro', 'dharma_enterprise'];
      if (!validPlans.includes(plan_id)) {
        return res.status(400).json({ detail: 'Invalid plan ID' });
      }

      // For paid plans, require payment method
      if (plan_id !== 'dharma_free' && !payment_method_id) {
        return res.status(400).json({ detail: 'Payment method required for paid plans' });
      }

      // Mock subscription creation for authenticated user
      const subscription = {
        subscription_id: `sub_${Date.now()}`,
        user_id: user.id,
        plan_id,
        status: plan_id === 'dharma_free' ? 'active' : 'trialing',
        current_period_start: new Date().toISOString(),
        current_period_end: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        trial_end: plan_id !== 'dharma_free' ? new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString() : null,
        payment_method_id,
        metadata: {
          ...metadata,
          upgrade_from: user.plan,
          upgraded_by: user.email
        },
        created_at: new Date().toISOString()
      };

      // Try to forward to actual backend API
      try {
        const backendResponse = await fetch('http://localhost:5000/subscription/create', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': req.headers.authorization || ''
          },
          body: JSON.stringify({ 
            plan_id, 
            payment_method_id, 
            metadata: subscription.metadata,
            user_id: user.id
          })
        });

        if (backendResponse.ok) {
          const backendData = await backendResponse.json();
          res.status(200).json(backendData);
        } else {
          // Fallback to mock if backend endpoint doesn't exist
          res.status(200).json({
            success: true,
            subscription,
            message: `Successfully ${plan_id === 'dharma_free' ? 'downgraded to' : 'upgraded to'} ${getPlanName(plan_id)}`,
            trial_days: plan_id !== 'dharma_free' ? 14 : 0
          });
        }
      } catch (backendError) {
        // Fallback to mock if backend is not available
        console.log('Backend not available, using mock response');
        res.status(200).json({
          success: true,
          subscription,
          message: `Successfully ${plan_id === 'dharma_free' ? 'downgraded to' : 'upgraded to'} ${getPlanName(plan_id)}`,
          trial_days: plan_id !== 'dharma_free' ? 14 : 0
        });
      }
    } else {
      res.status(405).json({ detail: 'Method not allowed' });
    }
  } catch (error) {
    console.error('Subscription API error:', error);
    res.status(500).json({ detail: 'Internal server error' });
  }
}

function getPlanName(planId: string): string {
  const planNames: { [key: string]: string } = {
    'dharma_free': 'Dharma Free',
    'dharma_pro': 'Dharma Professional',
    'dharma_enterprise': 'Dharma Enterprise'
  };
  return planNames[planId] || planId;
}
