import { NextApiRequest, NextApiResponse } from 'next';

// Helper function to validate JWT token
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
  // This is a simplified validation for demo purposes
  return { valid: false, error: 'Invalid token' };
}

// Mock payment methods API - in production this would connect to your actual backend
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

    if (req.method === 'GET') {
      // Return mock payment methods for the authenticated user
      const mockPaymentMethods = [
        {
          id: 'pm_1234567890',
          type: 'card',
          last_four: '4242',
          brand: 'visa',
          exp_month: 12,
          exp_year: 2025,
          default: true,
          user_id: user.id
        }
      ];

      res.status(200).json(mockPaymentMethods);
    } else if (req.method === 'POST') {
      const { method_type, card_details, billing_address, set_as_default } = req.body;

      if (!method_type || !card_details || !billing_address) {
        return res.status(400).json({ detail: 'Missing required payment method data' });
      }

      // Validate card details
      if (!card_details.number || !card_details.exp_month || !card_details.exp_year || !card_details.cvc) {
        return res.status(400).json({ detail: 'Invalid card details' });
      }

      // Mock payment method creation for authenticated user
      const newPaymentMethod = {
        payment_method_id: `pm_${Date.now()}`,
        type: method_type,
        last_four: card_details.number.slice(-4),
        brand: detectCardBrand(card_details.number),
        exp_month: card_details.exp_month,
        exp_year: card_details.exp_year,
        default: set_as_default || false,
        user_id: user.id,
        created_at: new Date().toISOString()
      };

      res.status(201).json(newPaymentMethod);
    } else {
      res.status(405).json({ detail: 'Method not allowed' });
    }
  } catch (error) {
    console.error('Payment methods API error:', error);
    res.status(500).json({ detail: 'Internal server error' });
  }
}

function detectCardBrand(cardNumber: string): string {
  const firstDigit = cardNumber[0];
  const firstTwoDigits = cardNumber.substring(0, 2);
  const firstFourDigits = cardNumber.substring(0, 4);

  if (firstDigit === '4') return 'visa';
  if (['51', '52', '53', '54', '55'].includes(firstTwoDigits)) return 'mastercard';
  if (['34', '37'].includes(firstTwoDigits)) return 'amex';
  if (firstFourDigits === '6011') return 'discover';
  
  return 'unknown';
}
