import React, { useState } from 'react';
import { useColors } from '../contexts/ColorContext';
import { paymentAPI, PaymentMethod, CreatePaymentMethodRequest, BillingAddress } from '../services/paymentAPI';

interface PaymentFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPaymentMethodAdded: (paymentMethod: PaymentMethod) => void;
}

interface PaymentFormData {
  // Payment method type
  paymentType: 'credit_card' | 'debit_card' | 'bank_transfer' | 'paypal' | 'stripe';
  
  // Card details (for credit/debit cards)
  cardNumber: string;
  expiryDate: string;
  cvc: string;
  cardholderName: string;
  
  // Bank transfer details
  bankName: string;
  accountNumber: string;
  routingNumber: string;
  accountHolderName: string;
  accountType: 'checking' | 'savings';
  
  // PayPal details
  paypalEmail: string;
  
  // Stripe details (for Connect accounts)
  stripeAccountId: string;
  
  // Common details
  billingAddress: BillingAddress;
  saveForFuture: boolean;
  setAsDefault: boolean;
}

const PaymentFormModal: React.FC<PaymentFormModalProps> = ({
  isOpen,
  onClose,
  onPaymentMethodAdded
}) => {
  const { currentTheme } = useColors();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<PaymentFormData>({
    paymentType: 'credit_card',
    cardNumber: '',
    expiryDate: '',
    cvc: '',
    cardholderName: '',
    bankName: '',
    accountNumber: '',
    routingNumber: '',
    accountHolderName: '',
    accountType: 'checking',
    paypalEmail: '',
    stripeAccountId: '',
    billingAddress: {
      line1: '',
      line2: '',
      city: '',
      state: '',
      postal_code: '',
      country: 'US'
    },
    saveForFuture: true,
    setAsDefault: false
  });

  if (!isOpen) return null;

  const handleInputChange = (field: string, value: string | boolean) => {
    if (field.startsWith('billingAddress.')) {
      const addressField = field.replace('billingAddress.', '');
      setFormData((prev: PaymentFormData) => ({
        ...prev,
        billingAddress: {
          ...prev.billingAddress,
          [addressField]: value
        }
      }));
    } else {
      setFormData((prev: PaymentFormData) => ({
        ...prev,
        [field]: value
      }));
    }
  };

  const formatCardNumber = (value: string) => {
    const digits = value.replace(/\D/g, '');
    return digits.replace(/(\d{4})(?=\d)/g, '$1 ');
  };

  const formatExpiryDate = (value: string) => {
    const digits = value.replace(/\D/g, '');
    if (digits.length >= 2) {
      return digits.substring(0, 2) + '/' + digits.substring(2, 4);
    }
    return digits;
  };

  const validateForm = (): string[] => {
    const errors: string[] = [];

    // Common validation
    if (!formData.billingAddress.line1.trim()) {
      errors.push('Please enter billing address');
    }
    if (!formData.billingAddress.city.trim()) {
      errors.push('Please enter billing city');
    }
    if (!formData.billingAddress.state.trim()) {
      errors.push('Please enter billing state');
    }
    if (!formData.billingAddress.postal_code.trim()) {
      errors.push('Please enter billing postal code');
    }

    // Payment method specific validation
    switch (formData.paymentType) {
      case 'credit_card':
      case 'debit_card':
        if (!formData.cardNumber || formData.cardNumber.replace(/\s/g, '').length < 13) {
          errors.push('Please enter a valid card number');
        }
        if (!formData.expiryDate || formData.expiryDate.length !== 5) {
          errors.push('Please enter a valid expiry date (MM/YY)');
        }
        if (!formData.cvc || formData.cvc.length < 3) {
          errors.push('Please enter a valid CVC');
        }
        if (!formData.cardholderName.trim()) {
          errors.push('Please enter the cardholder name');
        }
        break;

      case 'bank_transfer':
        if (!formData.bankName.trim()) {
          errors.push('Please enter bank name');
        }
        if (!formData.accountNumber.trim()) {
          errors.push('Please enter account number');
        }
        if (!formData.routingNumber.trim() || formData.routingNumber.length !== 9) {
          errors.push('Please enter a valid 9-digit routing number');
        }
        if (!formData.accountHolderName.trim()) {
          errors.push('Please enter account holder name');
        }
        break;

      case 'paypal':
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!formData.paypalEmail.trim() || !emailRegex.test(formData.paypalEmail)) {
          errors.push('Please enter a valid PayPal email address');
        }
        break;

      case 'stripe':
        if (!formData.stripeAccountId.trim()) {
          errors.push('Please enter Stripe account ID');
        }
        break;
    }

    return errors;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const validationErrors = validateForm();
    if (validationErrors.length > 0) {
      setError(validationErrors.join(', '));
      return;
    }

    setSubmitting(true);

    try {
      let paymentMethodRequest: CreatePaymentMethodRequest;

      switch (formData.paymentType) {
        case 'credit_card':
        case 'debit_card':
          const [expMonth, expYear] = formData.expiryDate.split('/');
          const fullYear = parseInt(`20${expYear}`);

          paymentMethodRequest = {
            type: formData.paymentType,
            card_number: formData.cardNumber.replace(/\s/g, ''),
            exp_month: parseInt(expMonth),
            exp_year: fullYear,
            cvc: formData.cvc,
            cardholder_name: formData.cardholderName,
            billing_address: formData.billingAddress,
            save_for_future: formData.saveForFuture,
            set_as_default: formData.setAsDefault
          };
          break;

        case 'bank_transfer':
          paymentMethodRequest = {
            type: 'bank_transfer',
            billing_address: formData.billingAddress,
            save_for_future: formData.saveForFuture,
            set_as_default: formData.setAsDefault,
            // Add bank-specific fields as custom metadata
            bank_name: formData.bankName,
            account_number: formData.accountNumber,
            routing_number: formData.routingNumber,
            account_holder_name: formData.accountHolderName,
            account_type: formData.accountType
          } as any; // Type assertion needed for additional fields
          break;

        case 'paypal':
          paymentMethodRequest = {
            type: 'paypal',
            billing_address: formData.billingAddress,
            save_for_future: formData.saveForFuture,
            set_as_default: formData.setAsDefault,
            paypal_email: formData.paypalEmail
          } as any; // Type assertion needed for additional fields
          break;

        case 'stripe':
          paymentMethodRequest = {
            type: 'stripe' as any, // Stripe type not in original enum
            billing_address: formData.billingAddress,
            save_for_future: formData.saveForFuture,
            set_as_default: formData.setAsDefault,
            stripe_account_id: formData.stripeAccountId
          } as any; // Type assertion needed for additional fields
          break;

        default:
          throw new Error('Invalid payment method type');
      }

      const newPaymentMethod = await paymentAPI.createPaymentMethod(paymentMethodRequest);
      onPaymentMethodAdded(newPaymentMethod);
      onClose();
    } catch (error) {
      console.error('Failed to add payment method:', error);
      setError(error instanceof Error ? error.message : 'Failed to add payment method');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-overlay backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="modal-content rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="p-6 border-b border-medium sticky top-0 modal-content">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-primary">Add Payment Method</h2>
            <button
              onClick={onClose}
              className="text-tertiary hover:text-secondary"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Form */}
        <div className="p-6">
          {error && (
            <div className="mb-6 bg-error-bg border border-error rounded-lg p-4">
              <div className="flex">
                <svg className="w-5 h-5 text-error" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <p className="ml-3 text-sm text-error">{error}</p>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Payment Method Type Selection */}
            <div>
              <h3 className="font-medium text-primary mb-4">Payment Method Type</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { value: 'credit_card', label: 'Credit Card', icon: '💳' },
                  { value: 'debit_card', label: 'Debit Card', icon: '💳' },
                  { value: 'bank_transfer', label: 'Bank Transfer', icon: '🏦' },
                  { value: 'paypal', label: 'PayPal', icon: '🅿️' },
                  { value: 'stripe', label: 'Stripe', icon: '💰' }
                ].map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => handleInputChange('paymentType', option.value)}
                    className={`p-4 border-2 rounded-lg text-center transition-all ${
                      formData.paymentType === option.value
                        ? 'border-accent bg-accent-bg text-accent'
                        : 'border-medium hover:border-secondary'
                    }`}
                  >
                    <div className="text-2xl mb-2">{option.icon}</div>
                    <div className="text-sm font-medium">{option.label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Card Information (for credit/debit cards) */}
            {(formData.paymentType === 'credit_card' || formData.paymentType === 'debit_card') && (
              <div>
                <h3 className="font-medium text-primary mb-4">Card Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-secondary mb-2">
                      Cardholder Name *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.cardholderName}
                      onChange={(e) => handleInputChange('cardholderName', e.target.value)}
                      className="input-primary w-full px-3 py-2 border rounded-lg"
                      placeholder="John Doe"
                    />
                  </div>
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-secondary mb-2">
                      Card Number *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.cardNumber}
                      onChange={(e) => handleInputChange('cardNumber', formatCardNumber(e.target.value))}
                      className="input-primary w-full px-3 py-2 border rounded-lg"
                      placeholder="1234 5678 9012 3456"
                      maxLength={19}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-secondary mb-2">
                      Expiry Date *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.expiryDate}
                      onChange={(e) => handleInputChange('expiryDate', formatExpiryDate(e.target.value))}
                      className="input-primary w-full px-3 py-2 border rounded-lg"
                      placeholder="MM/YY"
                      maxLength={5}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-secondary mb-2">
                      CVC *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.cvc}
                      onChange={(e) => handleInputChange('cvc', e.target.value.replace(/\D/g, ''))}
                      className="input-primary w-full px-3 py-2 border rounded-lg"
                      placeholder="123"
                      maxLength={4}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Bank Transfer Information */}
            {formData.paymentType === 'bank_transfer' && (
              <div>
                <h3 className="font-medium text-primary mb-4">Bank Transfer Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-primary mb-2">
                      Bank Name *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.bankName}
                      onChange={(e) => handleInputChange('bankName', e.target.value)}
                      className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                      placeholder="Chase Bank, Wells Fargo, etc."
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-primary mb-2">
                      Account Holder Name *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.accountHolderName}
                      onChange={(e) => handleInputChange('accountHolderName', e.target.value)}
                      className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                      placeholder="John Doe"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-primary mb-2">
                      Account Type *
                    </label>
                    <select
                      required
                      value={formData.accountType}
                      onChange={(e) => handleInputChange('accountType', e.target.value)}
                      className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    >
                      <option value="checking">Checking Account</option>
                      <option value="savings">Savings Account</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-primary mb-2">
                      Routing Number *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.routingNumber}
                      onChange={(e) => handleInputChange('routingNumber', e.target.value.replace(/\D/g, '').substring(0, 9))}
                      className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                      placeholder="123456789"
                      maxLength={9}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-primary mb-2">
                      Account Number *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.accountNumber}
                      onChange={(e) => handleInputChange('accountNumber', e.target.value.replace(/\D/g, ''))}
                      className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                      placeholder="1234567890"
                    />
                  </div>
                </div>
                <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex">
                    <svg className="w-5 h-5 text-blue-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-blue-800">Bank Transfer Information</h3>
                      <p className="text-sm text-blue-700 mt-1">
                        Bank transfers typically take 3-5 business days to process. Your account will be verified before the first transfer.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* PayPal Information */}
            {formData.paymentType === 'paypal' && (
              <div>
                <h3 className="font-medium text-primary mb-4">PayPal Information</h3>
                <div>
                  <label className="block text-sm font-medium text-primary mb-2">
                    PayPal Email Address *
                  </label>
                  <input
                    type="email"
                    required
                    value={formData.paypalEmail}
                    onChange={(e) => handleInputChange('paypalEmail', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    placeholder="your-email@example.com"
                  />
                </div>
                <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex">
                    <svg className="w-5 h-5 text-blue-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-blue-800">PayPal Integration</h3>
                      <p className="text-sm text-blue-700 mt-1">
                        You'll be redirected to PayPal to authorize payments from your account. Instant processing available.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Stripe Information */}
            {formData.paymentType === 'stripe' && (
              <div>
                <h3 className="font-medium text-primary mb-4">Stripe Connect Account</h3>
                <div>
                  <label className="block text-sm font-medium text-primary mb-2">
                    Stripe Account ID *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.stripeAccountId}
                    onChange={(e) => handleInputChange('stripeAccountId', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    placeholder="acct_xxxxxxxxxx"
                  />
                </div>
                <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex">
                    <svg className="w-5 h-5 text-blue-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-blue-800">Stripe Connect</h3>
                      <p className="text-sm text-blue-700 mt-1">
                        Connect your existing Stripe account for seamless payment processing. Professional-grade security and reporting.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Billing Address */}
            <div>
              <h3 className="font-medium text-primary mb-4">Billing Address</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-primary mb-2">
                    Address Line 1 *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.line1}
                    onChange={(e) => handleInputChange('billingAddress.line1', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    placeholder="123 Main St"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-primary mb-2">
                    Address Line 2
                  </label>
                  <input
                    type="text"
                    value={formData.billingAddress.line2}
                    onChange={(e) => handleInputChange('billingAddress.line2', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    placeholder="Apt, suite, etc."
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-primary mb-2">
                    City *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.city}
                    onChange={(e) => handleInputChange('billingAddress.city', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    placeholder="New York"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-primary mb-2">
                    State *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.state}
                    onChange={(e) => handleInputChange('billingAddress.state', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    placeholder="NY"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-primary mb-2">
                    Postal Code *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.postal_code}
                    onChange={(e) => handleInputChange('billingAddress.postal_code', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    placeholder="10001"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-primary mb-2">
                    Country *
                  </label>
                  <select
                    required
                    value={formData.billingAddress.country}
                    onChange={(e) => handleInputChange('billingAddress.country', e.target.value)}
                    className="w-full px-3 py-2 border border-brand-accent rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  >
                    <option value="US">United States</option>
                    <option value="CA">Canada</option>
                    <option value="GB">United Kingdom</option>
                    <option value="AU">Australia</option>
                    <option value="DE">Germany</option>
                    <option value="FR">France</option>
                    <option value="JP">Japan</option>
                    <option value="IN">India</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Options */}
            <div className="space-y-3">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="saveForFuture"
                  checked={formData.saveForFuture}
                  onChange={(e) => handleInputChange('saveForFuture', e.target.checked)}
                  className="h-4 w-4 text-amber-600 focus:ring-amber-500 border-brand-accent rounded"
                />
                <label htmlFor="saveForFuture" className="ml-2 block text-sm text-primary">
                  Save this payment method for future use
                </label>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="setAsDefault"
                  checked={formData.setAsDefault}
                  onChange={(e) => handleInputChange('setAsDefault', e.target.checked)}
                  className="h-4 w-4 text-amber-600 focus:ring-amber-500 border-brand-accent rounded"
                />
                <label htmlFor="setAsDefault" className="ml-2 block text-sm text-primary">
                  Set as default payment method
                </label>
              </div>
            </div>

            {/* Submit Buttons */}
            <div className="flex justify-end space-x-3 pt-6 border-t border-medium">
              <button
                type="button"
                onClick={onClose}
                className="btn-outline px-6 py-2 rounded-lg"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="btn-primary px-6 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting ? 'Adding...' : 'Add Payment Method'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default PaymentFormModal;
