import React, { useState, useEffect } from 'react';
import { paymentAPI, PaymentMethod, CreatePaymentMethodRequest, BillingAddress } from '../services/paymentAPI';

interface PaymentMethodManagerProps {
  onPaymentMethodAdded?: (paymentMethod: PaymentMethod) => void;
  onPaymentMethodDeleted?: (paymentMethodId: string) => void;
  onPaymentMethodSetDefault?: (paymentMethod: PaymentMethod) => void;
  showAddForm?: boolean;
  formOnly?: boolean; // New prop to show only the form without the payment methods list
}

interface CardFormData {
  cardNumber: string;
  expiryDate: string;
  cvc: string;
  cardholderName: string;
  billingAddress: BillingAddress;
  saveForFuture: boolean;
  setAsDefault: boolean;
}

const PaymentMethodManager: React.FC<PaymentMethodManagerProps> = ({
  onPaymentMethodAdded,
  onPaymentMethodDeleted,
  onPaymentMethodSetDefault,
  showAddForm = false,
  formOnly = false
}) => {
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [loading, setLoading] = useState(false);
  const [showForm, setShowForm] = useState(showAddForm);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [formData, setFormData] = useState<CardFormData>({
    cardNumber: '',
    expiryDate: '',
    cvc: '',
    cardholderName: '',
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

  // Load payment methods on component mount
  useEffect(() => {
    loadPaymentMethods();
  }, []);

  const loadPaymentMethods = async () => {
    setLoading(true);
    try {
      const methods = await paymentAPI.getPaymentMethods();
      setPaymentMethods(methods);
    } catch (error) {
      console.error('Failed to load payment methods:', error);
      setError('Failed to load payment methods');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    if (field.startsWith('billingAddress.')) {
      const addressField = field.replace('billingAddress.', '');
      setFormData(prev => ({
        ...prev,
        billingAddress: {
          ...prev.billingAddress,
          [addressField]: value
        }
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };

  const formatCardNumber = (value: string) => {
    // Remove all non-digits
    const digits = value.replace(/\D/g, '');
    // Add spaces every 4 digits
    return digits.replace(/(\d{4})(?=\d)/g, '$1 ');
  };

  const formatExpiryDate = (value: string) => {
    const digits = value.replace(/\D/g, '');
    if (digits.length >= 2) {
      return digits.substring(0, 2) + '/' + digits.substring(2, 4);
    }
    return digits;
  };

  const validateCardForm = (): string[] => {
    const errors: string[] = [];

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

    return errors;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    const validationErrors = validateCardForm();
    if (validationErrors.length > 0) {
      setError(validationErrors.join(', '));
      return;
    }

    setSubmitting(true);

    try {
      // Parse expiry date
      const [expMonth, expYear] = formData.expiryDate.split('/');
      const fullYear = parseInt(`20${expYear}`);

      const paymentMethodRequest: CreatePaymentMethodRequest = {
        type: 'credit_card',
        card_number: formData.cardNumber.replace(/\s/g, ''),
        exp_month: parseInt(expMonth),
        exp_year: fullYear,
        cvc: formData.cvc,
        cardholder_name: formData.cardholderName,
        billing_address: formData.billingAddress,
        save_for_future: formData.saveForFuture,
        set_as_default: formData.setAsDefault
      };

      const newPaymentMethod = await paymentAPI.createPaymentMethod(paymentMethodRequest);
      
      // Update local state
      setPaymentMethods(prev => [...prev, newPaymentMethod]);
      
      // Reset form
      setFormData({
        cardNumber: '',
        expiryDate: '',
        cvc: '',
        cardholderName: '',
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

      setSuccess('Payment method added successfully!');
      setShowForm(false);

      // Notify parent component
      if (onPaymentMethodAdded) {
        onPaymentMethodAdded(newPaymentMethod);
      }

    } catch (error) {
      console.error('Failed to add payment method:', error);
      setError(error instanceof Error ? error.message : 'Failed to add payment method');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (paymentMethodId: string) => {
    if (!confirm('Are you sure you want to delete this payment method?')) {
      return;
    }

    try {
      await paymentAPI.deletePaymentMethod(paymentMethodId);
      setPaymentMethods(prev => prev.filter(pm => pm.id !== paymentMethodId));
      setSuccess('Payment method deleted successfully');

      if (onPaymentMethodDeleted) {
        onPaymentMethodDeleted(paymentMethodId);
      }
    } catch (error) {
      console.error('Failed to delete payment method:', error);
      setError('Failed to delete payment method');
    }
  };

  const handleSetDefault = async (paymentMethodId: string) => {
    try {
      const updatedPaymentMethod = await paymentAPI.setDefaultPaymentMethod(paymentMethodId);
      
      // Update local state
      setPaymentMethods(prev => 
        prev.map(pm => ({
          ...pm,
          default: pm.id === paymentMethodId
        }))
      );

      setSuccess('Default payment method updated');

      if (onPaymentMethodSetDefault) {
        onPaymentMethodSetDefault(updatedPaymentMethod);
      }
    } catch (error) {
      console.error('Failed to set default payment method:', error);
      setError('Failed to update default payment method');
    }
  };

  const getCardIcon = (brand: string) => {
    switch (brand?.toLowerCase()) {
      case 'visa':
        return '💳';
      case 'mastercard':
        return '💳';
      case 'amex':
      case 'american express':
        return '💳';
      case 'discover':
        return '💳';
      default:
        return '💳';
    }
  };

  const isCardExpired = (expMonth: number, expYear: number) => {
    const now = new Date();
    const expiry = new Date(expYear, expMonth - 1);
    return expiry < now;
  };

  return (
    <div className="space-y-6">
      {/* Alerts */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p className="ml-3 text-sm text-red-800">{error}</p>
          </div>
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex">
            <svg className="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <p className="ml-3 text-sm text-green-800">{success}</p>
          </div>
        </div>
      )}

      {/* Payment Methods List */}
      {!formOnly && (
        <div>
        <div className="flex items-center justify-between mb-4">
<<<<<<< HEAD
          <h3 className="text-lg font-semibold text-neutral-900">Payment Methods</h3>
          {!showForm && (
            <button
              onClick={() => setShowForm(true)}
              className="bg-gradient-to-r from-gold-600 to-gold-700 text-white px-4 py-2 rounded-lg hover:from-gold-700 hover:to-gold-800 transition-colors text-sm font-medium"
=======
          <h3 className="text-lg font-semibold text-gray-900">Payment Methods</h3>
          {!showForm && (
            <button
              onClick={() => setShowForm(true)}
              className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-4 py-2 rounded-lg hover:from-amber-700 hover:to-emerald-700 transition-colors text-sm font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              + Add Payment Method
            </button>
          )}
        </div>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-600 mx-auto"></div>
<<<<<<< HEAD
            <p className="text-neutral-600 mt-2">Loading payment methods...</p>
=======
            <p className="text-gray-600 mt-2">Loading payment methods...</p>
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
          </div>
        ) : paymentMethods.length > 0 ? (
          <div className="space-y-3">
            {paymentMethods.map((method) => (
              <div
                key={method.id}
                className={`flex items-center justify-between p-4 border rounded-lg ${
<<<<<<< HEAD
                  method.default ? 'border-gold-300 bg-gold-50' : 'border-neutral-300 bg-neutral-100'
=======
                  method.default ? 'border-emerald-300 bg-emerald-50' : 'border-gray-200 bg-white'
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                }`}
              >
                <div className="flex items-center space-x-3">
                  <div className="text-2xl">{getCardIcon(method.brand || '')}</div>
                  <div>
                    <div className="flex items-center space-x-2">
<<<<<<< HEAD
                      <p className="font-medium text-neutral-900">
                        {method.brand} •••• {method.last_four}
                      </p>
                      {method.default && (
                        <span className="bg-gold-100 text-gold-700 text-xs px-2 py-1 rounded-full font-medium">
=======
                      <p className="font-medium text-gray-900">
                        {method.brand} •••• {method.last_four}
                      </p>
                      {method.default && (
                        <span className="bg-emerald-100 text-emerald-800 text-xs px-2 py-1 rounded-full font-medium">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                          Default
                        </span>
                      )}
                      {method.exp_month && method.exp_year && isCardExpired(method.exp_month, method.exp_year) && (
                        <span className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full font-medium">
                          Expired
                        </span>
                      )}
                    </div>
<<<<<<< HEAD
                    <p className="text-sm text-neutral-600">
=======
                    <p className="text-sm text-gray-600">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                      {method.exp_month && method.exp_year && `Expires ${method.exp_month}/${method.exp_year}`}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {!method.default && (
                    <button
                      onClick={() => handleSetDefault(method.id)}
<<<<<<< HEAD
                      className="text-sm text-gold-600 hover:text-gold-700 font-medium"
=======
                      className="text-sm text-amber-600 hover:text-amber-700 font-medium"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    >
                      Set Default
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(method.id)}
                    className="text-sm text-red-600 hover:text-red-700 font-medium"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
<<<<<<< HEAD
          <div className="text-center py-8 border border-neutral-300 rounded-lg">
            <svg className="w-12 h-12 text-neutral-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
            </svg>
            <h3 className="text-lg font-medium text-neutral-900 mb-2">No payment methods</h3>
            <p className="text-neutral-600 mb-4">Add a payment method to manage your subscription</p>
            <button
              onClick={() => setShowForm(true)}
              className="bg-gradient-to-r from-gold-600 to-gold-700 text-white px-6 py-2 rounded-lg hover:from-gold-700 hover:to-gold-800 transition-colors"
=======
          <div className="text-center py-8 border border-gray-200 rounded-lg">
            <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
            </svg>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No payment methods</h3>
            <p className="text-gray-600 mb-4">Add a payment method to manage your subscription</p>
            <button
              onClick={() => setShowForm(true)}
              className="bg-gradient-to-r from-amber-600 to-emerald-600 text-white px-6 py-2 rounded-lg hover:from-amber-700 hover:to-emerald-700 transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
            >
              Add Payment Method
            </button>
          </div>
        )}
      </div>
      )}

      {/* Add Payment Method Form */}
      {(showForm || formOnly) && (
<<<<<<< HEAD
        <div className={`bg-neutral-100 ${formOnly ? '' : 'border border-neutral-300 rounded-lg p-6'}`}>
          {!formOnly && (
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900">Add Payment Method</h3>
              <button
                onClick={() => setShowForm(false)}
                className="text-neutral-600 hover:text-gold-600"
=======
        <div className={`bg-white ${formOnly ? '' : 'border border-gray-200 rounded-lg p-6'}`}>
          {!formOnly && (
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Add Payment Method</h3>
              <button
                onClick={() => setShowForm(false)}
                className="text-gray-400 hover:text-gray-600"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Card Information */}
            <div>
<<<<<<< HEAD
              <h4 className="font-medium text-neutral-900 mb-4">Card Information</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
              <h4 className="font-medium text-gray-900 mb-4">Card Information</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Cardholder Name *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.cardholderName}
                    onChange={(e) => handleInputChange('cardholderName', e.target.value)}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="John Doe"
                  />
                </div>
                <div className="md:col-span-2">
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Card Number *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.cardNumber}
                    onChange={(e) => handleInputChange('cardNumber', formatCardNumber(e.target.value))}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="1234 5678 9012 3456"
                    maxLength={19}
                  />
                </div>
                <div>
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Expiry Date *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.expiryDate}
                    onChange={(e) => handleInputChange('expiryDate', formatExpiryDate(e.target.value))}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="MM/YY"
                    maxLength={5}
                  />
                </div>
                <div>
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    CVC *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.cvc}
                    onChange={(e) => handleInputChange('cvc', e.target.value.replace(/\D/g, ''))}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="123"
                    maxLength={4}
                  />
                </div>
              </div>
            </div>

            {/* Billing Address */}
            <div>
<<<<<<< HEAD
              <h4 className="font-medium text-neutral-900 mb-4">Billing Address</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
              <h4 className="font-medium text-gray-900 mb-4">Billing Address</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Address Line 1 *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.line1}
                    onChange={(e) => handleInputChange('billingAddress.line1', e.target.value)}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="123 Main St"
                  />
                </div>
                <div className="md:col-span-2">
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Address Line 2
                  </label>
                  <input
                    type="text"
                    value={formData.billingAddress.line2}
                    onChange={(e) => handleInputChange('billingAddress.line2', e.target.value)}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="Apt, suite, etc."
                  />
                </div>
                <div>
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    City *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.city}
                    onChange={(e) => handleInputChange('billingAddress.city', e.target.value)}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="New York"
                  />
                </div>
                <div>
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    State *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.state}
                    onChange={(e) => handleInputChange('billingAddress.state', e.target.value)}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="NY"
                  />
                </div>
                <div>
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Postal Code *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.billingAddress.postal_code}
                    onChange={(e) => handleInputChange('billingAddress.postal_code', e.target.value)}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    placeholder="10001"
                  />
                </div>
                <div>
<<<<<<< HEAD
                  <label className="block text-sm font-medium text-neutral-900 mb-2">
=======
                  <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                    Country *
                  </label>
                  <select
                    required
                    value={formData.billingAddress.country}
                    onChange={(e) => handleInputChange('billingAddress.country', e.target.value)}
<<<<<<< HEAD
                    className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-500 focus:border-transparent"
=======
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
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
                  onChange={(e) => handleInputChange('saveForFuture', e.target.checked.toString())}
<<<<<<< HEAD
                  className="h-4 w-4 text-gold-600 focus:ring-gold-500 border-neutral-300 rounded"
                />
                <label htmlFor="saveForFuture" className="ml-2 block text-sm text-neutral-900">
=======
                  className="h-4 w-4 text-amber-600 focus:ring-amber-500 border-gray-300 rounded"
                />
                <label htmlFor="saveForFuture" className="ml-2 block text-sm text-gray-900">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  Save this payment method for future use
                </label>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="setAsDefault"
                  checked={formData.setAsDefault}
                  onChange={(e) => handleInputChange('setAsDefault', e.target.checked.toString())}
<<<<<<< HEAD
                  className="h-4 w-4 text-gold-600 focus:ring-gold-500 border-neutral-300 rounded"
                />
                <label htmlFor="setAsDefault" className="ml-2 block text-sm text-neutral-900">
=======
                  className="h-4 w-4 text-amber-600 focus:ring-amber-500 border-gray-300 rounded"
                />
                <label htmlFor="setAsDefault" className="ml-2 block text-sm text-gray-900">
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
                  Set as default payment method
                </label>
              </div>
            </div>

            {/* Submit Buttons */}
<<<<<<< HEAD
            <div className="flex justify-end space-x-3 pt-6 border-t border-neutral-300">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-6 py-2 border border-neutral-300 text-neutral-900 rounded-lg hover:bg-neutral-100 transition-colors"
=======
            <div className="flex justify-end space-x-3 pt-6 border-t border-gray-200">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
<<<<<<< HEAD
                className="px-6 py-2 bg-gradient-to-r from-gold-600 to-gold-700 text-white rounded-lg hover:from-gold-700 hover:to-gold-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
=======
                className="px-6 py-2 bg-gradient-to-r from-amber-600 to-emerald-600 text-white rounded-lg hover:from-amber-700 hover:to-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
>>>>>>> 0a7b3468604638c47efcf853a27e0c92a7e9fccc
              >
                {submitting ? 'Adding...' : 'Add Payment Method'}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

export default PaymentMethodManager;
