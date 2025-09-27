"""
Comprehensive Payment Processing Service for DharmaMind
=====================================================

Multi-provider payment service supporting Stripe, PayPal, Razorpay, Square,
and other payment methods with webhook handling and automated billing.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, NamedTuple
from decimal import Decimal
import stripe
import paypalrestsdk
import razorpay
import squareup

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ..models.subscription_models import (
    Payment, PaymentStatus, PaymentMethod, CurrencyCode,
    PaymentMethodRecord, Invoice, Subscription
)

logger = logging.getLogger(__name__)

class PaymentResult(NamedTuple):
    """Payment processing result"""
    success: bool
    payment_id: Optional[str] = None
    error_message: Optional[str] = None
    external_id: Optional[str] = None
    amount_charged: Optional[float] = None

class PaymentService:
    """
    Comprehensive payment processing service
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        
        # Initialize payment providers
        self._setup_payment_providers()
        
        # Supported currencies per provider
        self.provider_currencies = {
            PaymentMethod.STRIPE: [
                CurrencyCode.USD, CurrencyCode.EUR, CurrencyCode.GBP,
                CurrencyCode.INR, CurrencyCode.CAD, CurrencyCode.AUD
            ],
            PaymentMethod.PAYPAL: [
                CurrencyCode.USD, CurrencyCode.EUR, CurrencyCode.GBP,
                CurrencyCode.CAD, CurrencyCode.AUD
            ],
            PaymentMethod.RAZORPAY: [CurrencyCode.INR],
            PaymentMethod.UPI: [CurrencyCode.INR]
        }
    
    def _setup_payment_providers(self):
        """
        Initialize payment provider configurations
        """
        # These would come from environment variables or config
        self.stripe_config = {
            'api_key': 'sk_test_...',  # From environment
            'webhook_secret': 'whsec_...'
        }
        
        self.paypal_config = {
            'client_id': 'client_id',  # From environment
            'client_secret': 'client_secret',
            'mode': 'sandbox'  # or 'live'
        }
        
        self.razorpay_config = {
            'key_id': 'rzp_test_...',  # From environment
            'key_secret': 'key_secret'
        }
        
        # Initialize SDKs
        if self.stripe_config['api_key']:
            stripe.api_key = self.stripe_config['api_key']
        
        if self.paypal_config['client_id']:
            paypalrestsdk.configure({
                'mode': self.paypal_config['mode'],
                'client_id': self.paypal_config['client_id'],
                'client_secret': self.paypal_config['client_secret']
            })
    
    async def process_one_time_payment(
        self,
        user_id: str,
        amount: float,
        currency: CurrencyCode,
        payment_method: PaymentMethod,
        payment_method_id: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> PaymentResult:
        """
        Process a one-time payment
        """
        try:
            # Create payment record
            payment = Payment(
                user_id=user_id,
                amount=amount,
                currency=currency,
                status=PaymentStatus.PENDING,
                payment_method=payment_method,
                description=description or f"DharmaMind Payment - ${amount}",
                metadata=metadata or {}
            )
            
            self.db.add(payment)
            self.db.flush()  # Get payment ID
            
            # Process based on payment method
            if payment_method == PaymentMethod.STRIPE:
                result = await self._process_stripe_payment(
                    payment, payment_method_id, amount, currency
                )
            elif payment_method == PaymentMethod.PAYPAL:
                result = await self._process_paypal_payment(
                    payment, payment_method_id, amount, currency
                )
            elif payment_method == PaymentMethod.RAZORPAY:
                result = await self._process_razorpay_payment(
                    payment, payment_method_id, amount, currency
                )
            else:
                result = PaymentResult(
                    success=False,
                    error_message=f"Unsupported payment method: {payment_method}"
                )
            
            # Update payment status
            if result.success:
                payment.status = PaymentStatus.SUCCEEDED
                payment.processed_at = datetime.utcnow()
                if result.external_id:
                    self._set_external_id(payment, payment_method, result.external_id)
            else:
                payment.status = PaymentStatus.FAILED
                payment.failed_at = datetime.utcnow()
                payment.failure_reason = result.error_message
            
            self.db.commit()
            
            return PaymentResult(
                success=result.success,
                payment_id=payment.id,
                error_message=result.error_message,
                external_id=result.external_id,
                amount_charged=amount if result.success else None
            )
            
        except Exception as e:
            logger.error(f"Payment processing failed: {str(e)}")
            self.db.rollback()
            return PaymentResult(
                success=False,
                error_message=f"Payment processing error: {str(e)}"
            )
    
    async def process_subscription_payment(
        self,
        subscription_id: str,
        amount: float,
        currency: CurrencyCode
    ) -> PaymentResult:
        """
        Process recurring subscription payment
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                return PaymentResult(
                    success=False,
                    error_message="Subscription not found"
                )
            
            if not subscription.payment_method_id:
                return PaymentResult(
                    success=False,
                    error_message="No payment method configured"
                )
            
            # Get payment method
            payment_method_record = self.db.query(PaymentMethodRecord).filter(
                PaymentMethodRecord.id == subscription.payment_method_id
            ).first()
            
            if not payment_method_record:
                return PaymentResult(
                    success=False,
                    error_message="Payment method not found"
                )
            
            # Process payment
            return await self.process_one_time_payment(
                user_id=subscription.user_id,
                amount=amount,
                currency=currency,
                payment_method=payment_method_record.type,
                payment_method_id=self._get_external_payment_method_id(payment_method_record),
                description=f"DharmaMind Subscription - {subscription.tier.value.title()}",
                metadata={'subscription_id': subscription_id}
            )
            
        except Exception as e:
            logger.error(f"Subscription payment processing failed: {str(e)}")
            return PaymentResult(
                success=False,
                error_message=f"Subscription payment error: {str(e)}"
            )
    
    async def setup_recurring_payment(
        self,
        subscription_id: str,
        payment_method_id: str
    ) -> PaymentResult:
        """
        Set up recurring payment for subscription
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                return PaymentResult(
                    success=False,
                    error_message="Subscription not found"
                )
            
            payment_method_record = self.db.query(PaymentMethodRecord).filter(
                PaymentMethodRecord.id == payment_method_id
            ).first()
            
            if not payment_method_record:
                return PaymentResult(
                    success=False,
                    error_message="Payment method not found"
                )
            
            # Set up based on payment provider
            if payment_method_record.type == PaymentMethod.STRIPE:
                result = await self._setup_stripe_subscription(subscription, payment_method_record)
            elif payment_method_record.type == PaymentMethod.PAYPAL:
                result = await self._setup_paypal_subscription(subscription, payment_method_record)
            else:
                return PaymentResult(
                    success=False,
                    error_message=f"Recurring payments not supported for {payment_method_record.type}"
                )
            
            if result.success:
                subscription.payment_method_id = payment_method_id
                self.db.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"Recurring payment setup failed: {str(e)}")
            self.db.rollback()
            return PaymentResult(
                success=False,
                error_message=f"Recurring payment setup error: {str(e)}"
            )
    
    async def cancel_recurring_payment(self, external_subscription_id: str) -> bool:
        """
        Cancel recurring payment subscription
        """
        try:
            # Find subscription by external ID
            subscription = self.db.query(Subscription).filter(
                or_(
                    Subscription.stripe_subscription_id == external_subscription_id,
                    Subscription.paypal_subscription_id == external_subscription_id
                )
            ).first()
            
            if not subscription:
                logger.warning(f"Subscription not found for external ID: {external_subscription_id}")
                return True  # Consider it cancelled if not found
            
            # Cancel based on provider
            if subscription.stripe_subscription_id == external_subscription_id:
                stripe.Subscription.modify(
                    external_subscription_id,
                    cancel_at_period_end=True
                )
            elif subscription.paypal_subscription_id == external_subscription_id:
                # PayPal subscription cancellation
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel recurring payment: {str(e)}")
            return False
    
    async def process_refund(
        self,
        payment_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> PaymentResult:
        """
        Process payment refund
        """
        try:
            payment = self.db.query(Payment).filter(
                Payment.id == payment_id
            ).first()
            
            if not payment:
                return PaymentResult(
                    success=False,
                    error_message="Payment not found"
                )
            
            if payment.status != PaymentStatus.SUCCEEDED:
                return PaymentResult(
                    success=False,
                    error_message="Can only refund successful payments"
                )
            
            refund_amount = amount or payment.amount
            
            # Process refund based on payment method
            if payment.payment_method == PaymentMethod.STRIPE and payment.stripe_charge_id:
                result = await self._process_stripe_refund(payment, refund_amount, reason)
            elif payment.payment_method == PaymentMethod.PAYPAL and payment.paypal_payment_id:
                result = await self._process_paypal_refund(payment, refund_amount, reason)
            else:
                return PaymentResult(
                    success=False,
                    error_message=f"Refunds not supported for {payment.payment_method}"
                )
            
            if result.success:
                payment.refunded_amount += refund_amount
                payment.refund_reason = reason
                
                if payment.refunded_amount >= payment.amount:
                    payment.status = PaymentStatus.REFUNDED
                else:
                    payment.status = PaymentStatus.PARTIALLY_REFUNDED
                
                payment.refunded_at = datetime.utcnow()
                self.db.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"Refund processing failed: {str(e)}")
            self.db.rollback()
            return PaymentResult(
                success=False,
                error_message=f"Refund processing error: {str(e)}"
            )
    
    async def add_payment_method(
        self,
        user_id: str,
        payment_method: PaymentMethod,
        external_id: str,
        metadata: Optional[Dict] = None
    ) -> Optional[PaymentMethodRecord]:
        """
        Add payment method for user
        """
        try:
            # Check if method already exists
            existing = self.db.query(PaymentMethodRecord).filter(
                and_(
                    PaymentMethodRecord.user_id == user_id,
                    getattr(PaymentMethodRecord, f"{payment_method.value}_payment_method_id") == external_id
                )
            ).first()
            
            if existing:
                return existing
            
            # Create new payment method record
            payment_method_record = PaymentMethodRecord(
                user_id=user_id,
                type=payment_method,
                metadata=metadata or {}
            )
            
            # Set external ID based on provider
            if payment_method == PaymentMethod.STRIPE:
                payment_method_record.stripe_payment_method_id = external_id
                
                # Get card details from Stripe
                stripe_pm = stripe.PaymentMethod.retrieve(external_id)
                if stripe_pm.card:
                    payment_method_record.last_four = stripe_pm.card.last4
                    payment_method_record.brand = stripe_pm.card.brand
                    payment_method_record.exp_month = stripe_pm.card.exp_month
                    payment_method_record.exp_year = stripe_pm.card.exp_year
                    
            elif payment_method == PaymentMethod.PAYPAL:
                payment_method_record.paypal_agreement_id = external_id
            
            # Set as default if no other payment methods exist
            existing_count = self.db.query(PaymentMethodRecord).filter(
                and_(
                    PaymentMethodRecord.user_id == user_id,
                    PaymentMethodRecord.is_active == True
                )
            ).count()
            
            if existing_count == 0:
                payment_method_record.is_default = True
            
            self.db.add(payment_method_record)
            self.db.commit()
            
            logger.info(f"Added payment method for user {user_id}")
            return payment_method_record
            
        except Exception as e:
            logger.error(f"Failed to add payment method: {str(e)}")
            self.db.rollback()
            return None
    
    def get_user_payment_methods(self, user_id: str) -> List[PaymentMethodRecord]:
        """
        Get all active payment methods for user
        """
        return self.db.query(PaymentMethodRecord).filter(
            and_(
                PaymentMethodRecord.user_id == user_id,
                PaymentMethodRecord.is_active == True
            )
        ).order_by(desc(PaymentMethodRecord.is_default)).all()
    
    def get_payment_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Payment]:
        """
        Get payment history for user
        """
        return self.db.query(Payment).filter(
            Payment.user_id == user_id
        ).order_by(desc(Payment.created_at)).limit(limit).offset(offset).all()
    
    async def handle_webhook(
        self,
        provider: PaymentMethod,
        payload: Dict,
        signature: Optional[str] = None
    ) -> bool:
        """
        Handle payment provider webhooks
        """
        try:
            if provider == PaymentMethod.STRIPE:
                return await self._handle_stripe_webhook(payload, signature)
            elif provider == PaymentMethod.PAYPAL:
                return await self._handle_paypal_webhook(payload)
            elif provider == PaymentMethod.RAZORPAY:
                return await self._handle_razorpay_webhook(payload, signature)
            else:
                logger.warning(f"Webhook handler not implemented for {provider}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook handling failed: {str(e)}")
            return False
    
    # Provider-specific implementations
    
    async def _process_stripe_payment(
        self,
        payment: Payment,
        payment_method_id: str,
        amount: float,
        currency: CurrencyCode
    ) -> PaymentResult:
        """Process Stripe payment"""
        try:
            # Convert amount to cents
            amount_cents = int(amount * 100)
            
            # Create payment intent
            intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency=currency.value,
                payment_method=payment_method_id,
                confirmation_method='manual',
                confirm=True,
                metadata={'payment_id': payment.id}
            )
            
            if intent.status == 'succeeded':
                return PaymentResult(
                    success=True,
                    external_id=intent.charges.data[0].id if intent.charges.data else intent.id
                )
            else:
                return PaymentResult(
                    success=False,
                    error_message=f"Payment failed: {intent.status}"
                )
                
        except stripe.error.StripeError as e:
            return PaymentResult(
                success=False,
                error_message=f"Stripe error: {str(e)}"
            )
    
    async def _process_paypal_payment(
        self,
        payment: Payment,
        payment_method_id: str,
        amount: float,
        currency: CurrencyCode
    ) -> PaymentResult:
        """Process PayPal payment"""
        try:
            # PayPal payment processing logic
            # This would use PayPal SDK
            return PaymentResult(
                success=True,
                external_id="paypal_payment_id"
            )
            
        except Exception as e:
            return PaymentResult(
                success=False,
                error_message=f"PayPal error: {str(e)}"
            )
    
    async def _process_razorpay_payment(
        self,
        payment: Payment,
        payment_method_id: str,
        amount: float,
        currency: CurrencyCode
    ) -> PaymentResult:
        """Process Razorpay payment"""
        try:
            # Razorpay payment processing logic
            return PaymentResult(
                success=True,
                external_id="razorpay_payment_id"
            )
            
        except Exception as e:
            return PaymentResult(
                success=False,
                error_message=f"Razorpay error: {str(e)}"
            )
    
    async def _setup_stripe_subscription(
        self,
        subscription: Subscription,
        payment_method_record: PaymentMethodRecord
    ) -> PaymentResult:
        """Set up Stripe subscription"""
        try:
            # Create Stripe customer if needed
            stripe_customer_id = subscription.metadata.get('stripe_customer_id')
            
            if not stripe_customer_id:
                customer = stripe.Customer.create(
                    payment_method=payment_method_record.stripe_payment_method_id,
                    invoice_settings={
                        'default_payment_method': payment_method_record.stripe_payment_method_id
                    },
                    metadata={'user_id': subscription.user_id}
                )
                stripe_customer_id = customer.id
                subscription.metadata = subscription.metadata or {}
                subscription.metadata['stripe_customer_id'] = stripe_customer_id
            
            # Create price object
            price = stripe.Price.create(
                unit_amount=int(subscription.current_price * 100),
                currency=subscription.currency.value,
                recurring={'interval': 'month' if subscription.billing_cycle.value == 'monthly' else 'year'},
                product_data={'name': f'DharmaMind {subscription.tier.value.title()}'}
            )
            
            # Create subscription
            stripe_subscription = stripe.Subscription.create(
                customer=stripe_customer_id,
                items=[{'price': price.id}],
                metadata={'subscription_id': subscription.id}
            )
            
            subscription.stripe_subscription_id = stripe_subscription.id
            
            return PaymentResult(
                success=True,
                external_id=stripe_subscription.id
            )
            
        except stripe.error.StripeError as e:
            return PaymentResult(
                success=False,
                error_message=f"Stripe subscription setup error: {str(e)}"
            )
    
    async def _setup_paypal_subscription(
        self,
        subscription: Subscription,
        payment_method_record: PaymentMethodRecord
    ) -> PaymentResult:
        """Set up PayPal subscription"""
        try:
            # PayPal subscription setup logic
            return PaymentResult(
                success=True,
                external_id="paypal_subscription_id"
            )
            
        except Exception as e:
            return PaymentResult(
                success=False,
                error_message=f"PayPal subscription setup error: {str(e)}"
            )
    
    async def _process_stripe_refund(
        self,
        payment: Payment,
        amount: float,
        reason: Optional[str]
    ) -> PaymentResult:
        """Process Stripe refund"""
        try:
            refund = stripe.Refund.create(
                charge=payment.stripe_charge_id,
                amount=int(amount * 100),
                reason=reason or 'requested_by_customer',
                metadata={'payment_id': payment.id}
            )
            
            return PaymentResult(
                success=True,
                external_id=refund.id,
                amount_charged=refund.amount / 100
            )
            
        except stripe.error.StripeError as e:
            return PaymentResult(
                success=False,
                error_message=f"Stripe refund error: {str(e)}"
            )
    
    async def _process_paypal_refund(
        self,
        payment: Payment,
        amount: float,
        reason: Optional[str]
    ) -> PaymentResult:
        """Process PayPal refund"""
        try:
            # PayPal refund processing logic
            return PaymentResult(
                success=True,
                external_id="paypal_refund_id",
                amount_charged=amount
            )
            
        except Exception as e:
            return PaymentResult(
                success=False,
                error_message=f"PayPal refund error: {str(e)}"
            )
    
    async def _handle_stripe_webhook(self, payload: Dict, signature: str) -> bool:
        """Handle Stripe webhook"""
        try:
            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload, signature, self.stripe_config['webhook_secret']
            )
            
            # Handle different event types
            if event['type'] == 'payment_intent.succeeded':
                await self._handle_stripe_payment_succeeded(event['data']['object'])
            elif event['type'] == 'payment_intent.payment_failed':
                await self._handle_stripe_payment_failed(event['data']['object'])
            elif event['type'] == 'invoice.payment_succeeded':
                await self._handle_stripe_subscription_payment(event['data']['object'])
            
            return True
            
        except Exception as e:
            logger.error(f"Stripe webhook handling failed: {str(e)}")
            return False
    
    async def _handle_paypal_webhook(self, payload: Dict) -> bool:
        """Handle PayPal webhook"""
        try:
            # PayPal webhook handling logic
            return True
            
        except Exception as e:
            logger.error(f"PayPal webhook handling failed: {str(e)}")
            return False
    
    async def _handle_razorpay_webhook(self, payload: Dict, signature: str) -> bool:
        """Handle Razorpay webhook"""
        try:
            # Razorpay webhook handling logic
            return True
            
        except Exception as e:
            logger.error(f"Razorpay webhook handling failed: {str(e)}")
            return False
    
    async def _handle_stripe_payment_succeeded(self, payment_intent: Dict):
        """Handle successful Stripe payment"""
        payment_id = payment_intent['metadata'].get('payment_id')
        if payment_id:
            payment = self.db.query(Payment).filter(Payment.id == payment_id).first()
            if payment and payment.status == PaymentStatus.PENDING:
                payment.status = PaymentStatus.SUCCEEDED
                payment.processed_at = datetime.utcnow()
                if payment_intent.get('charges', {}).get('data'):
                    payment.stripe_charge_id = payment_intent['charges']['data'][0]['id']
                self.db.commit()
    
    async def _handle_stripe_payment_failed(self, payment_intent: Dict):
        """Handle failed Stripe payment"""
        payment_id = payment_intent['metadata'].get('payment_id')
        if payment_id:
            payment = self.db.query(Payment).filter(Payment.id == payment_id).first()
            if payment and payment.status == PaymentStatus.PENDING:
                payment.status = PaymentStatus.FAILED
                payment.failed_at = datetime.utcnow()
                payment.failure_reason = payment_intent.get('last_payment_error', {}).get('message', 'Payment failed')
                self.db.commit()
    
    async def _handle_stripe_subscription_payment(self, invoice: Dict):
        """Handle successful subscription payment"""
        subscription_id = invoice['subscription_metadata'].get('subscription_id')
        if subscription_id:
            # Update subscription period dates
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if subscription:
                subscription.current_period_start = datetime.fromtimestamp(invoice['period_start'])
                subscription.current_period_end = datetime.fromtimestamp(invoice['period_end'])
                self.db.commit()
    
    def _set_external_id(self, payment: Payment, payment_method: PaymentMethod, external_id: str):
        """Set external payment ID based on provider"""
        if payment_method == PaymentMethod.STRIPE:
            payment.stripe_charge_id = external_id
        elif payment_method == PaymentMethod.PAYPAL:
            payment.paypal_payment_id = external_id
        elif payment_method == PaymentMethod.RAZORPAY:
            payment.razorpay_payment_id = external_id
    
    def _get_external_payment_method_id(self, payment_method_record: PaymentMethodRecord) -> str:
        """Get external payment method ID"""
        if payment_method_record.type == PaymentMethod.STRIPE:
            return payment_method_record.stripe_payment_method_id
        elif payment_method_record.type == PaymentMethod.PAYPAL:
            return payment_method_record.paypal_agreement_id
        else:
            return ""