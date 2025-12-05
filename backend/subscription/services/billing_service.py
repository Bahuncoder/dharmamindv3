"""
Comprehensive Billing Service for DharmaMind
===========================================

Advanced billing service handling invoices, automatic billing,
dunning management, and billing analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from ..models.subscription_models import (
    Invoice, Payment, Subscription, SubscriptionTier,
    BillingCycle, CurrencyCode, PaymentStatus
)
from .payment_service import PaymentService

logger = logging.getLogger(__name__)

class BillingService:
    """
    Comprehensive billing and invoice management service
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.payment_service = PaymentService(db_session)
    
    async def generate_invoice(
        self,
        subscription_id: str,
        billing_period_start: datetime,
        billing_period_end: datetime,
        line_items: Optional[List[Dict]] = None
    ) -> Invoice:
        """
        Generate invoice for subscription billing period
        """
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError("Subscription not found")
            
            # Calculate invoice amounts
            base_amount = subscription.current_price
            tax_amount = self._calculate_tax(base_amount, subscription.user_id)
            total_amount = base_amount + tax_amount
            
            # Generate invoice number
            invoice_number = self._generate_invoice_number()
            
            # Create invoice
            invoice = Invoice(
                subscription_id=subscription_id,
                user_id=subscription.user_id,
                invoice_number=invoice_number,
                amount=base_amount,
                tax_amount=tax_amount,
                total_amount=total_amount,
                currency=subscription.currency,
                due_date=datetime.utcnow() + timedelta(days=30),
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                line_items=line_items or self._generate_default_line_items(subscription),
                status="draft"
            )
            
            self.db.add(invoice)
            self.db.commit()
            
            logger.info(f"Generated invoice {invoice_number} for subscription {subscription_id}")
            return invoice
            
        except Exception as e:
            logger.error(f"Invoice generation failed: {str(e)}")
            self.db.rollback()
            raise
    
    async def process_automatic_billing(self):
        """
        Process automatic billing for all due subscriptions
        """
        try:
            # Find subscriptions due for billing
            due_date = datetime.utcnow() + timedelta(days=1)
            
            subscriptions = self.db.query(Subscription).filter(
                and_(
                    Subscription.current_period_end <= due_date,
                    Subscription.current_price > 0,
                    Subscription.status == 'active'
                )
            ).all()
            
            results = {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'errors': []
            }
            
            for subscription in subscriptions:
                try:
                    await self._process_subscription_billing(subscription)
                    results['successful'] += 1
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append({
                        'subscription_id': subscription.id,
                        'error': str(e)
                    })
                
                results['processed'] += 1
            
            logger.info(f"Automatic billing processed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Automatic billing failed: {str(e)}")
            return {'error': str(e)}
    
    async def _process_subscription_billing(self, subscription: Subscription):
        """
        Process billing for individual subscription
        """
        try:
            # Generate invoice
            invoice = await self.generate_invoice(
                subscription.id,
                subscription.current_period_start,
                subscription.current_period_end
            )
            
            # Mark invoice as sent
            invoice.status = "sent"
            invoice.issue_date = datetime.utcnow()
            
            # Process payment
            payment_result = await self.payment_service.process_subscription_payment(
                subscription.id,
                invoice.total_amount,
                subscription.currency
            )
            
            if payment_result.success:
                # Mark invoice as paid
                invoice.status = "paid"
                invoice.paid_date = datetime.utcnow()
                invoice.payment_id = payment_result.payment_id
                
                # Update subscription period
                self._update_subscription_period(subscription)
                
            else:
                # Handle failed payment
                await self._handle_failed_payment(subscription, invoice, payment_result.error_message)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Subscription billing failed for {subscription.id}: {str(e)}")
            self.db.rollback()
            raise
    
    def _update_subscription_period(self, subscription: Subscription):
        """
        Update subscription billing period after successful payment
        """
        old_end = subscription.current_period_end
        
        if subscription.billing_cycle == BillingCycle.MONTHLY:
            new_end = old_end + timedelta(days=30)
        elif subscription.billing_cycle == BillingCycle.QUARTERLY:
            new_end = old_end + timedelta(days=90)
        elif subscription.billing_cycle == BillingCycle.ANNUALLY:
            new_end = old_end + timedelta(days=365)
        else:
            new_end = old_end + timedelta(days=30)
        
        subscription.current_period_start = old_end
        subscription.current_period_end = new_end
        subscription.updated_at = datetime.utcnow()
    
    async def _handle_failed_payment(
        self,
        subscription: Subscription,
        invoice: Invoice,
        error_message: str
    ):
        """
        Handle failed payment scenarios
        """
        try:
            # Mark invoice as overdue
            invoice.status = "overdue"
            
            # Update subscription status
            subscription.status = "past_due"
            subscription.updated_at = datetime.utcnow()
            
            # Initialize dunning process
            await self._start_dunning_process(subscription, invoice)
            
            logger.warning(f"Payment failed for subscription {subscription.id}: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed payment handling error: {str(e)}")
    
    async def _start_dunning_process(self, subscription: Subscription, invoice: Invoice):
        """
        Start dunning management for failed payments
        """
        try:
            # Create dunning schedule
            dunning_attempts = [
                {'days': 1, 'action': 'email_reminder'},
                {'days': 3, 'action': 'email_warning'},
                {'days': 7, 'action': 'suspend_features'},
                {'days': 14, 'action': 'final_notice'},
                {'days': 30, 'action': 'cancel_subscription'}
            ]
            
            # Store dunning schedule in subscription metadata
            subscription.metadata = subscription.metadata or {}
            subscription.metadata['dunning'] = {
                'started': datetime.utcnow().isoformat(),
                'invoice_id': invoice.id,
                'attempts': dunning_attempts,
                'current_attempt': 0
            }
            
            logger.info(f"Started dunning process for subscription {subscription.id}")
            
        except Exception as e:
            logger.error(f"Dunning process start failed: {str(e)}")
    
    def get_invoice_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Invoice]:
        """
        Get invoice history for user
        """
        return self.db.query(Invoice).filter(
            Invoice.user_id == user_id
        ).order_by(desc(Invoice.created_at)).limit(limit).offset(offset).all()
    
    def get_billing_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get billing summary for user
        """
        try:
            # Get subscription
            subscription = self.db.query(Subscription).filter(
                and_(
                    Subscription.user_id == user_id,
                    Subscription.status.in_(['active', 'past_due', 'trialing'])
                )
            ).first()
            
            if not subscription:
                return {'error': 'No active subscription'}
            
            # Get latest invoice
            latest_invoice = self.db.query(Invoice).filter(
                Invoice.user_id == user_id
            ).order_by(desc(Invoice.created_at)).first()
            
            # Calculate totals
            total_paid = self.db.query(func.sum(Invoice.total_amount)).filter(
                and_(
                    Invoice.user_id == user_id,
                    Invoice.status == 'paid'
                )
            ).scalar() or 0
            
            outstanding_amount = self.db.query(func.sum(Invoice.total_amount)).filter(
                and_(
                    Invoice.user_id == user_id,
                    Invoice.status.in_(['sent', 'overdue'])
                )
            ).scalar() or 0
            
            return {
                'subscription': {
                    'id': subscription.id,
                    'tier': subscription.tier.value,
                    'status': subscription.status.value,
                    'current_price': subscription.current_price,
                    'billing_cycle': subscription.billing_cycle.value,
                    'next_billing_date': subscription.current_period_end.isoformat() if subscription.current_period_end else None
                },
                'billing': {
                    'total_paid': float(total_paid),
                    'outstanding_amount': float(outstanding_amount),
                    'currency': subscription.currency.value
                },
                'latest_invoice': {
                    'id': latest_invoice.id,
                    'number': latest_invoice.invoice_number,
                    'amount': latest_invoice.total_amount,
                    'status': latest_invoice.status,
                    'due_date': latest_invoice.due_date.isoformat()
                } if latest_invoice else None
            }
            
        except Exception as e:
            logger.error(f"Billing summary failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_tax(self, amount: float, user_id: str) -> float:
        """
        Calculate tax amount based on user location and regulations
        """
        # This would integrate with tax calculation services
        # For now, return 0 (no tax)
        return 0.0
    
    def _generate_invoice_number(self) -> str:
        """
        Generate unique invoice number
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"INV-DM-{timestamp}"
    
    def _generate_default_line_items(self, subscription: Subscription) -> List[Dict]:
        """
        Generate default line items for subscription invoice
        """
        return [{
            'description': f'DharmaMind {subscription.tier.value.title()} Subscription',
            'quantity': 1,
            'unit_price': subscription.current_price,
            'total': subscription.current_price,
            'period': f'{subscription.current_period_start.strftime("%Y-%m-%d")} to {subscription.current_period_end.strftime("%Y-%m-%d")}'
        }]