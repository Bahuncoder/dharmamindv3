"""
Payment Processing API Routes for DharmaMind
===========================================

RESTful API endpoints for payment processing, payment methods,
invoicing, and payment history.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging

from ..services.payment_service import PaymentService
from ..models.subscription_models import PaymentMethod, CurrencyCode

logger = logging.getLogger(__name__)

payment_bp = Blueprint('payment', __name__, url_prefix='/api/payment')

def get_db_session():
    """Get database session - would be implemented based on your DB setup"""
    pass

@payment_bp.route('/methods', methods=['GET'])
@jwt_required()
def get_payment_methods():
    """
    Get user's saved payment methods
    """
    try:
        user_id = get_jwt_identity()
        
        db_session = get_db_session()
        payment_service = PaymentService(db_session)
        
        payment_methods = payment_service.get_user_payment_methods(user_id)
        
        methods = []
        for method in payment_methods:
            methods.append({
                'id': method.id,
                'type': method.type.value,
                'is_default': method.is_default,
                'last_four': method.last_four,
                'brand': method.brand,
                'exp_month': method.exp_month,
                'exp_year': method.exp_year,
                'created_at': method.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'payment_methods': methods
        })
        
    except Exception as e:
        logger.error(f"Get payment methods failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@payment_bp.route('/methods/add', methods=['POST'])
@jwt_required()
def add_payment_method():
    """
    Add new payment method for user
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        required_fields = ['payment_method', 'external_id']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate payment method
        try:
            payment_method = PaymentMethod(data['payment_method'])
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid payment method'
            }), 400
        
        db_session = get_db_session()
        payment_service = PaymentService(db_session)
        
        payment_method_record = payment_service.add_payment_method(
            user_id=user_id,
            payment_method=payment_method,
            external_id=data['external_id'],
            metadata=data.get('metadata', {})
        )
        
        if not payment_method_record:
            return jsonify({
                'success': False,
                'error': 'Failed to add payment method'
            }), 500
        
        return jsonify({
            'success': True,
            'payment_method': {
                'id': payment_method_record.id,
                'type': payment_method_record.type.value,
                'is_default': payment_method_record.is_default,
                'last_four': payment_method_record.last_four,
                'brand': payment_method_record.brand
            }
        })
        
    except Exception as e:
        logger.error(f"Add payment method failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@payment_bp.route('/process', methods=['POST'])
@jwt_required()
def process_payment():
    """
    Process one-time payment
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        required_fields = ['amount', 'currency', 'payment_method', 'payment_method_id']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate payment method and currency
        try:
            payment_method = PaymentMethod(data['payment_method'])
            currency = CurrencyCode(data['currency'])
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid parameter: {str(e)}'
            }), 400
        
        db_session = get_db_session()
        payment_service = PaymentService(db_session)
        
        result = payment_service.process_one_time_payment(
            user_id=user_id,
            amount=float(data['amount']),
            currency=currency,
            payment_method=payment_method,
            payment_method_id=data['payment_method_id'],
            description=data.get('description'),
            metadata=data.get('metadata', {})
        )
        
        if result.success:
            return jsonify({
                'success': True,
                'payment_id': result.payment_id,
                'amount_charged': result.amount_charged,
                'external_id': result.external_id
            })
        else:
            return jsonify({
                'success': False,
                'error': result.error_message
            }), 400
        
    except Exception as e:
        logger.error(f"Process payment failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@payment_bp.route('/history', methods=['GET'])
@jwt_required()
def get_payment_history():
    """
    Get payment history for user
    """
    try:
        user_id = get_jwt_identity()
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        db_session = get_db_session()
        payment_service = PaymentService(db_session)
        
        payments = payment_service.get_payment_history(user_id, limit, offset)
        
        payment_history = []
        for payment in payments:
            payment_history.append({
                'id': payment.id,
                'amount': payment.amount,
                'currency': payment.currency.value,
                'status': payment.status.value,
                'payment_method': payment.payment_method.value,
                'description': payment.description,
                'created_at': payment.created_at.isoformat(),
                'processed_at': payment.processed_at.isoformat() if payment.processed_at else None,
                'failure_reason': payment.failure_reason
            })
        
        return jsonify({
            'success': True,
            'payment_history': payment_history,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': len(payment_history)
            }
        })
        
    except Exception as e:
        logger.error(f"Get payment history failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@payment_bp.route('/refund', methods=['POST'])
@jwt_required()
def request_refund():
    """
    Request refund for payment
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if 'payment_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: payment_id'
            }), 400
        
        db_session = get_db_session()
        payment_service = PaymentService(db_session)
        
        result = payment_service.process_refund(
            payment_id=data['payment_id'],
            amount=data.get('amount'),
            reason=data.get('reason', 'Customer request')
        )
        
        if result.success:
            return jsonify({
                'success': True,
                'refund_amount': result.amount_charged,
                'external_id': result.external_id
            })
        else:
            return jsonify({
                'success': False,
                'error': result.error_message
            }), 400
        
    except Exception as e:
        logger.error(f"Request refund failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@payment_bp.route('/webhook/<provider>', methods=['POST'])
def handle_webhook(provider: str):
    """
    Handle payment provider webhooks
    """
    try:
        # Validate provider
        try:
            payment_method = PaymentMethod(provider)
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid provider'
            }), 400
        
        payload = request.get_json()
        signature = request.headers.get('Stripe-Signature') or request.headers.get('X-PayPal-Transmission-Sig')
        
        db_session = get_db_session()
        payment_service = PaymentService(db_session)
        
        success = payment_service.handle_webhook(
            provider=payment_method,
            payload=payload,
            signature=signature
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Webhook processed'})
        else:
            return jsonify({'success': False, 'error': 'Webhook processing failed'}), 500
        
    except Exception as e:
        logger.error(f"Handle webhook failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@payment_bp.route('/supported-methods', methods=['GET'])
def get_supported_payment_methods():
    """
    Get list of supported payment methods and currencies
    """
    try:
        supported_methods = {
            PaymentMethod.STRIPE.value: {
                'name': 'Credit/Debit Card',
                'currencies': [c.value for c in [CurrencyCode.USD, CurrencyCode.EUR, CurrencyCode.GBP, CurrencyCode.INR]],
                'description': 'Secure card payments via Stripe'
            },
            PaymentMethod.PAYPAL.value: {
                'name': 'PayPal',
                'currencies': [c.value for c in [CurrencyCode.USD, CurrencyCode.EUR, CurrencyCode.GBP]],
                'description': 'Pay with your PayPal account'
            },
            PaymentMethod.RAZORPAY.value: {
                'name': 'Razorpay',
                'currencies': [CurrencyCode.INR.value],
                'description': 'Indian payment gateway supporting cards, UPI, wallets'
            },
            PaymentMethod.UPI.value: {
                'name': 'UPI',
                'currencies': [CurrencyCode.INR.value],
                'description': 'Unified Payments Interface (India)'
            }
        }
        
        return jsonify({
            'success': True,
            'supported_methods': supported_methods
        })
        
    except Exception as e:
        logger.error(f"Get supported methods failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@payment_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Payment endpoint not found'}), 404

@payment_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Payment processing error'}), 500