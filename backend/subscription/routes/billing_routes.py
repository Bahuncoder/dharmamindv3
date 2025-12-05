"""
Billing and Invoice API Routes for DharmaMind
============================================

RESTful API endpoints for billing management, invoice generation,
and billing analytics.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging

from ..services.billing_service import BillingService

logger = logging.getLogger(__name__)

billing_bp = Blueprint('billing', __name__, url_prefix='/api/billing')

def get_db_session():
    """Get database session - would be implemented based on your DB setup"""
    pass

@billing_bp.route('/summary', methods=['GET'])
@jwt_required()
def get_billing_summary():
    """
    Get billing summary for user
    """
    try:
        user_id = get_jwt_identity()
        
        db_session = get_db_session()
        billing_service = BillingService(db_session)
        
        summary = billing_service.get_billing_summary(user_id)
        
        return jsonify({
            'success': True,
            'billing_summary': summary
        })
        
    except Exception as e:
        logger.error(f"Get billing summary failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@billing_bp.route('/invoices', methods=['GET'])
@jwt_required()
def get_invoice_history():
    """
    Get invoice history for user
    """
    try:
        user_id = get_jwt_identity()
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        db_session = get_db_session()
        billing_service = BillingService(db_session)
        
        invoices = billing_service.get_invoice_history(user_id, limit, offset)
        
        invoice_history = []
        for invoice in invoices:
            invoice_history.append({
                'id': invoice.id,
                'invoice_number': invoice.invoice_number,
                'amount': invoice.amount,
                'tax_amount': invoice.tax_amount,
                'total_amount': invoice.total_amount,
                'currency': invoice.currency.value,
                'status': invoice.status,
                'issue_date': invoice.issue_date.isoformat() if invoice.issue_date else None,
                'due_date': invoice.due_date.isoformat(),
                'paid_date': invoice.paid_date.isoformat() if invoice.paid_date else None,
                'billing_period_start': invoice.billing_period_start.isoformat() if invoice.billing_period_start else None,
                'billing_period_end': invoice.billing_period_end.isoformat() if invoice.billing_period_end else None,
                'line_items': invoice.line_items
            })
        
        return jsonify({
            'success': True,
            'invoice_history': invoice_history,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': len(invoice_history)
            }
        })
        
    except Exception as e:
        logger.error(f"Get invoice history failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@billing_bp.route('/invoices/<invoice_id>', methods=['GET'])
@jwt_required()
def get_invoice_details(invoice_id: str):
    """
    Get detailed invoice information
    """
    try:
        user_id = get_jwt_identity()
        
        db_session = get_db_session()
        from ..models.subscription_models import Invoice
        
        invoice = db_session.query(Invoice).filter(
            and_(
                Invoice.id == invoice_id,
                Invoice.user_id == user_id
            )
        ).first()
        
        if not invoice:
            return jsonify({
                'success': False,
                'error': 'Invoice not found'
            }), 404
        
        return jsonify({
            'success': True,
            'invoice': {
                'id': invoice.id,
                'invoice_number': invoice.invoice_number,
                'amount': invoice.amount,
                'tax_amount': invoice.tax_amount,
                'total_amount': invoice.total_amount,
                'currency': invoice.currency.value,
                'status': invoice.status,
                'issue_date': invoice.issue_date.isoformat() if invoice.issue_date else None,
                'due_date': invoice.due_date.isoformat(),
                'paid_date': invoice.paid_date.isoformat() if invoice.paid_date else None,
                'billing_period_start': invoice.billing_period_start.isoformat() if invoice.billing_period_start else None,
                'billing_period_end': invoice.billing_period_end.isoformat() if invoice.billing_period_end else None,
                'line_items': invoice.line_items,
                'notes': invoice.notes,
                'metadata': invoice.metadata
            }
        })
        
    except Exception as e:
        logger.error(f"Get invoice details failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@billing_bp.route('/invoices/<invoice_id>/download', methods=['GET'])
@jwt_required()
def download_invoice(invoice_id: str):
    """
    Download invoice as PDF
    """
    try:
        user_id = get_jwt_identity()
        
        # This would generate and return a PDF invoice
        # For now, return invoice data that could be used to generate PDF
        
        db_session = get_db_session()
        from ..models.subscription_models import Invoice
        
        invoice = db_session.query(Invoice).filter(
            and_(
                Invoice.id == invoice_id,
                Invoice.user_id == user_id
            )
        ).first()
        
        if not invoice:
            return jsonify({
                'success': False,
                'error': 'Invoice not found'
            }), 404
        
        # Return invoice data for PDF generation
        return jsonify({
            'success': True,
            'message': 'PDF generation would be implemented here',
            'invoice_data': {
                'id': invoice.id,
                'invoice_number': invoice.invoice_number,
                'total_amount': invoice.total_amount,
                'currency': invoice.currency.value,
                'due_date': invoice.due_date.isoformat(),
                'line_items': invoice.line_items
            }
        })
        
    except Exception as e:
        logger.error(f"Download invoice failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Admin endpoints for billing management

@billing_bp.route('/admin/process-billing', methods=['POST'])
@jwt_required()
def process_automatic_billing():
    """
    Trigger automatic billing process (admin only)
    """
    try:
        # Would check if user is admin here
        
        db_session = get_db_session()
        billing_service = BillingService(db_session)
        
        results = billing_service.process_automatic_billing()
        
        return jsonify({
            'success': True,
            'billing_results': results
        })
        
    except Exception as e:
        logger.error(f"Process automatic billing failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@billing_bp.route('/admin/analytics', methods=['GET'])
@jwt_required()
def get_billing_analytics():
    """
    Get billing analytics (admin only)
    """
    try:
        # Would check if user is admin here
        
        db_session = get_db_session()
        from ..services.analytics_service import SubscriptionAnalyticsService
        
        analytics_service = SubscriptionAnalyticsService(db_session)
        revenue_analytics = analytics_service.get_revenue_analytics()
        
        return jsonify({
            'success': True,
            'billing_analytics': revenue_analytics
        })
        
    except Exception as e:
        logger.error(f"Get billing analytics failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@billing_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Billing endpoint not found'}), 404

@billing_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Billing processing error'}), 500