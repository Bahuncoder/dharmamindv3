"""
Comprehensive Subscription API Routes for DharmaMind
==================================================

RESTful API endpoints for subscription management including
creation, upgrades, downgrades, cancellation, and analytics.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import logging

from ..services.subscription_service import SubscriptionService
from ..services.usage_service import UsageService
from ..models.subscription_models import SubscriptionTier, BillingCycle

logger = logging.getLogger(__name__)

subscription_bp = Blueprint('subscription', __name__, url_prefix='/api/subscription')

def get_db_session():
    """Get database session - would be implemented based on your DB setup"""
    # This would return your SQLAlchemy session
    # return db.session
    pass

@subscription_bp.route('/tiers', methods=['GET'])
def get_subscription_tiers():
    """
    Get available subscription tiers and pricing
    """
    try:
        pricing = {
            SubscriptionTier.FREE.value: {
                'name': 'Seeker',
                'description': 'Basic spiritual guidance and limited features',
                'monthly_price': 0,
                'annual_price': 0,
                'features': {
                    'daily_messages': 10,
                    'ai_models': ['basic'],
                    'chat_history_days': 7,
                    'export_chats': False,
                    'priority_support': False,
                    'dharma_insights': 'basic',
                    'meditation_sessions': 3,
                    'scripture_access': 'limited'
                }
            },
            SubscriptionTier.PREMIUM.value: {
                'name': 'Devotee',
                'description': 'Enhanced spiritual features and unlimited access',
                'monthly_price': 19.99,
                'annual_price': 199.99,
                'annual_savings': 41.88,
                'features': {
                    'daily_messages': 200,
                    'ai_models': ['basic', 'advanced'],
                    'chat_history_days': 90,
                    'export_chats': True,
                    'priority_support': True,
                    'dharma_insights': 'advanced',
                    'meditation_sessions': 20,
                    'scripture_access': 'full',
                    'personalized_guidance': True,
                    'voice_interaction': True
                }
            },
            SubscriptionTier.ENTERPRISE.value: {
                'name': 'Guru',
                'description': 'Complete spiritual platform with unlimited access',
                'monthly_price': 99.99,
                'annual_price': 999.99,
                'annual_savings': 199.89,
                'features': {
                    'daily_messages': -1,  # unlimited
                    'ai_models': ['basic', 'advanced', 'premium'],
                    'chat_history_days': -1,
                    'export_chats': True,
                    'priority_support': True,
                    'dharma_insights': 'premium',
                    'meditation_sessions': -1,
                    'scripture_access': 'complete',
                    'personalized_guidance': True,
                    'voice_interaction': True,
                    'team_management': True,
                    'white_labeling': True,
                    'dedicated_support': True,
                    'api_access': 'full'
                }
            },
            SubscriptionTier.FAMILY.value: {
                'name': 'Ashram',
                'description': 'Family sharing plan with multiple user accounts',
                'monthly_price': 29.99,
                'annual_price': 299.99,
                'annual_savings': 59.89,
                'features': {
                    'daily_messages': 500,
                    'family_members': 5,
                    'shared_libraries': True,
                    'family_analytics': True
                }
            },
            SubscriptionTier.STUDENT.value: {
                'name': 'Shishya',
                'description': 'Educational discount for students and researchers',
                'monthly_price': 9.99,
                'annual_price': 99.99,
                'verification_required': True,
                'discount_percentage': 50
            }
        }
        
        return jsonify({
            'success': True,
            'tiers': pricing,
            'currency': 'USD'
        })
        
    except Exception as e:
        logger.error(f"Get subscription tiers failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/create', methods=['POST'])
@jwt_required()
def create_subscription():
    """
    Create new subscription for user
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        required_fields = ['tier']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate tier
        try:
            tier = SubscriptionTier(data['tier'])
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid subscription tier'
            }), 400
        
        # Validate billing cycle
        billing_cycle = BillingCycle.MONTHLY
        if 'billing_cycle' in data:
            try:
                billing_cycle = BillingCycle(data['billing_cycle'])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid billing cycle'
                }), 400
        
        # Create subscription
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        subscription = await subscription_service.create_subscription(
            user_id=user_id,
            tier=tier,
            billing_cycle=billing_cycle,
            payment_method_id=data.get('payment_method_id'),
            trial_days=data.get('trial_days', 0),
            discount_code=data.get('discount_code'),
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            'success': True,
            'subscription': {
                'id': subscription.id,
                'tier': subscription.tier.value,
                'status': subscription.status.value,
                'current_price': subscription.current_price,
                'billing_cycle': subscription.billing_cycle.value,
                'trial_end': subscription.trial_end.isoformat() if subscription.trial_end else None,
                'current_period_end': subscription.current_period_end.isoformat() if subscription.current_period_end else None
            }
        })
        
    except Exception as e:
        logger.error(f"Create subscription failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/current', methods=['GET'])
@jwt_required()
def get_current_subscription():
    """
    Get user's current active subscription
    """
    try:
        user_id = get_jwt_identity()
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        subscription = subscription_service.get_user_subscription(user_id)
        
        if not subscription:
            return jsonify({
                'success': True,
                'subscription': None,
                'message': 'No active subscription found'
            })
        
        # Get usage summary
        usage_service = UsageService(db_session)
        usage_summary = usage_service.get_usage_summary(user_id)
        
        return jsonify({
            'success': True,
            'subscription': {
                'id': subscription.id,
                'tier': subscription.tier.value,
                'status': subscription.status.value,
                'current_price': subscription.current_price,
                'billing_cycle': subscription.billing_cycle.value,
                'created_at': subscription.created_at.isoformat(),
                'current_period_start': subscription.current_period_start.isoformat() if subscription.current_period_start else None,
                'current_period_end': subscription.current_period_end.isoformat() if subscription.current_period_end else None,
                'trial_end': subscription.trial_end.isoformat() if subscription.trial_end else None,
                'days_until_renewal': subscription.days_until_renewal(),
                'is_trial': subscription.is_trial(),
                'features': subscription.get_tier_features(),
                'usage_summary': usage_summary
            }
        })
        
    except Exception as e:
        logger.error(f"Get current subscription failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/upgrade', methods=['POST'])
@jwt_required()
def upgrade_subscription():
    """
    Upgrade user's subscription to higher tier
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if 'new_tier' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: new_tier'
            }), 400
        
        try:
            new_tier = SubscriptionTier(data['new_tier'])
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid subscription tier'
            }), 400
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        # Get current subscription
        current_subscription = subscription_service.get_user_subscription(user_id)
        if not current_subscription:
            return jsonify({
                'success': False,
                'error': 'No active subscription found'
            }), 404
        
        # Perform upgrade
        subscription, proration_amount = await subscription_service.upgrade_subscription(
            subscription_id=current_subscription.id,
            new_tier=new_tier,
            immediate=data.get('immediate', True)
        )
        
        return jsonify({
            'success': True,
            'subscription': {
                'id': subscription.id,
                'tier': subscription.tier.value,
                'current_price': subscription.current_price,
                'proration_amount': proration_amount
            },
            'message': f'Successfully upgraded to {new_tier.value} tier'
        })
        
    except Exception as e:
        logger.error(f"Upgrade subscription failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/downgrade', methods=['POST'])
@jwt_required()
def downgrade_subscription():
    """
    Downgrade user's subscription to lower tier
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if 'new_tier' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: new_tier'
            }), 400
        
        try:
            new_tier = SubscriptionTier(data['new_tier'])
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid subscription tier'
            }), 400
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        # Get current subscription
        current_subscription = subscription_service.get_user_subscription(user_id)
        if not current_subscription:
            return jsonify({
                'success': False,
                'error': 'No active subscription found'
            }), 404
        
        # Perform downgrade
        subscription = await subscription_service.downgrade_subscription(
            subscription_id=current_subscription.id,
            new_tier=new_tier,
            immediate=data.get('immediate', False)
        )
        
        effective_date = datetime.utcnow() if data.get('immediate', False) else subscription.current_period_end
        
        return jsonify({
            'success': True,
            'subscription': {
                'id': subscription.id,
                'tier': subscription.tier.value if data.get('immediate', False) else current_subscription.tier.value,
                'scheduled_tier': new_tier.value if not data.get('immediate', False) else None,
                'effective_date': effective_date.isoformat() if effective_date else None
            },
            'message': f'Downgrade to {new_tier.value} tier scheduled for {effective_date.strftime("%B %d, %Y") if effective_date else "immediately"}'
        })
        
    except Exception as e:
        logger.error(f"Downgrade subscription failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/cancel', methods=['POST'])
@jwt_required()
def cancel_subscription():
    """
    Cancel user's subscription
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        # Get current subscription
        current_subscription = subscription_service.get_user_subscription(user_id)
        if not current_subscription:
            return jsonify({
                'success': False,
                'error': 'No active subscription found'
            }), 404
        
        # Cancel subscription
        subscription = await subscription_service.cancel_subscription(
            subscription_id=current_subscription.id,
            immediate=data.get('immediate', False),
            reason=data.get('reason', 'user_request')
        )
        
        return jsonify({
            'success': True,
            'subscription': {
                'id': subscription.id,
                'status': subscription.status.value,
                'cancelled_at': subscription.cancelled_at.isoformat() if subscription.cancelled_at else None,
                'ended_at': subscription.ended_at.isoformat() if subscription.ended_at else None
            },
            'message': 'Subscription cancelled successfully'
        })
        
    except Exception as e:
        logger.error(f"Cancel subscription failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/pause', methods=['POST'])
@jwt_required()
def pause_subscription():
    """
    Pause user's subscription temporarily
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        # Get current subscription
        current_subscription = subscription_service.get_user_subscription(user_id)
        if not current_subscription:
            return jsonify({
                'success': False,
                'error': 'No active subscription found'
            }), 404
        
        # Parse pause_until if provided
        pause_until = None
        if 'pause_until' in data:
            try:
                pause_until = datetime.fromisoformat(data['pause_until'])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid pause_until date format'
                }), 400
        
        # Pause subscription
        subscription = await subscription_service.pause_subscription(
            subscription_id=current_subscription.id,
            pause_until=pause_until
        )
        
        return jsonify({
            'success': True,
            'subscription': {
                'id': subscription.id,
                'status': subscription.status.value,
                'pause_until': pause_until.isoformat() if pause_until else None
            },
            'message': 'Subscription paused successfully'
        })
        
    except Exception as e:
        logger.error(f"Pause subscription failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/resume', methods=['POST'])
@jwt_required()
def resume_subscription():
    """
    Resume a paused subscription
    """
    try:
        user_id = get_jwt_identity()
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        # Get current subscription
        current_subscription = subscription_service.get_user_subscription(user_id)
        if not current_subscription:
            return jsonify({
                'success': False,
                'error': 'No subscription found'
            }), 404
        
        # Resume subscription
        subscription = await subscription_service.resume_subscription(
            subscription_id=current_subscription.id
        )
        
        return jsonify({
            'success': True,
            'subscription': {
                'id': subscription.id,
                'status': subscription.status.value,
                'current_period_start': subscription.current_period_start.isoformat(),
                'current_period_end': subscription.current_period_end.isoformat()
            },
            'message': 'Subscription resumed successfully'
        })
        
    except Exception as e:
        logger.error(f"Resume subscription failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/usage', methods=['GET'])
@jwt_required()
def get_usage_summary():
    """
    Get usage summary for current user
    """
    try:
        user_id = get_jwt_identity()
        period_days = request.args.get('period_days', 30, type=int)
        
        db_session = get_db_session()
        usage_service = UsageService(db_session)
        
        usage_summary = usage_service.get_usage_summary(user_id, period_days)
        
        return jsonify({
            'success': True,
            'usage_summary': usage_summary
        })
        
    except Exception as e:
        logger.error(f"Get usage summary failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/usage/check', methods=['POST'])
@jwt_required()
def check_usage_limit():
    """
    Check if user can perform action within usage limits
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if 'feature' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: feature'
            }), 400
        
        db_session = get_db_session()
        usage_service = UsageService(db_session)
        
        allowed, details = await usage_service.check_usage_limit(
            user_id=user_id,
            feature=data['feature'],
            requested_count=data.get('count', 1)
        )
        
        return jsonify({
            'success': True,
            'allowed': allowed,
            'usage_details': details
        })
        
    except Exception as e:
        logger.error(f"Check usage limit failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/history', methods=['GET'])
@jwt_required()
def get_subscription_history():
    """
    Get subscription history for user
    """
    try:
        user_id = get_jwt_identity()
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        subscriptions = subscription_service.get_subscription_history(user_id)
        
        history = []
        for sub in subscriptions:
            history.append({
                'id': sub.id,
                'tier': sub.tier.value,
                'status': sub.status.value,
                'created_at': sub.created_at.isoformat(),
                'cancelled_at': sub.cancelled_at.isoformat() if sub.cancelled_at else None,
                'current_price': sub.current_price,
                'billing_cycle': sub.billing_cycle.value
            })
        
        return jsonify({
            'success': True,
            'subscription_history': history
        })
        
    except Exception as e:
        logger.error(f"Get subscription history failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.route('/feature-access/<feature>', methods=['GET'])
@jwt_required()
def check_feature_access(feature: str):
    """
    Check if user has access to specific feature
    """
    try:
        user_id = get_jwt_identity()
        
        db_session = get_db_session()
        subscription_service = SubscriptionService(db_session)
        
        has_access = subscription_service.check_feature_access(user_id, feature)
        
        return jsonify({
            'success': True,
            'feature': feature,
            'has_access': has_access
        })
        
    except Exception as e:
        logger.error(f"Check feature access failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Admin endpoints (would require admin authentication)

@subscription_bp.route('/admin/analytics/overview', methods=['GET'])
@jwt_required()
def get_analytics_overview():
    """
    Get subscription analytics overview (admin only)
    """
    try:
        # Would check if user is admin here
        
        db_session = get_db_session()
        from ..services.analytics_service import SubscriptionAnalyticsService
        analytics_service = SubscriptionAnalyticsService(db_session)
        
        overview = analytics_service.get_subscription_overview()
        
        return jsonify({
            'success': True,
            'analytics': overview
        })
        
    except Exception as e:
        logger.error(f"Get analytics overview failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@subscription_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@subscription_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500