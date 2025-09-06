"""
🏢 DharmaMind Multi-Tenant Architecture

Enterprise-grade multi-tenancy with complete isolation and customization:

Core Features:
- Tenant-based data isolation
- Custom configurations per tenant
- Resource allocation and quotas
- Tenant-specific customizations
- Billing and usage tracking
- Scalable tenant management
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import sqlalchemy.dialects.postgresql as pg

# Tenant configuration
logger = logging.getLogger("dharmamind.tenant")

Base = declarative_base()


class TenantStatus(str, Enum):
    """Tenant status types"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    TRIAL = "trial"
    ENTERPRISE = "enterprise"


class ResourceType(str, Enum):
    """Resource types for quotas"""
    API_CALLS = "api_calls"
    STORAGE = "storage"
    USERS = "users"
    DHARMA_QUERIES = "dharma_queries"
    LLM_TOKENS = "llm_tokens"
    BANDWIDTH = "bandwidth"


@dataclass
class TenantQuota:
    """Tenant resource quota configuration"""
    resource_type: ResourceType
    limit: int
    used: int = 0
    reset_period: str = "monthly"  # daily, weekly, monthly, yearly
    
    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)
    
    @property
    def utilization_percent(self) -> float:
        return (self.used / self.limit * 100) if self.limit > 0 else 0


@dataclass
class TenantConfig:
    """Tenant-specific configuration"""
    dharma_level: str = "universal"  # basic, intermediate, advanced, universal
    llm_providers: List[str] = None
    custom_prompts: Dict[str, str] = None
    ui_branding: Dict[str, Any] = None
    feature_flags: Dict[str, bool] = None
    security_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.llm_providers is None:
            self.llm_providers = ["openai", "anthropic"]
        if self.custom_prompts is None:
            self.custom_prompts = {}
        if self.ui_branding is None:
            self.ui_branding = {}
        if self.feature_flags is None:
            self.feature_flags = {}
        if self.security_settings is None:
            self.security_settings = {}


class TenantModel(Base):
    """SQLAlchemy model for tenant data"""
    __tablename__ = "tenants"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False, default=TenantStatus.TRIAL.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Tenant configuration
    config = Column(JSON, default={})
    quotas = Column(JSON, default={})
    usage_stats = Column(JSON, default={})
    
    # Billing information
    plan_type = Column(String, default="trial")
    billing_email = Column(String)
    subscription_expires = Column(DateTime)
    
    # Custom settings
    custom_domain = Column(String)
    api_key = Column(String)
    webhook_url = Column(String)


class TenantUsageModel(Base):
    """Usage tracking for tenants"""
    __tablename__ = "tenant_usage"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)
    amount = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    usage_metadata = Column(JSON, default={})


class MultiTenantManager:
    """
    Comprehensive multi-tenant management system
    """
    
    def __init__(self, redis_client: redis.Redis, database_url: str):
        self.redis = redis_client
        self.db_engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.db_engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.db_engine)
        
        # Tenant cache
        self.tenant_cache: Dict[str, TenantModel] = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("🏢 Multi-tenant manager initialized")
    
    async def create_tenant(self, 
                          name: str, 
                          billing_email: str,
                          plan_type: str = "trial",
                          config: Optional[TenantConfig] = None) -> str:
        """Create a new tenant"""
        
        tenant_id = str(uuid.uuid4())
        
        # Default configuration
        if config is None:
            config = TenantConfig()
        
        # Default quotas based on plan
        default_quotas = self._get_default_quotas(plan_type)
        
        # Create tenant record
        with self.Session() as session:
            tenant = TenantModel(
                id=tenant_id,
                name=name,
                status=TenantStatus.TRIAL.value,
                plan_type=plan_type,
                billing_email=billing_email,
                config=asdict(config),
                quotas={quota.resource_type.value: asdict(quota) for quota in default_quotas},
                api_key=self._generate_api_key(tenant_id)
            )
            
            session.add(tenant)
            session.commit()
        
        # Cache tenant
        await self._cache_tenant(tenant)
        
        # Initialize tenant resources
        await self._initialize_tenant_resources(tenant_id)
        
        logger.info(f"🏢 Created new tenant: {name} ({tenant_id})")
        return tenant_id
    
    def _get_default_quotas(self, plan_type: str) -> List[TenantQuota]:
        """Get default quotas based on plan type"""
        
        quota_configs = {
            "trial": {
                ResourceType.API_CALLS: 1000,
                ResourceType.DHARMA_QUERIES: 100,
                ResourceType.LLM_TOKENS: 50000,
                ResourceType.USERS: 5,
                ResourceType.STORAGE: 100,  # MB
                ResourceType.BANDWIDTH: 1000  # MB
            },
            "basic": {
                ResourceType.API_CALLS: 10000,
                ResourceType.DHARMA_QUERIES: 1000,
                ResourceType.LLM_TOKENS: 500000,
                ResourceType.USERS: 50,
                ResourceType.STORAGE: 1000,
                ResourceType.BANDWIDTH: 10000
            },
            "professional": {
                ResourceType.API_CALLS: 100000,
                ResourceType.DHARMA_QUERIES: 10000,
                ResourceType.LLM_TOKENS: 5000000,
                ResourceType.USERS: 500,
                ResourceType.STORAGE: 10000,
                ResourceType.BANDWIDTH: 100000
            },
            "enterprise": {
                ResourceType.API_CALLS: -1,  # Unlimited
                ResourceType.DHARMA_QUERIES: -1,
                ResourceType.LLM_TOKENS: -1,
                ResourceType.USERS: -1,
                ResourceType.STORAGE: -1,
                ResourceType.BANDWIDTH: -1
            }
        }
        
        limits = quota_configs.get(plan_type, quota_configs["trial"])
        
        return [
            TenantQuota(resource_type=resource_type, limit=limit)
            for resource_type, limit in limits.items()
        ]
    
    def _generate_api_key(self, tenant_id: str) -> str:
        """Generate secure API key for tenant"""
        import hashlib
        import secrets
        
        # Create unique API key
        salt = secrets.token_hex(16)
        key_data = f"{tenant_id}:{salt}:{datetime.utcnow().isoformat()}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        
        return f"dhm_{tenant_id[:8]}_{api_key[:24]}"
    
    async def _cache_tenant(self, tenant: TenantModel):
        """Cache tenant data in Redis"""
        tenant_key = f"tenant:{tenant.id}"
        tenant_data = {
            "id": tenant.id,
            "name": tenant.name,
            "status": tenant.status,
            "plan_type": tenant.plan_type,
            "config": tenant.config,
            "quotas": tenant.quotas,
            "api_key": tenant.api_key
        }
        
        await self.redis.setex(
            tenant_key,
            self.cache_ttl,
            json.dumps(tenant_data)
        )
        
        # Cache API key mapping
        api_key_key = f"api_key:{tenant.api_key}"
        await self.redis.setex(api_key_key, self.cache_ttl, tenant.id)
    
    async def _initialize_tenant_resources(self, tenant_id: str):
        """Initialize tenant-specific resources"""
        
        # Create tenant-specific Redis namespace
        tenant_ns = f"tenant:{tenant_id}"
        
        # Initialize usage counters
        for resource_type in ResourceType:
            usage_key = f"{tenant_ns}:usage:{resource_type.value}"
            await self.redis.set(usage_key, 0)
        
        # Initialize tenant-specific configurations
        config_key = f"{tenant_ns}:config"
        await self.redis.hset(config_key, mapping={
            "initialized": datetime.utcnow().isoformat(),
            "status": "active"
        })
        
        logger.info(f"🔧 Initialized resources for tenant: {tenant_id}")
    
    async def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant information"""
        
        # Try cache first
        tenant_key = f"tenant:{tenant_id}"
        cached_data = await self.redis.get(tenant_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        # Fetch from database
        with self.Session() as session:
            tenant = session.query(TenantModel).filter_by(id=tenant_id).first()
            
            if tenant:
                # Cache and return
                await self._cache_tenant(tenant)
                return {
                    "id": tenant.id,
                    "name": tenant.name,
                    "status": tenant.status,
                    "plan_type": tenant.plan_type,
                    "config": tenant.config,
                    "quotas": tenant.quotas,
                    "api_key": tenant.api_key
                }
        
        return None
    
    async def get_tenant_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get tenant by API key"""
        
        # Check cache first
        api_key_key = f"api_key:{api_key}"
        tenant_id = await self.redis.get(api_key_key)
        
        if tenant_id:
            return await self.get_tenant(tenant_id)
        
        # Fetch from database
        with self.Session() as session:
            tenant = session.query(TenantModel).filter_by(api_key=api_key).first()
            
            if tenant:
                await self._cache_tenant(tenant)
                return await self.get_tenant(tenant.id)
        
        return None
    
    async def check_quota(self, tenant_id: str, resource_type: ResourceType, amount: int = 1) -> bool:
        """Check if tenant has quota for resource usage"""
        
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        quotas = tenant.get("quotas", {})
        quota_data = quotas.get(resource_type.value)
        
        if not quota_data:
            return False
        
        quota = TenantQuota(**quota_data)
        
        # Unlimited quota
        if quota.limit == -1:
            return True
        
        # Check if usage would exceed limit
        return quota.used + amount <= quota.limit
    
    async def consume_quota(self, tenant_id: str, resource_type: ResourceType, amount: int = 1, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Consume tenant quota and track usage"""
        
        # Check quota first
        if not await self.check_quota(tenant_id, resource_type, amount):
            logger.warning(f"🚫 Quota exceeded for tenant {tenant_id}: {resource_type.value}")
            return False
        
        # Update usage in Redis
        tenant_ns = f"tenant:{tenant_id}"
        usage_key = f"{tenant_ns}:usage:{resource_type.value}"
        new_usage = await self.redis.incrby(usage_key, amount)
        
        # Update database quota
        with self.Session() as session:
            tenant = session.query(TenantModel).filter_by(id=tenant_id).first()
            if tenant:
                quotas = tenant.quotas or {}
                if resource_type.value in quotas:
                    quotas[resource_type.value]["used"] = new_usage
                    tenant.quotas = quotas
                    session.commit()
        
        # Record usage event
        await self._record_usage_event(tenant_id, resource_type, amount, metadata)
        
        # Invalidate cache
        tenant_key = f"tenant:{tenant_id}"
        await self.redis.delete(tenant_key)
        
        logger.debug(f"📊 Consumed quota for {tenant_id}: {resource_type.value} +{amount}")
        return True
    
    async def _record_usage_event(self, tenant_id: str, resource_type: ResourceType, amount: int, metadata: Optional[Dict[str, Any]]):
        """Record usage event for analytics"""
        
        usage_id = str(uuid.uuid4())
        
        with self.Session() as session:
            usage = TenantUsageModel(
                id=usage_id,
                tenant_id=tenant_id,
                resource_type=resource_type.value,
                amount=amount,
                usage_metadata=metadata or {}
            )
            
            session.add(usage)
            session.commit()
        
        # Also store in Redis for real-time analytics
        usage_key = f"tenant:{tenant_id}:events:{datetime.now().strftime('%Y-%m-%d')}"
        event_data = {
            "id": usage_id,
            "resource_type": resource_type.value,
            "amount": amount,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        await self.redis.lpush(usage_key, json.dumps(event_data))
        await self.redis.expire(usage_key, 86400 * 30)  # 30 days
    
    async def get_tenant_usage(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Get tenant usage statistics"""
        
        # Get current quotas
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}
        
        quotas = tenant.get("quotas", {})
        
        # Get usage events from database
        with self.Session() as session:
            start_date = datetime.utcnow() - timedelta(days=days)
            usage_events = session.query(TenantUsageModel).filter(
                TenantUsageModel.tenant_id == tenant_id,
                TenantUsageModel.timestamp >= start_date
            ).all()
        
        # Aggregate usage by resource type
        usage_summary = {}
        for resource_type in ResourceType:
            quota_data = quotas.get(resource_type.value, {})
            quota = TenantQuota(**quota_data) if quota_data else None
            
            # Calculate usage from events
            events = [e for e in usage_events if e.resource_type == resource_type.value]
            total_usage = sum(e.amount for e in events)
            
            usage_summary[resource_type.value] = {
                "current_usage": quota.used if quota else 0,
                "limit": quota.limit if quota else 0,
                "remaining": quota.remaining if quota else 0,
                "utilization_percent": quota.utilization_percent if quota else 0,
                "period_usage": total_usage,
                "event_count": len(events)
            }
        
        return {
            "tenant_id": tenant_id,
            "period_days": days,
            "usage_summary": usage_summary,
            "total_events": len(usage_events)
        }
    
    async def update_tenant_config(self, tenant_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update tenant configuration"""
        
        with self.Session() as session:
            tenant = session.query(TenantModel).filter_by(id=tenant_id).first()
            
            if not tenant:
                return False
            
            # Update configuration
            current_config = tenant.config or {}
            current_config.update(config_updates)
            tenant.config = current_config
            tenant.updated_at = datetime.utcnow()
            
            session.commit()
        
        # Invalidate cache
        tenant_key = f"tenant:{tenant_id}"
        await self.redis.delete(tenant_key)
        
        logger.info(f"🔧 Updated config for tenant: {tenant_id}")
        return True
    
    async def list_tenants(self, status: Optional[TenantStatus] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List tenants with optional filtering"""
        
        with self.Session() as session:
            query = session.query(TenantModel)
            
            if status:
                query = query.filter(TenantModel.status == status.value)
            
            query = query.limit(limit)
            tenants = query.all()
            
            return [
                {
                    "id": tenant.id,
                    "name": tenant.name,
                    "status": tenant.status,
                    "plan_type": tenant.plan_type,
                    "created_at": tenant.created_at.isoformat(),
                    "billing_email": tenant.billing_email
                }
                for tenant in tenants
            ]
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant and all associated data"""
        
        # Delete from database
        with self.Session() as session:
            tenant = session.query(TenantModel).filter_by(id=tenant_id).first()
            
            if not tenant:
                return False
            
            # Delete usage records
            session.query(TenantUsageModel).filter_by(tenant_id=tenant_id).delete()
            
            # Delete tenant
            session.delete(tenant)
            session.commit()
        
        # Delete from Redis cache
        tenant_key = f"tenant:{tenant_id}"
        api_key_key = f"api_key:{tenant.api_key}" if tenant else None
        
        await self.redis.delete(tenant_key)
        if api_key_key:
            await self.redis.delete(api_key_key)
        
        # Delete tenant namespace in Redis
        tenant_ns_pattern = f"tenant:{tenant_id}:*"
        keys = await self.redis.keys(tenant_ns_pattern)
        if keys:
            await self.redis.delete(*keys)
        
        logger.info(f"🗑️ Deleted tenant: {tenant_id}")
        return True


# Middleware for tenant isolation
class TenantIsolationMiddleware:
    """Middleware to handle tenant isolation and context"""
    
    def __init__(self, tenant_manager: MultiTenantManager):
        self.tenant_manager = tenant_manager
    
    async def __call__(self, request, call_next):
        """Process request with tenant context"""
        
        # Extract tenant information
        tenant_id = None
        tenant_data = None
        
        # Method 1: API Key in header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        
        if api_key:
            tenant_data = await self.tenant_manager.get_tenant_by_api_key(api_key)
            if tenant_data:
                tenant_id = tenant_data["id"]
        
        # Method 2: Tenant ID in header (for internal use)
        if not tenant_id:
            tenant_id = request.headers.get("X-Tenant-ID")
            if tenant_id:
                tenant_data = await self.tenant_manager.get_tenant(tenant_id)
        
        # Method 3: Subdomain (if using custom domains)
        if not tenant_id:
            host = request.headers.get("Host", "")
            if "." in host:
                subdomain = host.split(".")[0]
                # Look up tenant by subdomain logic here
                pass
        
        # Set tenant context
        if tenant_data:
            request.state.tenant_id = tenant_id
            request.state.tenant_data = tenant_data
            request.state.tenant_config = tenant_data.get("config", {})
        else:
            # Default tenant or public access
            request.state.tenant_id = None
            request.state.tenant_data = None
            request.state.tenant_config = {}
        
        # Process request
        response = await call_next(request)
        
        # Add tenant info to response headers (if needed)
        if tenant_id:
            response.headers["X-Tenant-ID"] = tenant_id
        
        return response


# Dependency injection for FastAPI
async def get_current_tenant(request) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to get current tenant"""
    return getattr(request.state, "tenant_data", None)


async def require_tenant(request) -> Dict[str, Any]:
    """FastAPI dependency that requires a valid tenant"""
    tenant_data = getattr(request.state, "tenant_data", None)
    if not tenant_data:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Valid tenant required")
    return tenant_data


# Export main components
__all__ = [
    "MultiTenantManager",
    "TenantModel",
    "TenantUsageModel",
    "TenantConfig",
    "TenantQuota",
    "TenantStatus",
    "ResourceType",
    "TenantIsolationMiddleware",
    "get_current_tenant",
    "require_tenant"
]
