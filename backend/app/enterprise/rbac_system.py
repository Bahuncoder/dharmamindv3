"""
🔐 DharmaMind Advanced RBAC & Permissions System

Enterprise-grade role-based access control with fine-grained permissions:

Core Features:
- Hierarchical role system
- Fine-grained permission controls
- Resource-based access control
- Dynamic permission evaluation
- Audit logging for security
- Multi-tenant permission isolation
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Boolean, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID

# RBAC configuration
logger = logging.getLogger("dharmamind.rbac")

Base = declarative_base()

# Association tables for many-to-many relationships
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String, ForeignKey('users.id')),
    Column('role_id', String, ForeignKey('roles.id'))
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', String, ForeignKey('roles.id')),
    Column('permission_id', String, ForeignKey('permissions.id'))
)


class ActionType(str, Enum):
    """Types of actions that can be performed"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Types of resources that can be controlled"""
    DHARMA_QUERY = "dharma_query"
    CHAT_SESSION = "chat_session"
    USER_PROFILE = "user_profile"
    SYSTEM_CONFIG = "system_config"
    ANALYTICS = "analytics"
    BILLING = "billing"
    TENANT = "tenant"
    API_KEY = "api_key"
    DASHBOARD = "dashboard"
    WISDOM_RATING = "wisdom_rating"


class PermissionScope(str, Enum):
    """Scope of permissions"""
    GLOBAL = "global"  # Across all tenants (super admin)
    TENANT = "tenant"  # Within a specific tenant
    USER = "user"     # User's own resources only
    CUSTOM = "custom"  # Custom resource-specific scope


@dataclass
class Permission:
    """Permission definition"""
    id: str
    name: str
    description: str
    resource_type: ResourceType
    action: ActionType
    scope: PermissionScope
    conditions: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


@dataclass
class Role:
    """Role definition with permissions"""
    id: str
    name: str
    description: str
    permissions: List[Permission]
    is_system_role: bool = False
    parent_role_id: Optional[str] = None
    tenant_id: Optional[str] = None


class PermissionModel(Base):
    """SQLAlchemy model for permissions"""
    __tablename__ = "permissions"
    
    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    resource_type = Column(String, nullable=False)
    action = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    conditions = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class RoleModel(Base):
    """SQLAlchemy model for roles"""
    __tablename__ = "roles"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    is_system_role = Column(Boolean, default=False)
    parent_role_id = Column(String, ForeignKey('roles.id'), nullable=True)
    tenant_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    permissions = relationship("PermissionModel", secondary=role_permissions, backref="roles")
    parent = relationship("RoleModel", remote_side="RoleModel.id", backref="children")


class UserModel(Base):
    """SQLAlchemy model for users with RBAC"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    tenant_id = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    roles = relationship("RoleModel", secondary=user_roles, backref="users")


class AccessLogModel(Base):
    """Access log for security auditing"""
    __tablename__ = "access_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    tenant_id = Column(String, nullable=True)
    resource_type = Column(String, nullable=False)
    action = Column(String, nullable=False)
    resource_id = Column(String, nullable=True)
    result = Column(String, nullable=False)  # allowed, denied
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)
    user_agent = Column(String)
    request_metadata = Column(JSON, default={})


class AdvancedRBACManager:
    """
    Advanced Role-Based Access Control system
    """
    
    def __init__(self, redis_client: redis.Redis, database_url: str):
        self.redis = redis_client
        self.db_engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.db_engine)
        
        # Create tables
        Base.metadata.create_all(self.db_engine)
        
        # Permission cache
        self.permission_cache: Dict[str, List[Permission]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize default permissions and roles
        self._initialize_default_rbac()
        
        logger.info("🔐 Advanced RBAC manager initialized")
    
    def _initialize_default_rbac(self):
        """Initialize default permissions and roles"""
        
        with self.Session() as session:
            # Check if already initialized
            if session.query(PermissionModel).count() > 0:
                return
            
            # Create default permissions
            default_permissions = self._get_default_permissions()
            
            for perm in default_permissions:
                permission = PermissionModel(
                    id=perm.id,
                    name=perm.name,
                    description=perm.description,
                    resource_type=perm.resource_type.value,
                    action=perm.action.value,
                    scope=perm.scope.value,
                    conditions=perm.conditions
                )
                session.add(permission)
            
            session.commit()
            
            # Create default roles
            default_roles = self._get_default_roles()
            
            for role_data in default_roles:
                role = RoleModel(
                    id=role_data["id"],
                    name=role_data["name"],
                    description=role_data["description"],
                    is_system_role=True
                )
                
                # Add permissions to role
                for perm_id in role_data["permission_ids"]:
                    permission = session.query(PermissionModel).filter_by(id=perm_id).first()
                    if permission:
                        role.permissions.append(permission)
                
                session.add(role)
            
            session.commit()
            
            logger.info("🔧 Initialized default RBAC system")
    
    def _get_default_permissions(self) -> List[Permission]:
        """Get default system permissions"""
        
        permissions = []
        
        # Dharma query permissions
        permissions.extend([
            Permission("dharma.query.create", "Create Dharma Query", "Create new dharma queries", 
                      ResourceType.DHARMA_QUERY, ActionType.CREATE, PermissionScope.USER),
            Permission("dharma.query.read", "Read Dharma Query", "View dharma queries", 
                      ResourceType.DHARMA_QUERY, ActionType.READ, PermissionScope.USER),
            Permission("dharma.query.admin", "Admin Dharma Queries", "Manage all dharma queries", 
                      ResourceType.DHARMA_QUERY, ActionType.ADMIN, PermissionScope.TENANT),
        ])
        
        # Chat session permissions
        permissions.extend([
            Permission("chat.session.create", "Create Chat Session", "Start new chat sessions", 
                      ResourceType.CHAT_SESSION, ActionType.CREATE, PermissionScope.USER),
            Permission("chat.session.read", "Read Chat Session", "View chat sessions", 
                      ResourceType.CHAT_SESSION, ActionType.READ, PermissionScope.USER),
            Permission("chat.session.manage", "Manage Chat Sessions", "Manage all chat sessions", 
                      ResourceType.CHAT_SESSION, ActionType.MANAGE, PermissionScope.TENANT),
        ])
        
        # User profile permissions
        permissions.extend([
            Permission("user.profile.read", "Read User Profile", "View user profiles", 
                      ResourceType.USER_PROFILE, ActionType.READ, PermissionScope.USER),
            Permission("user.profile.update", "Update User Profile", "Update user profiles", 
                      ResourceType.USER_PROFILE, ActionType.UPDATE, PermissionScope.USER),
            Permission("user.profile.admin", "Admin User Profiles", "Manage all user profiles", 
                      ResourceType.USER_PROFILE, ActionType.ADMIN, PermissionScope.TENANT),
        ])
        
        # System configuration permissions
        permissions.extend([
            Permission("system.config.read", "Read System Config", "View system configuration", 
                      ResourceType.SYSTEM_CONFIG, ActionType.READ, PermissionScope.TENANT),
            Permission("system.config.update", "Update System Config", "Update system configuration", 
                      ResourceType.SYSTEM_CONFIG, ActionType.UPDATE, PermissionScope.TENANT),
            Permission("system.config.admin", "Admin System Config", "Full system configuration access", 
                      ResourceType.SYSTEM_CONFIG, ActionType.ADMIN, PermissionScope.GLOBAL),
        ])
        
        # Analytics permissions
        permissions.extend([
            Permission("analytics.read", "Read Analytics", "View analytics data", 
                      ResourceType.ANALYTICS, ActionType.READ, PermissionScope.TENANT),
            Permission("analytics.admin", "Admin Analytics", "Manage analytics system", 
                      ResourceType.ANALYTICS, ActionType.ADMIN, PermissionScope.GLOBAL),
        ])
        
        # Billing permissions
        permissions.extend([
            Permission("billing.read", "Read Billing", "View billing information", 
                      ResourceType.BILLING, ActionType.READ, PermissionScope.TENANT),
            Permission("billing.manage", "Manage Billing", "Manage billing and subscriptions", 
                      ResourceType.BILLING, ActionType.MANAGE, PermissionScope.TENANT),
        ])
        
        # Tenant management permissions
        permissions.extend([
            Permission("tenant.read", "Read Tenant", "View tenant information", 
                      ResourceType.TENANT, ActionType.READ, PermissionScope.TENANT),
            Permission("tenant.admin", "Admin Tenants", "Manage all tenants", 
                      ResourceType.TENANT, ActionType.ADMIN, PermissionScope.GLOBAL),
        ])
        
        return permissions
    
    def _get_default_roles(self) -> List[Dict[str, Any]]:
        """Get default system roles"""
        
        return [
            {
                "id": "super_admin",
                "name": "Super Admin",
                "description": "Full system access across all tenants",
                "permission_ids": [
                    "system.config.admin", "analytics.admin", "tenant.admin"
                ]
            },
            {
                "id": "tenant_admin",
                "name": "Tenant Admin",
                "description": "Full access within tenant",
                "permission_ids": [
                    "dharma.query.admin", "chat.session.manage", "user.profile.admin",
                    "system.config.read", "system.config.update", "analytics.read",
                    "billing.read", "billing.manage", "tenant.read"
                ]
            },
            {
                "id": "dharma_teacher",
                "name": "Dharma Teacher",
                "description": "Advanced dharma guidance capabilities",
                "permission_ids": [
                    "dharma.query.create", "dharma.query.read", "chat.session.create",
                    "chat.session.read", "user.profile.read", "user.profile.update",
                    "analytics.read"
                ]
            },
            {
                "id": "student",
                "name": "Student",
                "description": "Basic user access for learning",
                "permission_ids": [
                    "dharma.query.create", "dharma.query.read", "chat.session.create",
                    "chat.session.read", "user.profile.read", "user.profile.update"
                ]
            },
            {
                "id": "guest",
                "name": "Guest",
                "description": "Limited read-only access",
                "permission_ids": [
                    "dharma.query.read", "chat.session.read"
                ]
            }
        ]
    
    async def create_permission(self, permission: Permission) -> bool:
        """Create a new permission"""
        
        with self.Session() as session:
            # Check if permission already exists
            existing = session.query(PermissionModel).filter_by(id=permission.id).first()
            if existing:
                return False
            
            perm_model = PermissionModel(
                id=permission.id,
                name=permission.name,
                description=permission.description,
                resource_type=permission.resource_type.value,
                action=permission.action.value,
                scope=permission.scope.value,
                conditions=permission.conditions
            )
            
            session.add(perm_model)
            session.commit()
        
        logger.info(f"🔐 Created permission: {permission.name}")
        return True
    
    async def create_role(self, role: Role) -> bool:
        """Create a new role"""
        
        with self.Session() as session:
            # Check if role already exists
            existing = session.query(RoleModel).filter_by(id=role.id).first()
            if existing:
                return False
            
            role_model = RoleModel(
                id=role.id,
                name=role.name,
                description=role.description,
                is_system_role=role.is_system_role,
                parent_role_id=role.parent_role_id,
                tenant_id=role.tenant_id
            )
            
            # Add permissions
            for permission in role.permissions:
                perm_model = session.query(PermissionModel).filter_by(id=permission.id).first()
                if perm_model:
                    role_model.permissions.append(perm_model)
            
            session.add(role_model)
            session.commit()
        
        logger.info(f"🔐 Created role: {role.name}")
        return True
    
    async def assign_role_to_user(self, user_id: str, role_id: str, tenant_id: Optional[str] = None) -> bool:
        """Assign a role to a user"""
        
        with self.Session() as session:
            # Get or create user
            user = session.query(UserModel).filter_by(id=user_id).first()
            if not user:
                user = UserModel(id=user_id, email=f"user_{user_id}@example.com", tenant_id=tenant_id)
                session.add(user)
            
            # Get role
            role = session.query(RoleModel).filter_by(id=role_id).first()
            if not role:
                return False
            
            # Check if already assigned
            if role in user.roles:
                return True
            
            user.roles.append(role)
            session.commit()
        
        # Clear user permissions cache
        cache_key = f"user_permissions:{user_id}"
        await self.redis.delete(cache_key)
        
        logger.info(f"🔐 Assigned role {role_id} to user {user_id}")
        return True
    
    async def get_user_permissions(self, user_id: str, tenant_id: Optional[str] = None) -> List[Permission]:
        """Get all permissions for a user"""
        
        # Check cache first
        cache_key = f"user_permissions:{user_id}:{tenant_id or 'global'}"
        cached_perms = await self.redis.get(cache_key)
        
        if cached_perms:
            perms_data = json.loads(cached_perms)
            return [Permission(**perm) for perm in perms_data]
        
        # Get from database
        permissions = []
        
        with self.Session() as session:
            user = session.query(UserModel).filter_by(id=user_id).first()
            
            if user:
                for role in user.roles:
                    # Filter by tenant if specified
                    if tenant_id and role.tenant_id and role.tenant_id != tenant_id:
                        continue
                    
                    for perm_model in role.permissions:
                        permission = Permission(
                            id=perm_model.id,
                            name=perm_model.name,
                            description=perm_model.description,
                            resource_type=ResourceType(perm_model.resource_type),
                            action=ActionType(perm_model.action),
                            scope=PermissionScope(perm_model.scope),
                            conditions=perm_model.conditions
                        )
                        permissions.append(permission)
        
        # Remove duplicates
        unique_permissions = []
        seen_ids = set()
        for perm in permissions:
            if perm.id not in seen_ids:
                unique_permissions.append(perm)
                seen_ids.add(perm.id)
        
        # Cache results
        perms_data = [asdict(perm) for perm in unique_permissions]
        await self.redis.setex(cache_key, self.cache_ttl, json.dumps(perms_data))
        
        return unique_permissions
    
    async def check_permission(self, 
                             user_id: str, 
                             resource_type: ResourceType, 
                             action: ActionType,
                             resource_id: Optional[str] = None,
                             tenant_id: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has permission for specific action"""
        
        # Get user permissions
        permissions = await self.get_user_permissions(user_id, tenant_id)
        
        # Check for matching permission
        for permission in permissions:
            if (permission.resource_type == resource_type and 
                permission.action == action):
                
                # Check scope-specific conditions
                if await self._check_permission_scope(permission, user_id, resource_id, tenant_id, context):
                    await self._log_access(user_id, resource_type, action, resource_id, "allowed", tenant_id, context)
                    return True
        
        # Check for admin permissions
        for permission in permissions:
            if (permission.resource_type == resource_type and 
                permission.action == ActionType.ADMIN):
                
                if await self._check_permission_scope(permission, user_id, resource_id, tenant_id, context):
                    await self._log_access(user_id, resource_type, action, resource_id, "allowed", tenant_id, context)
                    return True
        
        # Permission denied
        await self._log_access(user_id, resource_type, action, resource_id, "denied", tenant_id, context)
        return False
    
    async def _check_permission_scope(self, 
                                    permission: Permission,
                                    user_id: str,
                                    resource_id: Optional[str],
                                    tenant_id: Optional[str],
                                    context: Optional[Dict[str, Any]]) -> bool:
        """Check if permission scope allows access"""
        
        if permission.scope == PermissionScope.GLOBAL:
            return True
        
        if permission.scope == PermissionScope.TENANT:
            # User must be in the same tenant
            if tenant_id:
                with self.Session() as session:
                    user = session.query(UserModel).filter_by(id=user_id).first()
                    return user and user.tenant_id == tenant_id
            return False
        
        if permission.scope == PermissionScope.USER:
            # User can only access their own resources
            if context and context.get("resource_owner_id"):
                return context["resource_owner_id"] == user_id
            return True  # Default allow for user scope
        
        if permission.scope == PermissionScope.CUSTOM:
            # Evaluate custom conditions
            return await self._evaluate_custom_conditions(permission.conditions, user_id, resource_id, tenant_id, context)
        
        return False
    
    async def _evaluate_custom_conditions(self,
                                        conditions: Dict[str, Any],
                                        user_id: str,
                                        resource_id: Optional[str],
                                        tenant_id: Optional[str],
                                        context: Optional[Dict[str, Any]]) -> bool:
        """Evaluate custom permission conditions"""
        
        if not conditions:
            return True
        
        # Example custom condition evaluations
        if "time_based" in conditions:
            # Check time-based restrictions
            time_restriction = conditions["time_based"]
            current_hour = datetime.now().hour
            
            if "allowed_hours" in time_restriction:
                return current_hour in time_restriction["allowed_hours"]
        
        if "ip_restriction" in conditions:
            # Check IP-based restrictions
            if context and "client_ip" in context:
                allowed_ips = conditions["ip_restriction"].get("allowed_ips", [])
                return context["client_ip"] in allowed_ips
        
        if "resource_attribute" in conditions:
            # Check resource-specific attributes
            attr_conditions = conditions["resource_attribute"]
            if context:
                for attr, expected_value in attr_conditions.items():
                    if context.get(attr) != expected_value:
                        return False
        
        return True
    
    async def _log_access(self,
                        user_id: str,
                        resource_type: ResourceType,
                        action: ActionType,
                        resource_id: Optional[str],
                        result: str,
                        tenant_id: Optional[str],
                        context: Optional[Dict[str, Any]]):
        """Log access attempt for security auditing"""
        
        log_id = str(uuid.uuid4())
        
        with self.Session() as session:
            access_log = AccessLogModel(
                id=log_id,
                user_id=user_id,
                tenant_id=tenant_id,
                resource_type=resource_type.value,
                action=action.value,
                resource_id=resource_id,
                result=result,
                ip_address=context.get("client_ip") if context else None,
                user_agent=context.get("user_agent") if context else None,
                request_metadata=context or {}
            )
            
            session.add(access_log)
            session.commit()
        
        # Also log to Redis for real-time monitoring
        log_key = f"access_logs:{datetime.now().strftime('%Y-%m-%d')}"
        log_data = {
            "id": log_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "resource_type": resource_type.value,
            "action": action.value,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis.lpush(log_key, json.dumps(log_data))
        await self.redis.expire(log_key, 86400 * 30)  # 30 days
    
    async def get_access_logs(self, 
                            user_id: Optional[str] = None,
                            tenant_id: Optional[str] = None,
                            days: int = 7,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get access logs for auditing"""
        
        with self.Session() as session:
            query = session.query(AccessLogModel)
            
            if user_id:
                query = query.filter(AccessLogModel.user_id == user_id)
            
            if tenant_id:
                query = query.filter(AccessLogModel.tenant_id == tenant_id)
            
            start_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(AccessLogModel.timestamp >= start_date)
            
            query = query.order_by(AccessLogModel.timestamp.desc())
            query = query.limit(limit)
            
            logs = query.all()
            
            return [
                {
                    "id": log.id,
                    "user_id": log.user_id,
                    "tenant_id": log.tenant_id,
                    "resource_type": log.resource_type,
                    "action": log.action,
                    "resource_id": log.resource_id,
                    "result": log.result,
                    "timestamp": log.timestamp.isoformat(),
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent,
                    "metadata": log.metadata
                }
                for log in logs
            ]


# FastAPI dependencies
async def require_permission(resource_type: ResourceType, action: ActionType):
    """FastAPI dependency that requires specific permission"""
    def permission_checker(request, rbac_manager: AdvancedRBACManager, current_user: dict):
        async def check():
            user_id = current_user.get("id")
            tenant_id = getattr(request.state, "tenant_id", None)
            
            context = {
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent")
            }
            
            has_permission = await rbac_manager.check_permission(
                user_id=user_id,
                resource_type=resource_type,
                action=action,
                tenant_id=tenant_id,
                context=context
            )
            
            if not has_permission:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403, 
                    detail=f"Permission denied: {resource_type.value}:{action.value}"
                )
            
            return True
        
        return check()
    
    return permission_checker


# Export main components
__all__ = [
    "AdvancedRBACManager",
    "Permission",
    "Role",
    "ActionType",
    "ResourceType",
    "PermissionScope",
    "PermissionModel",
    "RoleModel",
    "UserModel",
    "AccessLogModel",
    "require_permission"
]
