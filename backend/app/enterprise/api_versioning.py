"""
📚 DharmaMind API Versioning & Compatibility System

Enterprise-grade API versioning with backward compatibility:

Core Features:
- Semantic versioning (v1, v2, etc.)
- Backward compatibility management
- Automatic API documentation versioning
- Deprecation warnings and timeline
- Client SDK auto-generation
- Version-specific middleware
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import re
from packaging import version
from pydantic import BaseModel, Field, validator
from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.routing import APIRoute
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.models import OpenAPI
import redis.asyncio as redis

# Versioning configuration
logger = logging.getLogger("dharmamind.versioning")


class VersionStatus(str, Enum):
    """API version status"""
    DEVELOPMENT = "development"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class BreakingChangeType(str, Enum):
    """Types of breaking changes"""
    FIELD_REMOVED = "field_removed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    ENDPOINT_REMOVED = "endpoint_removed"
    REQUIRED_FIELD_ADDED = "required_field_added"
    RESPONSE_FORMAT_CHANGED = "response_format_changed"
    AUTHENTICATION_CHANGED = "authentication_changed"


@dataclass
class APIVersion:
    """API version definition"""
    version: str
    status: VersionStatus
    release_date: datetime
    sunset_date: Optional[datetime] = None
    description: str = ""
    breaking_changes: List[str] = None
    compatible_versions: List[str] = None
    
    def __post_init__(self):
        if self.breaking_changes is None:
            self.breaking_changes = []
        if self.compatible_versions is None:
            self.compatible_versions = []
    
    @property
    def is_active(self) -> bool:
        """Check if version is still active"""
        if self.sunset_date:
            return datetime.utcnow() < self.sunset_date
        return self.status not in [VersionStatus.SUNSET]
    
    @property
    def is_deprecated(self) -> bool:
        """Check if version is deprecated"""
        return self.status == VersionStatus.DEPRECATED


@dataclass
class CompatibilityRule:
    """Compatibility rule for version migration"""
    from_version: str
    to_version: str
    transformer: Callable[[Dict[str, Any]], Dict[str, Any]]
    reverse_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class VersionedResponse(BaseModel):
    """Versioned API response wrapper"""
    api_version: str
    data: Any
    warnings: List[str] = Field(default_factory=list)
    deprecation_notice: Optional[str] = None
    migration_guide: Optional[str] = None


class VersionedRequest(BaseModel):
    """Versioned API request wrapper"""
    api_version: str
    data: Dict[str, Any]


class APIVersionManager:
    """
    Comprehensive API version management system
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.versions: Dict[str, APIVersion] = {}
        self.compatibility_rules: List[CompatibilityRule] = []
        self.routers: Dict[str, APIRouter] = {}
        
        # Initialize default versions
        self._initialize_default_versions()
        
        logger.info("📚 API Version Manager initialized")
    
    def _initialize_default_versions(self):
        """Initialize default API versions"""
        
        # Version 1.0 - Legacy
        v1 = APIVersion(
            version="1.0",
            status=VersionStatus.DEPRECATED,
            release_date=datetime(2024, 1, 1),
            sunset_date=datetime(2025, 12, 31),
            description="Legacy API with basic dharma functionality",
            breaking_changes=[
                "Simple response format",
                "Limited authentication"
            ]
        )
        
        # Version 2.0 - Current stable
        v2 = APIVersion(
            version="2.0",
            status=VersionStatus.STABLE,
            release_date=datetime(2024, 6, 1),
            description="Enhanced API with advanced features and security",
            compatible_versions=["1.0"]
        )
        
        # Version 2.1 - Latest with observability
        v21 = APIVersion(
            version="2.1",
            status=VersionStatus.STABLE,
            release_date=datetime(2025, 8, 1),
            description="Latest API with observability and enterprise features",
            compatible_versions=["2.0"]
        )
        
        # Version 3.0 - Future
        v3 = APIVersion(
            version="3.0",
            status=VersionStatus.DEVELOPMENT,
            release_date=datetime(2026, 1, 1),
            description="Next generation API with AI enhancements",
            breaking_changes=[
                "New authentication system",
                "Redesigned response format",
                "Advanced dharma features"
            ]
        )
        
        self.versions = {
            "1.0": v1,
            "2.0": v2,
            "2.1": v21,
            "3.0": v3
        }
        
        # Set up compatibility rules
        self._setup_compatibility_rules()
    
    def _setup_compatibility_rules(self):
        """Set up version compatibility transformation rules"""
        
        # v1.0 to v2.0 transformation
        def v1_to_v2_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
            """Transform v1.0 response to v2.0 format"""
            if isinstance(data, dict):
                # Wrap simple response in structured format
                return {
                    "success": True,
                    "data": data,
                    "metadata": {
                        "version": "2.0",
                        "transformed_from": "1.0"
                    }
                }
            return data
        
        def v2_to_v1_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
            """Transform v2.0 response back to v1.0 format"""
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data
        
        self.compatibility_rules.append(
            CompatibilityRule(
                from_version="1.0",
                to_version="2.0",
                transformer=v1_to_v2_transformer,
                reverse_transformer=v2_to_v1_transformer
            )
        )
        
        # v2.0 to v2.1 transformation (mostly compatible)
        def v2_to_v21_transformer(data: Dict[str, Any]) -> Dict[str, Any]:
            """Transform v2.0 response to v2.1 format"""
            if isinstance(data, dict):
                # Add observability metadata
                data.setdefault("metadata", {})
                data["metadata"]["observability"] = {
                    "trace_id": "auto-generated",
                    "version": "2.1"
                }
            return data
        
        self.compatibility_rules.append(
            CompatibilityRule(
                from_version="2.0",
                to_version="2.1",
                transformer=v2_to_v21_transformer
            )
        )
    
    def register_version(self, api_version: APIVersion) -> bool:
        """Register a new API version"""
        
        if api_version.version in self.versions:
            logger.warning(f"Version {api_version.version} already exists")
            return False
        
        self.versions[api_version.version] = api_version
        logger.info(f"📚 Registered API version: {api_version.version}")
        return True
    
    def get_version(self, version_str: str) -> Optional[APIVersion]:
        """Get API version by version string"""
        return self.versions.get(version_str)
    
    def get_active_versions(self) -> List[APIVersion]:
        """Get all active API versions"""
        return [v for v in self.versions.values() if v.is_active]
    
    def get_latest_version(self) -> APIVersion:
        """Get the latest stable version"""
        stable_versions = [
            v for v in self.versions.values() 
            if v.status == VersionStatus.STABLE and v.is_active
        ]
        
        if not stable_versions:
            return list(self.versions.values())[-1]
        
        # Sort by version number
        stable_versions.sort(key=lambda x: version.parse(x.version), reverse=True)
        return stable_versions[0]
    
    def parse_version_from_request(self, request: Request) -> str:
        """Parse API version from request"""
        
        # Method 1: Header
        version_header = request.headers.get("API-Version") or request.headers.get("X-API-Version")
        if version_header:
            return version_header
        
        # Method 2: Query parameter
        version_query = request.query_params.get("version") or request.query_params.get("api_version")
        if version_query:
            return version_query
        
        # Method 3: Path prefix (e.g., /api/v2.1/endpoint)
        path = request.url.path
        version_match = re.search(r'/api/v(\d+\.\d+)/', path)
        if version_match:
            return version_match.group(1)
        
        # Method 4: Accept header
        accept_header = request.headers.get("Accept", "")
        version_match = re.search(r'application/vnd\.dharmamind\.v(\d+\.\d+)', accept_header)
        if version_match:
            return version_match.group(1)
        
        # Default to latest stable version
        return self.get_latest_version().version
    
    def transform_response(self, data: Any, from_version: str, to_version: str) -> Any:
        """Transform response data between versions"""
        
        if from_version == to_version:
            return data
        
        # Find transformation rule
        for rule in self.compatibility_rules:
            if rule.from_version == from_version and rule.to_version == to_version:
                return rule.transformer(data)
        
        # Try reverse transformation
        for rule in self.compatibility_rules:
            if (rule.to_version == from_version and 
                rule.from_version == to_version and 
                rule.reverse_transformer):
                return rule.reverse_transformer(data)
        
        # No transformation rule found
        logger.warning(f"No transformation rule from {from_version} to {to_version}")
        return data
    
    def create_versioned_router(self, version_str: str, prefix: str = "") -> APIRouter:
        """Create a versioned router for specific API version"""
        
        version_info = self.get_version(version_str)
        if not version_info:
            raise ValueError(f"Unknown version: {version_str}")
        
        router = APIRouter(
            prefix=f"/api/v{version_str}" if not prefix else prefix,
            tags=[f"API v{version_str}"],
            responses={
                200: {"description": "Success"},
                400: {"description": "Bad Request"},
                401: {"description": "Unauthorized"},
                403: {"description": "Forbidden"},
                404: {"description": "Not Found"},
                429: {"description": "Rate Limit Exceeded"},
                500: {"description": "Internal Server Error"}
            }
        )
        
        self.routers[version_str] = router
        return router
    
    def create_version_middleware(self):
        """Create middleware for version handling"""
        
        async def version_middleware(request: Request, call_next):
            """Middleware to handle API versioning"""
            
            # Parse requested version
            requested_version = self.parse_version_from_request(request)
            
            # Get version info
            version_info = self.get_version(requested_version)
            if not version_info:
                # Default to latest version
                version_info = self.get_latest_version()
                requested_version = version_info.version
            
            # Check if version is active
            if not version_info.is_active:
                return Response(
                    content=json.dumps({
                        "error": "API version not available",
                        "version": requested_version,
                        "status": version_info.status.value,
                        "sunset_date": version_info.sunset_date.isoformat() if version_info.sunset_date else None
                    }),
                    status_code=410,  # Gone
                    media_type="application/json"
                )
            
            # Set version context
            request.state.api_version = requested_version
            request.state.version_info = version_info
            
            # Process request
            response = await call_next(request)
            
            # Add version headers
            response.headers["API-Version"] = requested_version
            response.headers["API-Version-Status"] = version_info.status.value
            
            # Add deprecation warning if needed
            if version_info.is_deprecated:
                response.headers["Deprecation"] = "true"
                if version_info.sunset_date:
                    response.headers["Sunset"] = version_info.sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
            
            return response
        
        return version_middleware
    
    async def generate_openapi_spec(self, version_str: str, app_title: str = "DharmaMind API") -> Dict[str, Any]:
        """Generate OpenAPI specification for specific version"""
        
        version_info = self.get_version(version_str)
        if not version_info:
            raise ValueError(f"Unknown version: {version_str}")
        
        # Get router for version
        router = self.routers.get(version_str)
        if not router:
            raise ValueError(f"No router found for version: {version_str}")
        
        # Generate OpenAPI spec
        openapi_schema = get_openapi(
            title=f"{app_title} v{version_str}",
            version=version_str,
            description=version_info.description,
            routes=router.routes
        )
        
        # Add version-specific information
        openapi_schema["info"]["x-api-version"] = version_str
        openapi_schema["info"]["x-version-status"] = version_info.status.value
        openapi_schema["info"]["x-release-date"] = version_info.release_date.isoformat()
        
        if version_info.sunset_date:
            openapi_schema["info"]["x-sunset-date"] = version_info.sunset_date.isoformat()
        
        if version_info.breaking_changes:
            openapi_schema["info"]["x-breaking-changes"] = version_info.breaking_changes
        
        if version_info.compatible_versions:
            openapi_schema["info"]["x-compatible-versions"] = version_info.compatible_versions
        
        # Store in Redis for caching
        cache_key = f"openapi_spec:{version_str}"
        await self.redis.setex(
            cache_key,
            3600,  # 1 hour cache
            json.dumps(openapi_schema)
        )
        
        return openapi_schema
    
    async def get_version_changelog(self, from_version: str, to_version: str) -> Dict[str, Any]:
        """Get changelog between versions"""
        
        from_info = self.get_version(from_version)
        to_info = self.get_version(to_version)
        
        if not from_info or not to_info:
            return {"error": "Version not found"}
        
        # Parse version numbers for comparison
        from_ver = version.parse(from_version)
        to_ver = version.parse(to_version)
        
        if from_ver >= to_ver:
            return {"error": "Invalid version range"}
        
        # Collect changes between versions
        changelog = {
            "from_version": from_version,
            "to_version": to_version,
            "breaking_changes": [],
            "new_features": [],
            "deprecations": [],
            "migration_notes": []
        }
        
        # Get all versions between from and to
        intermediate_versions = [
            v for v in self.versions.values()
            if from_ver < version.parse(v.version) <= to_ver
        ]
        
        # Sort by version
        intermediate_versions.sort(key=lambda x: version.parse(x.version))
        
        for v in intermediate_versions:
            changelog["breaking_changes"].extend(v.breaking_changes)
            
            if v.is_deprecated:
                changelog["deprecations"].append({
                    "version": v.version,
                    "sunset_date": v.sunset_date.isoformat() if v.sunset_date else None
                })
        
        return changelog
    
    def get_version_compatibility_matrix(self) -> Dict[str, Any]:
        """Get compatibility matrix for all versions"""
        
        matrix = {}
        
        for version_str, version_info in self.versions.items():
            matrix[version_str] = {
                "status": version_info.status.value,
                "is_active": version_info.is_active,
                "compatible_with": version_info.compatible_versions,
                "breaking_changes_count": len(version_info.breaking_changes)
            }
        
        return {
            "compatibility_matrix": matrix,
            "latest_stable": self.get_latest_version().version,
            "active_versions": [v.version for v in self.get_active_versions()]
        }


# FastAPI dependencies
async def get_api_version(request: Request) -> str:
    """FastAPI dependency to get current API version"""
    return getattr(request.state, "api_version", "2.1")


async def get_version_info(request: Request) -> APIVersion:
    """FastAPI dependency to get current version info"""
    return getattr(request.state, "version_info", None)


def require_version(min_version: str):
    """FastAPI dependency that requires minimum API version"""
    def version_checker(current_version: str = Depends(get_api_version)):
        current_ver = version.parse(current_version)
        min_ver = version.parse(min_version)
        
        if current_ver < min_ver:
            raise HTTPException(
                status_code=400,
                detail=f"API version {min_version} or higher required. Current: {current_version}"
            )
        
        return current_version
    
    return version_checker


# Create version-aware response wrapper
def create_versioned_response(data: Any, api_version: str, warnings: List[str] = None) -> VersionedResponse:
    """Create a versioned response with appropriate metadata"""
    
    return VersionedResponse(
        api_version=api_version,
        data=data,
        warnings=warnings or []
    )


# Export main components
__all__ = [
    "APIVersionManager",
    "APIVersion",
    "VersionStatus",
    "BreakingChangeType",
    "VersionedResponse",
    "VersionedRequest",
    "CompatibilityRule",
    "get_api_version",
    "get_version_info",
    "require_version",
    "create_versioned_response"
]
