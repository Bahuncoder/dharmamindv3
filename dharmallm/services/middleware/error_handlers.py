"""
Secure Error Handling Middleware
Prevents information disclosure through error messages
"""

import logging
import traceback
import uuid
from typing import Union
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorResponse:
    """Standardized error response"""
    
    @staticmethod
    def create(
        error_code: str,
        message: str,
        status_code: int,
        request_id: str = None,
        details: dict = None
    ) -> dict:
        """Create a standardized error response"""
        response = {
            "error": {
                "code": error_code,
                "message": message,
                "request_id": request_id or str(uuid.uuid4())
            }
        }
        
        # Only include details in development mode
        if details and logger.level == logging.DEBUG:
            response["error"]["details"] = details
        
        return response


async def global_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Global exception handler - prevents information disclosure
    
    - Logs full error internally
    - Returns generic message to user
    - Includes request ID for tracking
    """
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())
    
    # Log full error details internally
    logger.error(
        f"Unhandled exception (Request ID: {request_id})",
        exc_info=True,
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    # Return generic error to user (hide internal details)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse.create(
            error_code="INTERNAL_ERROR",
            message="An internal error occurred. Please try again later.",
            status_code=500,
            request_id=request_id
        )
    )


async def http_exception_handler(
    request: Request,
    exc: Union[HTTPException, StarletteHTTPException]
) -> JSONResponse:
    """
    Handler for HTTP exceptions
    
    Returns error with status code but sanitized details
    """
    request_id = str(uuid.uuid4())
    
    # Log the error
    logger.warning(
        f"HTTP Exception (Request ID: {request_id}): "
        f"{exc.status_code} - {exc.detail}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code
        }
    )
    
    # Map status codes to error codes
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE"
    }
    
    error_code = error_code_map.get(exc.status_code, "ERROR")
    
    # Use exception detail if safe, otherwise use generic message
    safe_messages = {
        400: "Invalid request",
        401: "Authentication required",
        403: "Access denied",
        404: "Resource not found",
        405: "Method not allowed",
        409: "Resource conflict",
        422: "Invalid input data",
        429: "Too many requests",
        500: "Internal server error",
        503: "Service temporarily unavailable"
    }
    
    # Use provided detail for client errors, generic for server errors
    if 400 <= exc.status_code < 500:
        message = str(exc.detail)
    else:
        message = safe_messages.get(exc.status_code, "An error occurred")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.create(
            error_code=error_code,
            message=message,
            status_code=exc.status_code,
            request_id=request_id
        ),
        headers=getattr(exc, "headers", None)
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handler for validation errors
    
    Sanitizes validation errors to prevent information disclosure
    """
    request_id = str(uuid.uuid4())
    
    # Log validation errors
    logger.warning(
        f"Validation Error (Request ID: {request_id})",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "errors": exc.errors()
        }
    )
    
    # Sanitize error messages
    sanitized_errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        sanitized_errors.append({
            "field": field,
            "message": "Invalid value provided"
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse.create(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            status_code=422,
            request_id=request_id,
            details={"errors": sanitized_errors}
        )
    )


async def custom_404_handler(request: Request, exc: Exception) -> JSONResponse:
    """Custom 404 handler"""
    request_id = str(uuid.uuid4())
    
    logger.info(
        f"404 Not Found (Request ID: {request_id}): {request.url.path}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=404,
        content=ErrorResponse.create(
            error_code="NOT_FOUND",
            message="The requested resource was not found",
            status_code=404,
            request_id=request_id
        )
    )


def register_error_handlers(app):
    """
    Register all error handlers with the FastAPI application
    
    Usage:
        from services.middleware.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    # Global exception handler
    app.add_exception_handler(Exception, global_exception_handler)
    
    # HTTP exception handler
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Validation error handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # 404 handler
    app.add_exception_handler(404, custom_404_handler)
    
    logger.info("✓ Error handlers registered")


# Custom exception classes

class DharmaMindException(Exception):
    """Base exception for DharmaMind application"""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(DharmaMindException):
    """Authentication failed"""
    def __init__(self, message: str = "Authentication failed", details: dict = None):
        super().__init__(message, 401, details)


class AuthorizationError(DharmaMindException):
    """Authorization failed"""
    def __init__(self, message: str = "Access denied", details: dict = None):
        super().__init__(message, 403, details)


class ValidationError(DharmaMindException):
    """Validation failed"""
    def __init__(self, message: str = "Validation failed", details: dict = None):
        super().__init__(message, 422, details)


class ResourceNotFoundError(DharmaMindException):
    """Resource not found"""
    def __init__(self, message: str = "Resource not found", details: dict = None):
        super().__init__(message, 404, details)


class RateLimitError(DharmaMindException):
    """Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", details: dict = None):
        super().__init__(message, 429, details)


class DatabaseError(DharmaMindException):
    """Database operation failed"""
    def __init__(self, message: str = "Database error", details: dict = None):
        super().__init__(message, 500, details)


# Example usage:
"""
from fastapi import FastAPI
from services.middleware.error_handlers import register_error_handlers

app = FastAPI()

# Register error handlers
register_error_handlers(app)

# In your route:
from services.middleware.error_handlers import AuthenticationError

@app.get("/protected")
async def protected_route():
    if not is_authenticated():
        raise AuthenticationError("Please login to access this resource")
    return {"message": "Success"}
"""
