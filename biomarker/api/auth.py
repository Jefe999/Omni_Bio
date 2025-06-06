"""
Authentication middleware for OmniBio API
Handles API key authentication and user verification.
"""

from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyHeader
import hashlib

from biomarker.models.db_manager import get_database_manager
from biomarker.models.database import User


# API Key authentication schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Custom authentication error."""
    pass


async def verify_api_key(api_key: str) -> Optional[User]:
    """Verify API key and return user if valid."""
    if not api_key:
        return None
    
    try:
        db_manager = get_database_manager()
        user = await db_manager.get_user_by_api_key(api_key)
        
        if user and user.is_active:
            # Update last login
            await db_manager.update_user_last_login(str(user.id))
            return user
        
        return None
    except Exception:
        return None


async def get_current_user(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth)
) -> Optional[User]:
    """Get current authenticated user from API key or bearer token."""
    
    # Try API key first
    if api_key:
        user = await verify_api_key(api_key)
        if user:
            return user
    
    # Try bearer token as API key
    if bearer_token and bearer_token.credentials:
        user = await verify_api_key(bearer_token.credentials)
        if user:
            return user
    
    return None


async def require_auth(current_user: User = Depends(get_current_user)) -> User:
    """Require authentication - raises 401 if not authenticated."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


async def require_write_permission(current_user: User = Depends(require_auth)) -> User:
    """Require write permissions."""
    if not current_user.permissions.get("write", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permission required"
        )
    return current_user


async def require_admin_permission(current_user: User = Depends(require_auth)) -> User:
    """Require admin permissions."""
    if not current_user.permissions.get("admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    return current_user


def generate_api_key(length: int = 32) -> str:
    """Generate a secure API key."""
    import secrets
    import string
    
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# Optional auth dependency - doesn't raise error if not authenticated
async def optional_auth(current_user: User = Depends(get_current_user)) -> Optional[User]:
    """Optional authentication - returns None if not authenticated."""
    return current_user 