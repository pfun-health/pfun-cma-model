"""JWT-secured authentication routes using OrcID SSO provider."""

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Security, status
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyCookie, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from jose import JWTError, jwt

# SSO imports
import fastapi_sso.sso
import pfun_cma_model.sso.providers as pfun_providers
from fastapi_sso.sso.base import OpenID

fastapi_sso.sso.__dict__.update(pfun_providers.__dict__)
from pfun_cma_model.sso.providers.orcid import OrcidSSO

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_EXPIRATION_MINUTES", "1440")
)  # 24 hours default

ORCID_CLIENT_ID = os.getenv("ORCID_CLIENT_ID", "")
ORCID_CLIENT_SECRET = os.getenv("ORCID_CLIENT_SECRET", "")
ORCID_REDIRECT_URI = os.getenv(
    "ORCID_REDIRECT_URI", "http://localhost:8001/auth/orcid/callback"
)

router = APIRouter(prefix="/auth", tags=["auth"])

# ==================== Response Models ====================


class Token(BaseModel):
    """JWT token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """JWT token data model."""

    sub: str  # subject (user ID)
    exp: datetime
    iat: datetime
    provider: str


class UserInfo(BaseModel):
    """User information model."""

    id: str
    first_name: Optional[str] = None
    display_name: Optional[str] = None
    picture: Optional[str] = None
    provider: str


# ==================== JWT Token Functions ====================


def create_access_token(
    user_id: str, provider: str, expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token.

    Args:
        user_id: The user identifier from the SSO provider
        provider: The SSO provider name
        expires_delta: Optional token expiration delta (defaults to configured expiration)

    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    to_encode = {
        "sub": user_id,
        "provider": provider,
        "iat": now,
        "exp": expire,
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token.

    Args:
        token: JWT token string to verify

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        provider: str = payload.get("provider")
        iat = payload.get("iat")
        exp = payload.get("exp")

        if user_id is None or provider is None:
            raise credentials_exception

        token_data = TokenData(
            sub=user_id,
            provider=provider,
            iat=(
                datetime.fromtimestamp(iat, tz=timezone.utc)
                if iat
                else datetime.now(timezone.utc)
            ),
            exp=(
                datetime.fromtimestamp(exp, tz=timezone.utc)
                if exp
                else datetime.now(timezone.utc)
            ),
        )
    except JWTError:
        raise credentials_exception

    return token_data


# ==================== Dependency Functions ====================


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(HTTPBearer())],
) -> TokenData:
    """Get current authenticated user from Bearer token.

    Args:
        credentials: HTTP Bearer credentials from the request

    Returns:
        Verified token data

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    return await verify_token(token)


async def get_optional_user(request: Request) -> Optional[TokenData]:
    """Get optional current user (doesn't raise if not authenticated).

    Args:
        request: HTTP request object

    Returns:
        Verified token data if present, None otherwise
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            return None
        return await verify_token(token)
    except (ValueError, HTTPException):
        return None


# ==================== SSO Routes ====================


async def get_orcid_provider() -> OrcidSSO:
    """Get OrcID SSO provider instance.

    Returns:
        Configured OrcidSSO provider

    Raises:
        RuntimeError: If OrcID credentials are not configured
    """
    if not ORCID_CLIENT_ID or not ORCID_CLIENT_SECRET:
        raise RuntimeError(
            "OrcID credentials not configured. "
            "Set ORCID_CLIENT_ID and ORCID_CLIENT_SECRET environment variables."
        )

    return OrcidSSO(
        client_id=ORCID_CLIENT_ID,
        client_secret=ORCID_CLIENT_SECRET,
        redirect_uri=ORCID_REDIRECT_URI,
    )


@router.get("/orcid/login")
async def orcid_login(request: Request) -> RedirectResponse:
    """Initiate OrcID login flow.

    Redirects to OrcID authorization endpoint.
    """
    try:
        sso_provider = await get_orcid_provider()
        authorization_url = await sso_provider.get_login_url()
        return RedirectResponse(url=authorization_url)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.get("/orcid/callback")
async def orcid_callback(code: str, request: Request) -> Token:
    """Handle OrcID callback after user authorization.

    Args:
        code: Authorization code from OrcID
        request: HTTP request object

    Returns:
        JWT access token

    Raises:
        HTTPException: If callback processing fails
    """
    try:
        sso_provider = await get_orcid_provider()
        user_info: OpenID = await sso_provider.verify_and_process_callback(
            request, code
        )

        access_token = create_access_token(
            user_id=user_info.id,
            provider=user_info.provider,
        )

        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(expires_delta.total_seconds()),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"OrcID authentication failed: {str(e)}",
        )


# ==================== Token Management Routes ====================


@router.post("/token/refresh")
async def refresh_token(
    current_user: Annotated[TokenData, Depends(get_current_user)],
) -> Token:
    """Refresh an existing JWT token.

    Args:
        current_user: Current authenticated user from token

    Returns:
        New JWT access token
    """
    access_token = create_access_token(
        user_id=current_user.sub,
        provider=current_user.provider,
    )

    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(expires_delta.total_seconds()),
    )


@router.post("/token/verify")
async def verify_access_token(
    current_user: Annotated[TokenData, Depends(get_current_user)],
) -> dict:
    """Verify the validity of current JWT token.

    Args:
        current_user: Current authenticated user

    Returns:
        Verification result with user info
    """
    return {
        "valid": True,
        "user_id": current_user.sub,
        "provider": current_user.provider,
        "issued_at": current_user.iat.isoformat(),
        "expires_at": current_user.exp.isoformat(),
    }


# ==================== User Info Routes ====================


@router.get("/user/me", response_model=UserInfo)
async def get_current_user_info(
    current_user: Annotated[TokenData, Depends(get_current_user)],
) -> UserInfo:
    """Get current authenticated user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User information from token
    """
    return UserInfo(
        id=current_user.sub,
        provider=current_user.provider,
    )


@router.post("/logout")
async def logout(current_user: Annotated[TokenData, Depends(get_current_user)]) -> dict:
    """Logout current user (invalidate token).

    Note: Token invalidation in this basic implementation is client-side
    (delete the token). For robust logout, implement token blacklisting
    using Redis or database.

    Args:
        current_user: Current authenticated user

    Returns:
        Logout confirmation
    """
    return {
        "message": "Successfully logged out",
        "user_id": current_user.sub,
    }


# ==================== Health Check Routes ====================


@router.get("/health")
async def auth_health_check() -> dict:
    """Check authentication service health.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "auth",
        "jwt_algorithm": ALGORITHM,
        "providers": ["orcid"],
    }


@router.get("/health/verify")
async def auth_health_verify_token(
    token: Optional[str] = None,
) -> dict:
    """Verify token validity without requiring Bearer auth.

    Args:
        token: Optional JWT token to verify

    Returns:
        Token verification status
    """
    if not token:
        return {
            "status": "no_token",
            "message": "No token provided",
        }

    try:
        token_data = await verify_token(token)
        return {
            "status": "valid",
            "user_id": token_data.sub,
            "provider": token_data.provider,
            "expires_at": token_data.exp.isoformat(),
        }
    except HTTPException:
        return {
            "status": "invalid",
            "message": "Token is invalid or expired",
        }
