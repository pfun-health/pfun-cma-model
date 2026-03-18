"""SQLAdmin SSO Auth Backend using fastapi-sso.

This example demonstrates how to integrate fastapi-sso with sqladmin for OAuth-based
authentication. It shows the recommended patterns for:

1. Setting up SSO providers (Google, Microsoft, GitHub, etc.)
2. Creating a custom AuthenticationBackend that supports both SSO and form login
3. Handling the OAuth callback flow with proper state management
4. Managing user sessions and tokens
5. Integrating with your existing User model

Prerequisites:
- Install required packages:
    uv add fastapi-sso sqladmin[full] passlib[bcrypt] python-jose
- Create OAuth credentials for your provider:
    - Google: https://console.cloud.google.com/apis/credentials
    - Microsoft: https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade
    - GitHub: https://github.com/settings/developers

Usage:
    Set environment variables:
        GOOGLE_CLIENT_ID=your-client-id
        GOOGLE_CLIENT_SECRET=your-client-secret
        SECRET_KEY=your-secret-key-min-32-chars
        ADMIN_BASE_URL=http://localhost:8000/admin

    Then run:
        uvicorn examples.code_samples.sqladmin_sso_auth_backend:app --reload
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional, Union

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi_sso.sso.base import OpenID
from fastapi_sso.sso.google import GoogleSSO
from fastapi_sso.sso.microsoft import MicrosoftSSO
from fastapi_sso.sso.github import GithubSSO
from passlib.context import CryptContext
from sqlalchemy import Column, Integer, String, Boolean, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.middleware.sessions import SessionMiddleware

# =============================================================================
# Configuration
# =============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
ADMIN_BASE_URL = os.getenv("ADMIN_BASE_URL", "http://localhost:8000")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =============================================================================
# Database Setup
# =============================================================================

Base = declarative_base()
engine = create_async_engine(
    "sqlite+aiosqlite:///./sso_admin_example.db",
    connect_args={"check_same_thread": False},
)
async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class User(Base):
    """Example User model for demonstration."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    provider = Column(String, default="local")
    provider_id = Column(String, nullable=True)
    hashed_password = Column(String, nullable=True)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)


async def init_db():
    """Initialize the database and create tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(title="SQLAdmin SSO Example")

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=1200,  # 20 minutes
)

# =============================================================================
# SSO Provider Setup
# =============================================================================

SSO_PROVIDERS: dict[str, Any] = {}


def get_sso_provider(provider_name: str) -> Optional[Any]:
    """Get or create an SSO provider instance.

    In production, you would typically use a dependency injection pattern
    or store the provider instances in application state.
    """
    if provider_name in SSO_PROVIDERS:
        return SSO_PROVIDERS[provider_name]

    redirect_uri = f"{ADMIN_BASE_URL}/admin/auth/{provider_name}/callback"

    if provider_name == "google":
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            return None
        SSO_PROVIDERS[provider_name] = GoogleSSO(
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            redirect_uri=redirect_uri,
        )
    elif provider_name == "microsoft":
        client_id = os.getenv("MICROSOFT_CLIENT_ID", "")
        client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            return None
        SSO_PROVIDERS[provider_name] = MicrosoftSSO(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )
    elif provider_name == "github":
        client_id = os.getenv("GITHUB_CLIENT_ID", "")
        client_secret = os.getenv("GITHUB_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            return None
        SSO_PROVIDERS[provider_name] = GithubSSO(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )

    return SSO_PROVIDERS.get(provider_name)


# =============================================================================
# User Management Utilities
# =============================================================================


async def get_or_create_sso_user(openid: OpenID, provider: str) -> Optional[User]:
    """Get an existing user or create a new one from SSO data.

    Args:
        openid: The OpenID object containing user information from SSO
        provider: The name of the SSO provider (e.g., "google", "microsoft")

    Returns:
        The User object if found/created, None if creation fails
    """
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == openid.email))
        user = result.scalars().first()

        if user is None:
            user = User(
                email=openid.email,
                name=openid.display_name or openid.first_name or "Unknown",
                provider=provider,
                provider_id=openid.id,
                is_active=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        else:
            user.last_login = datetime.now(timezone.utc)
            await session.commit()

        return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.

    Args:
        data: Dictionary containing claims to encode
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token.

    Args:
        token: The JWT token string to decode

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


# =============================================================================
# Custom Login Page Template
# =============================================================================

LOGIN_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Admin Login</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .login-container {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 400px;
        }
        h1 { margin: 0 0 1.5rem; color: #333; text-align: center; }
        .sso-buttons { margin-bottom: 1.5rem; }
        .sso-btn {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: opacity 0.2s;
        }
        .sso-btn:hover { opacity: 0.9; }
        .sso-btn.google { background: #4285f4; color: white; }
        .sso-btn.microsoft { background: #0078d4; color: white; }
        .sso-btn.github { background: #24292e; color: white; }
        .divider {
            text-align: center;
            margin: 1.5rem 0;
            color: #666;
            position: relative;
        }
        .divider::before, .divider::after {
            content: '';
            position: absolute;
            top: 50%;
            width: 45%;
            height: 1px;
            background: #ddd;
        }
        .divider::before { left: 0; }
        .divider::after { right: 0; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; margin-bottom: 0.5rem; color: #333; }
        input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 14px;
        }
        input:focus { outline: none; border-color: #667eea; }
        .submit-btn {
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }
        .submit-btn:hover { background: #5568d3; }
        .error { color: #dc3545; margin-bottom: 1rem; text-align: center; }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>Admin Login</h1>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <div class="sso-buttons">
            <a href="/admin/auth/google/login">
                <button type="button" class="sso-btn google">
                    Sign in with Google
                </button>
            </a>
            <a href="/admin/auth/microsoft/login">
                <button type="button" class="sso-btn microsoft">
                    Sign in with Microsoft
                </button>
            </a>
            <a href="/admin/auth/github/login">
                <button type="button" class="sso-btn github">
                    Sign in with GitHub
                </button>
            </a>
        </div>

        <div class="divider">or</div>

        <form method="post" action="/admin/login">
            <div class="form-group">
                <label for="username">Email or Username</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="submit-btn">Sign In</button>
        </form>
    </div>
</body>
</html>
"""


# =============================================================================
# SQLAdmin Authentication Backend
# =============================================================================


class SSOAuthBackend(AuthenticationBackend):
    """Custom authentication backend supporting both SSO and traditional login.

    This backend integrates with fastapi-sso to provide OAuth-based authentication
    alongside traditional username/password login.

    Key features:
    - Supports multiple SSO providers (Google, Microsoft, GitHub, etc.)
    - Falls back to traditional username/password authentication
    - Stores user info in session for authenticated requests
    - Uses JWT tokens for stateless authentication
    """

    def __init__(self, secret_key: str):
        super().__init__(secret_key=secret_key)
        self.secret_key = secret_key

    async def login(self, request: Request) -> bool:
        """Handle form-based login.

        This method is called when the user submits the login form.
        For SSO logins, the user is redirected to the provider before
        reaching this method.

        Args:
            request: The incoming request containing form data

        Returns:
            True if login successful, False otherwise
        """
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

        if not username or not password:
            return False

        async with async_session_maker() as session:
            result = await session.execute(
                select(User).where((User.email == username) | (User.name == username))
            )
            user = result.scalars().first()

            if not user or not user.hashed_password:
                return False

            if not pwd_context.verify(password, user.hashed_password):
                return False

            if not user.is_active:
                return False

            token_data = {
                "sub": str(user.id),
                "email": user.email,
                "name": user.name,
                "is_admin": user.is_admin,
            }
            access_token = create_access_token(token_data)
            request.session.update(
                {
                    "token": access_token,
                    "user_id": user.id,
                    "user_email": user.email,
                }
            )
            return True

    async def logout(self, request: Request) -> bool:
        """Clear the session on logout.

        Args:
            request: The incoming request

        Returns:
            Always returns True
        """
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> Union[bool, RedirectResponse]:
        """Authenticate incoming requests.

        This method checks for a valid session token. If no token is found,
        it returns False which will redirect to the login page.

        For SSO, the authenticate method is called after the callback
        completes and user data is stored in the session.

        Args:
            request: The incoming request

        Returns:
            True if authenticated, False to show login page,
            or RedirectResponse to redirect elsewhere
        """
        token = request.session.get("token")
        if not token:
            return False

        try:
            payload = decode_token(token)
            user_id = int(payload.get("sub"))

            async with async_session_maker() as session:
                result = await session.execute(select(User).where(User.id == user_id))
                user = result.scalars().first()

                if not user or not user.is_active:
                    return False

                return True
        except (HTTPException, ValueError):
            return False


# =============================================================================
# SSO Authentication Endpoints
# =============================================================================


@app.get("/admin/auth/{provider}/login")
async def sso_login(request: Request, provider: str) -> RedirectResponse:
    """Initiate SSO login flow.

    Redirects the user to the SSO provider's authorization page.

    Args:
        request: The incoming request (used to build redirect URI)
        provider: The SSO provider name (google, microsoft, github)

    Returns:
        RedirectResponse to the SSO provider's login page
    """
    sso = get_sso_provider(provider)
    if not sso:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider}' not configured or credentials missing",
        )

    async with sso:
        return await sso.get_login_redirect()


@app.get("/admin/auth/{provider}/callback")
async def sso_callback(request: Request, provider: str) -> RedirectResponse:
    """Handle SSO callback from the provider.

    Processes the OAuth callback, verifies the user, creates or retrieves
    the user from the database, and creates a session.

    Args:
        request: The callback request containing OAuth parameters
        provider: The SSO provider name

    Returns:
        RedirectResponse to the admin index page
    """
    sso = get_sso_provider(provider)
    if not sso:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider}' not configured",
        )

    try:
        async with sso:
            user_openid = await sso.verify_and_process(request)

        if not user_openid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to verify SSO response",
            )

        user = await get_or_create_sso_user(user_openid, provider)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create or retrieve user",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled",
            )

        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "name": user.name,
            "is_admin": user.is_admin,
            "pld": user_openid.model_dump(),
        }
        access_token = create_access_token(token_data)

        request.session.update(
            {
                "token": access_token,
                "user_id": user.id,
                "user_email": user.email,
            }
        )

        return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SSO authentication failed: {str(e)}",
        )


# =============================================================================
# Admin Setup
# =============================================================================

authentication_backend = SSOAuthBackend(secret_key=SECRET_KEY)
admin = Admin(
    app=app,
    engine=engine,
    authentication_backend=authentication_backend,
    title="SSO Admin",
)


class UserAdmin(ModelView, model=User):
    """Admin view for User model."""

    column_list = [
        User.id,
        User.email,
        User.name,
        User.provider,
        User.is_admin,
        User.is_active,
    ]
    column_searchable_list = [User.email, User.name]
    column_sortable_list = [User.id, User.email, User.name]
    form_columns = [
        User.email,
        User.name,
        User.hashed_password,
        User.is_admin,
        User.is_active,
    ]


admin.add_view(UserAdmin)


@app.get("/admin/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str | None = None):
    """Custom login page with SSO buttons.

    Args:
        request: The incoming request
        error: Optional error message to display

    Returns:
        HTMLResponse with the login page template
    """
    return LOGIN_PAGE_TEMPLATE.format(error=error or "")


@app.on_event("startup")
async def startup():
    """Initialize database on application startup."""
    await init_db()


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sqladmin_sso_auth_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
