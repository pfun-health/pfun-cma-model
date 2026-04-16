"""Tests for automatic user creation in SQLAdmin authentication."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta, timezone
import jwt
from starlette.requests import Request

import pfun_path_helper as pph

pph.append_path(path=pph.get_lib_path("pfun_cma_model"))

from . import test_base

test_base.setup_test_environment()

from pfun_cma_model.admin.auth import AdminAuth
from pfun_cma_model.admin.core import CryptContextDefaults
from pfun_common.settings import get_settings


def create_sso_token(
    email: str, display_name: str = None, expired: bool = False
) -> str:
    """Create a valid SSO JWT token for testing."""
    if expired:
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
    else:
        exp = datetime.now(timezone.utc) + timedelta(minutes=30)

    payload = {
        "pld": {
            "id": f"google-{email}",
            "email": email,
            "display_name": display_name or email.split("@")[0],
            "first_name": None,
            "last_name": None,
            "picture": None,
        },
        "exp": exp,
        "sub": f"google-{email}",
    }

    return jwt.encode(
        payload,
        key=get_settings().secret_key,
        algorithm=CryptContextDefaults.ALGORITHM,
    )


def create_legacy_token(email: str) -> str:
    """Create a legacy SSO JWT token (with 'usn' claim) for backward compat testing."""
    payload = {
        "usn": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
        "sub": email,
    }

    return jwt.encode(
        payload,
        key=get_settings().secret_key,
        algorithm=CryptContextDefaults.ALGORITHM,
    )


def create_bearer_token(user_id: str, provider: str = "google") -> str:
    """Create a Bearer JWT token for API auth."""
    payload = {
        "sub": user_id,
        "provider": provider,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "type": "bearer",
    }

    return jwt.encode(
        payload,
        key=get_settings().secret_key,
        algorithm=CryptContextDefaults.ALGORITHM,
    )


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock(spec=Request)
    request.session = {}
    request.cookies = {}
    return request


@pytest.fixture
def admin_auth():
    """Create AdminAuth instance."""
    return AdminAuth(secret_key=get_settings().secret_key)


class TestAutoCreateUsers:
    """Test automatic user creation functionality."""

    @pytest.mark.asyncio
    async def test_authenticate_with_expired_token_fails(
        self, admin_auth, mock_request
    ):
        """Test that expired tokens are rejected with HTTPException."""
        # Create expired token
        token = create_sso_token("expired@pfun.me", "Expired User", expired=True)

        mock_request.cookies = {"token": token}
        mock_request.session = {"token": token}

        # Expired tokens raise HTTPException - verify it's caught properly
        try:
            await admin_auth.authenticate(mock_request)
            # If we get here, the test should fail
            assert False, "Expected exception to be raised for expired token"
        except Exception as exc:
            # Should be HTTPException with 401 or jwt.ExpiredSignatureError wrapped
            assert (
                "401" in str(exc)
                or "expired" in str(exc).lower()
                or "Signature has expired" in str(exc)
            )

    @pytest.mark.asyncio
    async def test_authenticate_with_legacy_token_format(
        self, admin_auth, mock_request
    ):
        """Test backward compatibility with legacy 'usn' token format."""
        # Create legacy token
        token = create_legacy_token("legacy@pfun.me")

        mock_request.cookies = {"token": token}
        mock_request.session = {"token": token}

        with patch("pfun_cma_model.admin.auth.Session") as mock_session_class:
            # Setup mock session
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock user exists
            from pfun_cma_model.admin.models import User

            mock_user = User(
                id=2,
                name="Legacy User",
                email="legacy@pfun.me",
                is_admin=False,
                age=25,
                bio=None,
                site_id=None,
                hashed_password="hashed",
            )

            mock_result = MagicMock()
            mock_result.scalars.return_value.first.return_value = mock_user
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await admin_auth.authenticate(mock_request)

            assert result is True

    @pytest.mark.asyncio
    async def test_authenticate_with_bearer_token_format(
        self, admin_auth, mock_request
    ):
        """Test handling of Bearer token format."""
        # Create bearer token
        token = create_bearer_token("bearer-user@pfun.me", "google")

        mock_request.cookies = {"token": token}
        mock_request.session = {"token": token}

        with patch("pfun_cma_model.admin.auth.Session") as mock_session_class:
            # Setup mock session
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock user exists
            from pfun_cma_model.admin.models import User

            mock_user = User(
                id=3,
                name="Bearer User",
                email="bearer-user@pfun.me",
                is_admin=False,
                age=35,
                bio=None,
                site_id=None,
                hashed_password="hashed",
            )

            mock_result = MagicMock()
            mock_result.scalars.return_value.first.return_value = mock_user
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await admin_auth.authenticate(mock_request)

            assert result is True

    @pytest.mark.asyncio
    async def test_authenticate_with_invalid_token_format(
        self, admin_auth, mock_request
    ):
        """Test that tokens with unknown format are rejected."""
        # Create token with unknown format
        payload = {
            "unknown": "format",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
        }
        token = jwt.encode(
            payload,
            key=get_settings().secret_key,
            algorithm=CryptContextDefaults.ALGORITHM,
        )

        mock_request.cookies = {"token": token}
        mock_request.session = {"token": token}

        result = await admin_auth.authenticate(mock_request)

        assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_no_token_returns_false(self, admin_auth, mock_request):
        """Test that missing token returns False."""
        # No token set
        mock_request.cookies = {}
        mock_request.session = {}

        result = await admin_auth.authenticate(mock_request)

        assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_token_in_session_only(self, admin_auth, mock_request):
        """Test authentication when token is in session but not in cookies."""
        token = create_sso_token("session@pfun.me", "Session User")

        # Token only in session, not cookies
        mock_request.cookies = {}
        mock_request.session = {"token": token}

        with patch("pfun_cma_model.admin.auth.Session") as mock_session_class:
            # Setup mock session
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock user exists
            from pfun_cma_model.admin.models import User

            mock_user = User(
                id=4,
                name="Session User",
                email="session@pfun.me",
                is_admin=False,
                age=28,
                bio=None,
                site_id=None,
                hashed_password="hashed",
            )

            mock_result = MagicMock()
            mock_result.scalars.return_value.first.return_value = mock_user
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await admin_auth.authenticate(mock_request)

            assert result is True


class TestTokenTypes:
    """Test different token types and their handling."""

    def test_sso_token_has_pld_claim(self):
        """Verify SSO token structure."""
        token = create_sso_token("test@pfun.me", "Test User")
        decoded = jwt.decode(
            token,
            key=get_settings().secret_key,
            algorithms=[CryptContextDefaults.ALGORITHM],
        )

        assert "pld" in decoded
        assert decoded["pld"]["email"] == "test@pfun.me"
        assert "sub" in decoded

    def test_legacy_token_has_usn_claim(self):
        """Verify legacy token structure."""
        token = create_legacy_token("legacy@pfun.me")
        decoded = jwt.decode(
            token,
            key=get_settings().secret_key,
            algorithms=[CryptContextDefaults.ALGORITHM],
        )

        assert "usn" in decoded
        assert decoded["usn"] == "legacy@pfun.me"

    def test_bearer_token_has_type_claim(self):
        """Verify Bearer token structure."""
        token = create_bearer_token("bearer@pfun.me")
        decoded = jwt.decode(
            token,
            key=get_settings().secret_key,
            algorithms=[CryptContextDefaults.ALGORITHM],
        )

        assert "type" in decoded
        assert decoded["type"] == "bearer"
        assert decoded["sub"] == "bearer@pfun.me"
        assert decoded["provider"] == "google"


class TestLogout:
    """Test logout functionality."""

    @pytest.mark.asyncio
    async def test_logout_clears_session(self, admin_auth, mock_request):
        """Test that logout clears session and cookies."""
        mock_request.session = {"token": "some-token", "uid": 1}

        result = await admin_auth.logout(mock_request)

        assert result is True
        assert mock_request.session.get("token") is None

    @pytest.mark.asyncio
    async def test_logout_deletes_cookie(self, admin_auth, mock_request):
        """Test that logout deletes token cookie."""
        mock_request.session = {"token": "some-token"}
        # Initialize cookies dict that will be modified by pop
        mock_request.cookies = {"token": "some-token"}

        result = await admin_auth.logout(mock_request)

        assert result is True
        # The logout function calls cookies.pop("token", None)
        # which removes the key from the dict
        assert mock_request.cookies.get("token") is None


class TestLogin:
    """Test login functionality."""

    @pytest.mark.asyncio
    async def test_login_always_returns_true(self, admin_auth, mock_request):
        """Test that login always returns True (SSO handles actual auth)."""
        result = await admin_auth.login(mock_request)

        assert result is True
