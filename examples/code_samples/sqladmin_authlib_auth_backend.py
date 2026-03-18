"""SQLAdmin SSO Auth Backend using authlib.

This example demonstrates integrating Google OAuth with sqladmin using authlib.
Note: This is a legacy example. For a more modern approach using fastapi-sso,
see sqladmin_sso_auth_backend.py

For setup instructions, see the fastapi-sso example file.
"""

from typing import Optional, Union

from authlib.integrations.starlette_client import OAuth
from sqladmin import Admin
from sqladmin.authentication import AuthenticationBackend
from starlette.applications import Starlette
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

app = Starlette()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here")

oauth = OAuth()
oauth.register(
    name="google",
    client_id="your-google-client-id",
    client_secret="your-google-client-secret",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile",
        "prompt": "select_account",
    },
)
google = oauth.create_client("google")


class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(
        self, request: Request
    ) -> Union[bool, Optional[RedirectResponse]]:
        user = request.session.get("user")
        if not user:
            redirect_uri = request.url_for("login_google")
            return await google.authorize_redirect(request, redirect_uri)
        return True


async def login_google(request: Request) -> Response:
    token = await google.authorize_access_token(request)
    user = token.get("userinfo")
    if user:
        request.session["user"] = user
    return RedirectResponse(request.url_for("admin:index"))


def setup_admin(engine: any) -> Admin:
    """Setup admin with SSO authentication.

    Args:
        engine: SQLAlchemy async engine

    Returns:
        Configured Admin instance
    """
    admin_instance = Admin(
        app=app, engine=engine, authentication_backend=AdminAuth("your-secret-key-here")
    )
    admin_instance.app.router.add_route("/auth/google", login_google)
    return admin_instance
