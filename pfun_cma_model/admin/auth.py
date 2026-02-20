from sqladmin import Admin
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import RedirectResponse
from fastapi import Security, HTTPException
from fastapi.security import APIKeyCookie
from fastapi_sso.sso.base import OpenID
from jose import jwt
from pfun_cma_model.admin.models import *
from pfun_cma_model.admin.core import Base, engine, Session
from pfun_common.settings import get_settings


"""

TODO: Implement real authentication logic, e.g. validate username/password against database, check user permissions, etc.

_References:_

+ ref: https://tomasvotava.github.io/fastapi-sso/how-to-guides/use-with-fastapi-security/
+ ref: https://aminalaee.github.io/sqladmin/authentication/

"""


async def get_logged_user(cookie: str = Security(APIKeyCookie(name="token"))) -> OpenID:
    """Get user's JWT stored in cookie 'token', parse it and return the user's OpenID."""
    try:
        claims = jwt.decode(cookie, key=get_settings().secret_key, algorithms=["HS256"])
        return OpenID(**claims["pld"])
    except Exception as error:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        ) from error


class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        username, password = form["username"], form["password"]
        # Validate username/password credentials
        # And update session
        # TODO: Implement real authentication logic
        token = jwt.encode(
            {"username": username, "password": password},
            get_settings().secret_key,
            algorithm="HS256",
        )
        request.session.update({"token": token})
        return True

    async def logout(self, request: Request) -> bool:
        # Usually you'd want to clear the session
        # TODO: Implement real logout logic, e.g. invalidate token, clear session, etc.
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        token = request.session.get("token")
        if not token:
            return False
        # TODO: Implement real authentication logic, e.g. check token validity, user permissions, etc.
        # For demonstration, we just check if the token can be decoded with the secret key
        try:
            jwt.decode(token, key=get_settings().secret_key, algorithms=["HS256"])
        except Exception:
            return False
        return True


authentication_backend = AdminAuth(secret_key=get_settings().secret_key)
