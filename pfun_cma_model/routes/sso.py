"""This module defines the routes related to Single Sign-On (SSO) authentication, e.g. using Google SSO."""

from typing import Annotated
from pfun_common.settings import get_settings
from pfun_cma_model.admin.core import CryptContextDefaults, OIDAuthenticatedUser
from pfun_cma_model.admin.sso import setup_google_sso
from contextlib import asynccontextmanager
import jwt
from fastapi_sso import OpenID
import datetime
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import APIRouter, Depends, HTTPException, Request
import logging

logger = logging.getLogger(__name__)

logger.setLevel(level=logging.DEBUG if get_settings().debug is True else logging.INFO)

# This module defines the routes related to Single Sign-On (SSO) authentication, e.g. using Google SSO.

# Global SSO backend instance, initialized in the lifespan function below
sso = None


@asynccontextmanager
async def lifespan(router: APIRouter):
    """Lifespan function to setup the Google SSO backend when the application starts."""
    global sso
    sso = setup_google_sso(
        redirect_host=get_settings().ssl_server_host, redirect_path="/sso/auth/callback"
    )
    yield  # This allows the application to run until it is shutdown


#: The APIRouter for SSO routes, with a lifespan function to setup the SSO backend when the application starts.
router = APIRouter(lifespan=lifespan)


async def sso_protected_google(user: OIDAuthenticatedUser, **kwargs) -> dict:
    """Authenticates for Google SSO-protected endpoint."""
    if not isinstance(user, OpenID):
        raise HTTPException(
            status_code=403, detail="Failed to authenticate via Google SSO."
        )
    kwargs.update({"user": user})
    return kwargs


GoogleUser = Annotated[dict, Depends(sso_protected_google)]
#: Annotation for SSO (google) protected endpoint. Raises an error without user: OIDAuthenticatedUser.


@router.get("/protected")
async def protected_endpoint(request: Request, user: GoogleUser):
    """This endpoint will say hello to the logged user.
    If the user is not logged, it will return a 401 error from `get_logged_user`."""
    logging.debug(
        "Accessing protected endpoint. User: %s", user.email if user else "None"
    )
    response = HTMLResponse(
        content=f"<head><title>Protected Endpoint</title><meta charset='UTF-8'><meta http-equiv='Refresh' content='0;url=/admin/' /></head>"
        f"<body><h1>Hello, {user.email}!</h1><p>You have successfully accessed the protected endpoint.</p></body>"
    )
    return response


@router.get("/auth/login")
async def login():
    """Redirect the user to the Google login page."""
    global sso
    if sso is None:
        raise HTTPException(status_code=500, detail="SSO backend not initialized")
    async with sso:
        return await sso.get_login_redirect()


@router.get("/auth/logout")
async def logout(request: Request):
    """Forget the user's session."""
    response = RedirectResponse(
        url=request.base_url
    )  # Redirect to home page after logout
    response.delete_cookie(key="token")
    return response


@router.get("/auth/callback")
async def login_callback(request: Request):
    """Process login and redirect the user to the protected endpoint."""
    global sso
    if sso is None:
        # NOTE: this check is for the linter to behave.
        raise HTTPException(status_code=500, detail="SSO backend not initialized")
    async with sso:
        openid = await sso.verify_and_process(request)

    if not openid:
        raise HTTPException(status_code=401, detail="Authentication failed")

    # Create a JWT with the user's OpenID
    expiration = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(
        minutes=CryptContextDefaults.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    token = jwt.encode(
        {"pld": openid.model_dump(), "exp": expiration, "sub": openid.id},
        key=get_settings().secret_key,
        algorithm=CryptContextDefaults.ALGORITHM,
    )
    request.session.update(
        {"token": token}
    )  # Store the token in the session for future authentication
    # Redirect to user profile endpoint after login
    response = RedirectResponse(url="/user/")
    response.set_cookie(
        key="token", value=token, expires=expiration
    )  # This cookie will make sure /admin/ knows the user
    logging.debug(
        "Login successful for user: %s. Redirecting to protected endpoint.",
        openid.email,
    )
    return response
