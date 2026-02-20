from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
import datetime
from fastapi_sso import OpenID
import jwt
from contextlib import asynccontextmanager
from pfun_cma_model.admin.sso import setup_google_sso, get_logged_user
from pfun_common.settings import get_settings


@asynccontextmanager
async def lifespan(router: APIRouter):
    """Lifespan function to setup the Google SSO backend when the application starts."""
    global sso
    sso = setup_google_sso(
        redirect_host=get_settings().server_url, redirect_path="/auth/callback"
    )
    yield  # This allows the application to run until it is shutdown


router = APIRouter(lifespan=lifespan)


@router.get("/protected")
async def protected_endpoint(user: OpenID = Depends(get_logged_user)):
    """This endpoint will say hello to the logged user.
    If the user is not logged, it will return a 401 error from `get_logged_user`."""
    return {
        "message": f"You are very welcome, {user.email}!",
    }


@router.get("/auth/login")
async def login():
    """Redirect the user to the Google login page."""
    async with sso:
        return await sso.get_login_redirect()


@router.get("/auth/logout")
async def logout():
    """Forget the user's session."""
    response = RedirectResponse(url="/protected")
    response.delete_cookie(key="token")
    return response


@router.get("/auth/callback")
async def login_callback(request: Request):
    """Process login and redirect the user to the protected endpoint."""
    async with sso:
        openid = await sso.verify_and_process(request)

    if not openid:
        raise HTTPException(status_code=401, detail="Authentication failed")

    # Create a JWT with the user's OpenID
    expiration = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(
        days=1
    )
    token = jwt.encode(
        {"pld": openid.model_dump(), "exp": expiration, "sub": openid.id},
        key=get_settings().secret_key,
        algorithm="HS256",
    )
    response = RedirectResponse(url="/protected")
    response.set_cookie(
        key="token", value=token, expires=expiration
    )  # This cookie will make sure /protected knows the user
    return response
