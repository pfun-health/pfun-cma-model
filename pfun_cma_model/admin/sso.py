"""pfun_cma_model/admin/sso.py : SSO authentication backend for sqladmin."""

from fastapi import Security, HTTPException
from fastapi.security import APIKeyCookie
from fastapi_sso.sso.base import OpenID
from fastapi_sso.sso.google import GoogleSSO
from jose import jwt
from packages.pfun_common.pfun_common.settings import get_settings
import urllib.parse as urlparse


def setup_google_sso(
    redirect_host: str, redirect_path: str = "/auth/callback"
) -> GoogleSSO:
    """Setup the Google SSO authentication backend."""
    redirect_uri = str(urlparse.urljoin(redirect_host, redirect_path))
    return GoogleSSO(
        client_id=get_settings().google_cloud_client_id,
        client_secret=get_settings().google_cloud_client_secret,
        redirect_uri=redirect_uri,
    )


async def get_logged_user(cookie: str = Security(APIKeyCookie(name="token"))) -> OpenID:
    """Get user's JWT stored in cookie 'token', parse it and return the user's OpenID.

    This function can be used as a dependency in your admin views to get the logged-in user.
    """
    try:
        claims = jwt.decode(cookie, key=get_settings().secret_key, algorithms=["HS256"])
        return OpenID(**claims["pld"])
    except Exception as error:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        ) from error
