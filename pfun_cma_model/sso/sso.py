import datetime  # to calculate expiration of the JWT
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyCookie  # this is the part that puts the lock icon to the docs
from fastapi_sso.sso.google import GoogleSSO  # pip install fastapi-sso
from fastapi_sso.sso.base import OpenID

from jose import jwt  # pip install python-jose[cryptography]
"""
SECRET_KEY = "this-is-very-secret"  # used to sign JWTs, make sure it is really secret
CLIENT_ID = "your-client-id"  # your Google OAuth2 client ID
CLIENT_SECRET = "your-client-secret"  # your Google OAuth2 client secret
sso_provider = GoogleSSO(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri="http://127.0.0.1:5000/auth/callback")async def get_logged_user(cookie: str = Security(APIKeyCookie(name="token"))) -> OpenID:
# Get the user's JWT stored in cookie 'token', parse it and return the user's OpenID.
"""

from fastapi_sso.sso.generic import create_provider

def createOrcidProvider():
    discovery = {
            "authorization_endpoint": "http://localhost:9090/auth",
            "token_endpoint": "http://localhost:9090/token",
            "userinfo_endpoint": "http://localhost:9090/me",
    }

    OrcidSSOProvider = create_provider(name="ORCiD", discovery_document=discovery)
    sso = OrcidSSOProvider(
        client_id="test",
        client_secret="secret",
        redirect_uri="http://localhost:8080/callback",
        allow_insecure_http=True
    )