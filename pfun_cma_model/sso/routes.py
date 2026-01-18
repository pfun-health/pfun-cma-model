import datetime  # to calculate expiration of the JWT
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.responses import RedirectResponse

# this is the part that adds the lock icon in the swaggerUI docs
from fastapi.security import APIKeyCookie

# monkey-patch SSO providers
import fastapi_sso.sso
import pfun_cma_model.sso.providers as pfun_providers
fastapi_sso.sso.__dict__.update(pfun_providers.__dict__)

# for validation, safe token (de-)serialization
from fastapi_sso.sso.base import OpenID
from jose import jwt


async def setup_sso_provider(provider_name: str = "orcid"):
    pass


# # used to sign JWTs, make sure it is really secret
# SECRET_KEY = "this-is-very-secret"
# CLIENT_ID = "your-client-id"  # your Google OAuth2 client ID
# CLIENT_SECRET = "your-client-secret"  # your Google OAuth2 client secret
# sso_provider = GoogleSSO(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
#                          redirect_uri="http://127.0.0.1:5000/auth/callback")

# async def get_logged_user(cookie: str = Security(APIKeyCookie(name="token"))) -> OpenID:
#     # Get the user's JWT stored in cookie 'token', parse it and return the user's OpenID.
#     pass
