"""
Dexcom API routes.
"""
import os
import logging
from fastapi import APIRouter, Request, Depends
from fastapi.responses import RedirectResponse
import httpx
import pkce

# Configure logging
logger = logging.getLogger(__name__)

# Create a new router
router = APIRouter()

# Dexcom API configuration
DEXCOM_CLIENT_ID = os.getenv("DEXCOM_CLIENT_ID")
DEXCOM_CLIENT_SECRET = os.getenv("DEXCOM_CLIENT_SECRET")
DEXCOM_REDIRECT_URI = os.getenv("DEXCOM_REDIRECT_URI")
DEXCOM_BASE_URL = "https://sandbox-api.dexcom.com" # or "https://api.dexcom.com" for production

@router.get("/dexcom/auth")
async def dexcom_auth(request: Request):
    """
    Redirects the user to the Dexcom login page to initiate the OAuth 2.0 flow.
    """
    code_verifier, code_challenge = pkce.generate_pkce_pair()
    request.session['code_verifier'] = code_verifier

    params = {
        "client_id": DEXCOM_CLIENT_ID,
        "redirect_uri": DEXCOM_REDIRECT_URI,
        "response_type": "code",
        "scope": "offline_access",
        "state": "some_random_state_string", # Should be a random string
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    auth_url = f"{DEXCOM_BASE_URL}/v2/oauth2/login?{httpx.URL(params).query.decode()}"
    return RedirectResponse(url=auth_url)

@router.get("/dexcom/auth/callback")
async def dexcom_auth_callback(request: Request):
    """
    Handles the callback from Dexcom after the user has authenticated.
    """
    code = request.query_params.get("code")
    code_verifier = request.session.pop('code_verifier', None)

    if not code or not code_verifier:
        return {"error": "Invalid request"}

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": DEXCOM_REDIRECT_URI,
        "code_verifier": code_verifier,
        "client_id": DEXCOM_CLIENT_ID,
        "client_secret": DEXCOM_CLIENT_SECRET,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{DEXCOM_BASE_URL}/v2/oauth2/token", data=data)

    if response.status_code != 200:
        return {"error": "Failed to fetch access token"}

    token_data = response.json()
    request.session['dexcom_access_token'] = token_data['access_token']
    request.session['dexcom_refresh_token'] = token_data['refresh_token']

    return RedirectResponse(url="/demo/dexcom")

@router.api_route("/dexcom/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def dexcom_api_proxy(request: Request, path: str):
    """
    A proxy for the Dexcom API.
    """
    access_token = request.session.get('dexcom_access_token')
    if not access_token:
        return {"error": "Not authenticated"}, 401

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": request.headers.get("Content-Type"),
    }

    url = f"{DEXCOM_BASE_URL}/{path}"

    async with httpx.AsyncClient() as client:
        if request.method == "GET":
            response = await client.get(url, headers=headers, params=request.query_params)
        elif request.method == "POST":
            response = await client.post(url, headers=headers, json=await request.json())
        # Add other methods as needed
        else:
            return {"error": "Method not supported"}, 405

    return response.json(), response.status_code
