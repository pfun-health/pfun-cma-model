"""
PFun CMA Model - Dexcom API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
import httpx
import os
from typing import Dict

router = APIRouter()

DEXCOM_API_BASE_URL = "https://sandbox-api.dexcom.com"
DEXCOM_CLIENT_ID = os.getenv("DEXCOM_API_CLIENT_ID")
DEXCOM_CLIENT_SECRET = os.getenv("DEXCOM_API_SECRET")

@router.get("/test")
async def test_dexcom_route():
    return {"message": "Dexcom router is working"}

@router.post("/token")
async def get_token(request: Request, payload: Dict):
    """
    Exchange authorization code for an access token.
    """
    code = payload.get("code")
    redirect_uri = payload.get("redirect_uri")

    if not code:
        raise HTTPException(status_code=400, detail="Authorization code is required")

    token_url = f"{DEXCOM_API_BASE_URL}/v2/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": DEXCOM_CLIENT_ID,
        "client_secret": DEXCOM_CLIENT_SECRET,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=data)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    token_data = response.json()
    request.session["dexcom_access_token"] = token_data["access_token"]
    request.session["dexcom_refresh_token"] = token_data["refresh_token"]

    return {"message": "Token acquired and stored in session"}


@router.get("/auth/callback")
async def auth_callback(request: Request):
    """
    Dexcom authorization callback.
    """
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not found in callback")

    # Store the code in the session to be retrieved by the frontend
    request.session["dexcom_auth_code"] = code

    # Redirect to the frontend, which will then call the /token endpoint
    return RedirectResponse(url="/demo/dexcom")


async def get_access_token(request: Request) -> str:
    """
    Dependency to get the access token from the session.
    """
    access_token = request.session.get("dexcom_access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return access_token


@router.get("/users/self/egvs")
async def get_egvs(request: Request, token: str = Depends(get_access_token)):
    """
    Proxy endpoint for fetching EGV data.
    """
    start_date = request.query_params.get("startDate")
    end_date = request.query_params.get("endDate")

    url = f"{DEXCOM_API_BASE_URL}/v3/users/self/egvs?startDate={start_date}&endDate={end_date}"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()


@router.get("/users/self/devices")
async def get_devices(token: str = Depends(get_access_token)):
    """
    Proxy endpoint for fetching device data.
    """
    url = f"{DEXCOM_API_BASE_URL}/v3/users/self/devices"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()
