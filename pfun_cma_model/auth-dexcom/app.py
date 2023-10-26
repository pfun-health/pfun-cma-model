import json
import os

import requests
from flask import Flask, redirect, request

app = Flask(__name__)


def get_creds():
    creds = json.loads(
        open(os.path.expanduser("~/.credentials/dexcom_pfun-app_glucose.json")).read()
    )
    os.environ["DEXCOM_CLIENT_ID"] = creds["client_id"]
    os.environ["DEXCOM_CLIENT_SECRET"] = creds["client_secret"]
    return creds["client_id"], creds["client_secret"]


client_id, client_secret = get_creds()
redirect_uri = "http://127.0.0.1:5000/callback"


@app.route("/")
def home():
    auth_url = f"https://api.dexcom.com/v2/oauth2/login?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=offline_access"
    return f'<a href="{auth_url}">Login with Dexcom</a>'


@app.route("/callback")
def callback():
    auth_code = request.args.get("code")
    token_url = "https://api.dexcom.com/v2/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": auth_code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    response = requests.post(token_url, headers=headers, data=data)
    if response.ok:
        access_token = response.json().get("access_token")
        return {"authorization": f"Bearer {access_token}", "access_token": access_token}
    return {
        "error": "Error: Unable to retrieve access token.",
        "status_code": response.status_code,
    }, response.status_code


if __name__ == "__main__":
    app.run(debug=True)
