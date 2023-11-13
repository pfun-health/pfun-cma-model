import json
import os

import requests
from flask import Flask, redirect, request
import pfun_path_helper as pph
from pfun_cma_model.dexcom_api.utils import get_creds
import ssl

app = Flask(__name__)


client_id, client_secret = get_creds()
redirect_uri = "https://127.0.0.1:5000/callback"


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
    print(response.json())
    if response.ok:
        access_token = response.json().get("access_token")
        return {"authorization": f"Bearer {access_token}", "access_token": access_token}
    return {
        "error": "Error: Unable to retrieve access token.",
        "status_code": response.status_code,
    }, response.status_code


if __name__ == "__main__":

    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    certs_dirpath = os.path.abspath(os.path.join(pph.get_lib_path('pfun_cma_model'), '..', 'certs'))
    context.load_cert_chain(os.path.join(certs_dirpath, 'cert.pem'), os.path.join(certs_dirpath,'key.pem'))

    app.run(host="127.0.0.1", port=5000, ssl_context=context, debug=True)
